"""
Authentication and role helpers for the Streamlit HARE app.
"""

from __future__ import annotations

import secrets
import string
from typing import Dict, Optional, Tuple

import streamlit as st
from passlib.context import CryptContext

from app_config import load_app_config
from database import db, utc_now


SESSION_USER_ID = "hare_user_id"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return pwd_context.verify(password, password_hash)
    except Exception:
        return False


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user_by_email(email: str) -> Optional[Dict]:
    return db.fetch_one("SELECT * FROM users WHERE email = ?", (normalize_email(email),))


def get_user_by_id(user_id: int) -> Optional[Dict]:
    return db.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))


def current_user() -> Optional[Dict]:
    user_id = st.session_state.get(SESSION_USER_ID)
    if not user_id:
        return None
    user = get_user_by_id(int(user_id))
    if user is None or user["status"] in {"deleted", "disabled"}:
        st.session_state.pop(SESSION_USER_ID, None)
        return None
    return user


def login(email: str, password: str) -> Tuple[bool, str]:
    user = get_user_by_email(email)
    actor_id = int(user["id"]) if user else None
    if not user or not verify_password(password, user["password_hash"]):
        db.audit("login_failed", actor_user_id=actor_id, success=False, details={"email": normalize_email(email)})
        return False, "Invalid email or password."
    if user["status"] == "pending":
        db.audit("login_blocked_pending", actor_user_id=actor_id, success=False)
        return False, "Your account is pending researcher approval."
    if user["status"] != "active":
        db.audit("login_blocked_inactive", actor_user_id=actor_id, success=False, details={"status": user["status"]})
        return False, "This account is not active."
    st.session_state[SESSION_USER_ID] = int(user["id"])
    db.audit("login_success", actor_user_id=int(user["id"]))
    return True, "Signed in."


def logout() -> None:
    user = current_user()
    if user:
        db.audit("logout", actor_user_id=int(user["id"]))
    st.session_state.pop(SESSION_USER_ID, None)


def register_patient(email: str, password: str) -> Tuple[bool, str]:
    email = normalize_email(email)
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if get_user_by_email(email):
        db.audit("registration_duplicate", success=False, details={"email": email})
        return False, "An account with that email already exists."
    user_id = db.execute_returning_id(
        """
        INSERT INTO users (email, password_hash, role, status, created_at)
        VALUES (?, ?, 'patient', 'pending', ?)
        """,
        (email, hash_password(password), utc_now()),
    )
    db.audit("registration_created", actor_user_id=user_id, target_resource=f"user:{user_id}", details={"role": "patient"})
    return True, "Registration submitted. A researcher must approve your account before login."


def bootstrap_researcher() -> None:
    config = load_app_config()
    if not config.admin_bootstrap_email or not config.admin_bootstrap_password:
        return
    email = normalize_email(config.admin_bootstrap_email)
    if get_user_by_email(email):
        return
    user_id = db.execute_returning_id(
        """
        INSERT INTO users (email, password_hash, role, status, created_at, approved_at)
        VALUES (?, ?, 'researcher', 'active', ?, ?)
        """,
        (email, hash_password(config.admin_bootstrap_password), utc_now(), utc_now()),
    )
    db.audit("researcher_bootstrapped", actor_user_id=user_id, target_resource=f"user:{user_id}")


def require_login() -> Dict:
    user = current_user()
    if user is None:
        st.warning("Please sign in to access this page.")
        st.stop()
    if user["status"] == "pending":
        st.info("Your account is pending researcher approval.")
        st.stop()
    return user


def require_role(role: str) -> Dict:
    user = require_login()
    if user["role"] != role:
        db.audit("rbac_denied", actor_user_id=int(user["id"]), success=False, details={"required_role": role})
        st.error("You do not have permission to access this page.")
        st.stop()
    return user


def set_user_status(target_user_id: int, status: str, actor_user_id: int) -> None:
    approved_at = utc_now() if status == "active" else None
    db.execute(
        "UPDATE users SET status = ?, approved_at = COALESCE(?, approved_at), approved_by = COALESCE(?, approved_by) WHERE id = ?",
        (status, approved_at, actor_user_id if status == "active" else None, target_user_id),
    )
    db.audit(
        "user_status_changed",
        actor_user_id=actor_user_id,
        target_resource=f"user:{target_user_id}",
        details={"status": status},
    )


def reset_password(target_user_id: int, actor_user_id: int) -> str:
    alphabet = string.ascii_letters + string.digits
    temporary_password = "".join(secrets.choice(alphabet) for _ in range(14))
    db.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(temporary_password), target_user_id))
    db.audit("password_reset", actor_user_id=actor_user_id, target_resource=f"user:{target_user_id}")
    return temporary_password
