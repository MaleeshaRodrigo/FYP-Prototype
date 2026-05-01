"""
Researcher administrative interface for user management.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import require_role, reset_password, set_user_status
from database import db


st.set_page_config(page_title="HARE Researcher Admin", page_icon="👤", layout="wide")

user = require_role("researcher")

st.title("Researcher Admin")
st.caption("Approve accounts, disable/delete users, and issue temporary passwords.")

users = db.fetch_all("SELECT id, email, role, status, created_at, approved_at FROM users ORDER BY created_at DESC")
if not users:
    st.info("No users found.")
    st.stop()

st.dataframe(users, use_container_width=True, hide_index=True)

st.markdown("### Manage User")
target_id = st.number_input("User ID", min_value=1, step=1)
action = st.selectbox("Action", ["Approve/activate", "Disable", "Mark deleted", "Reset password"])

if st.button("Apply action", type="primary"):
    if int(target_id) == int(user["id"]) and action in {"Disable", "Mark deleted"}:
        st.error("You cannot disable or delete your own active researcher account.")
    elif action == "Approve/activate":
        set_user_status(int(target_id), "active", int(user["id"]))
        st.success("User activated.")
    elif action == "Disable":
        set_user_status(int(target_id), "disabled", int(user["id"]))
        st.success("User disabled.")
    elif action == "Mark deleted":
        set_user_status(int(target_id), "deleted", int(user["id"]))
        st.success("User marked deleted.")
    else:
        temporary_password = reset_password(int(target_id), int(user["id"]))
        st.success("Temporary password generated. Show this once to the user.")
        st.code(temporary_password)
