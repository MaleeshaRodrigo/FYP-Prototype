"""
Login and registration page for the HARE thesis-demo system.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import current_user, login, register_patient
from database import db


st.set_page_config(page_title="HARE Login", page_icon="🔐", layout="centered")

st.title("HARE Access")
st.caption("Sign in to use the melanoma screening support workflow.")

existing_user = current_user()
if existing_user:
    st.success(f"You are signed in as {existing_user['email']}.")
    st.stop()

researcher_count = db.fetch_one(
    "SELECT COUNT(*) AS count FROM users WHERE role = 'researcher' AND status = 'active'"
)
if not researcher_count or int(researcher_count["count"]) == 0:
    st.warning(
        "No active researcher account is configured. Set ADMIN_BOOTSTRAP_EMAIL and "
        "ADMIN_BOOTSTRAP_PASSWORD, then restart the app to create the first researcher."
    )

login_tab, register_tab = st.tabs(["Sign In", "Register"])

with login_tab:
    with st.form("login_form"):
        email = st.text_input("Email address")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        ok, message = login(email, password)
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

with register_tab:
    st.info("New patient accounts require researcher approval before system access.")
    with st.form("register_form"):
        email = st.text_input("Email address", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create patient account", use_container_width=True)

    if submitted:
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            ok, message = register_patient(email, password)
            if ok:
                st.success(message)
            else:
                st.error(message)
