"""
Streamlit navigation shell for the HARE prototype.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import bootstrap_researcher, current_user, logout
from database import initialize_database


initialize_database()
bootstrap_researcher()


login_page = st.Page(
    "pages/0_Login_Register.py",
    title="Login / Register",
    icon="🔐",
    default=True,
)

home_page = st.Page(
    "home_page.py",
    title="Home",
    icon="🩺",
)

image_history_page = st.Page(
    "pages/2_Image_History.py",
    title="Image History",
    icon="🖼️",
)

analysis_page = st.Page(
    "pages/3_Analysis_Report.py",
    title="Analysis Report",
    icon="📋",
)

technical_page = st.Page(
    "pages/1_Technical_Research.py",
    title="Technical Research",
    icon="🔬",
)

admin_page = st.Page(
    "pages/4_Researcher_Admin.py",
    title="Researcher Admin",
    icon="👤",
)

audit_page = st.Page(
    "pages/5_Audit_Log.py",
    title="Audit Log",
    icon="🧾",
)

about_page = st.Page(
    "about_us_page.py",
    title="About Us",
    icon="ℹ️",
)

user = current_user()
if user:
    with st.sidebar:
        st.caption(f"Signed in as {user['email']}")
        st.caption(f"Role: {user['role']}")
        if st.button("Sign out", use_container_width=True):
            logout()
            st.rerun()

    if user["role"] == "researcher":
        pages = {
            "HARE Prototype": [home_page, image_history_page, analysis_page, about_page],
            "Researcher": [technical_page, admin_page, audit_page],
        }
    else:
        pages = {
            "HARE Prototype": [home_page, image_history_page, analysis_page, about_page],
        }
else:
    pages = {
        "Access": [login_page, about_page],
    }

navigation = st.navigation(pages)
navigation.run()
