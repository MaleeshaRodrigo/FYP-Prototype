"""
Authenticated landing page for the HARE thesis-demo workflow.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import require_login
from database import db


st.set_page_config(
    page_title="HARE Home",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

user = require_login()

st.title("HARE Melanoma Screening Support")
st.caption("Research prototype with authenticated image history, robust analysis, and audit logging.")

st.info(
    "This app provides screening support only. It does not diagnose melanoma, prescribe treatment, "
    "or replace dermatologist review."
)

image_count = db.fetch_one(
    "SELECT COUNT(*) AS count FROM image_records WHERE user_id = ? AND status = 'active'",
    (int(user["id"]),),
)
analysis_count = db.fetch_one(
    "SELECT COUNT(*) AS count FROM analysis_results WHERE user_id = ?",
    (int(user["id"]),),
)

cols = st.columns(3)
cols[0].metric("Active images", image_count["count"] if image_count else 0)
cols[1].metric("Analysis reports", analysis_count["count"] if analysis_count else 0)
cols[2].metric("Account status", user["status"].title())

st.markdown("### Workflow")
workflow_cols = st.columns(3)
with workflow_cols[0]:
    st.markdown("**1. Upload**")
    st.write("Add JPEG, PNG, or DICOM images through Image History.")
with workflow_cols[1]:
    st.markdown("**2. Select**")
    st.write("Choose a stored image for the robust analysis report.")
with workflow_cols[2]:
    st.markdown("**3. Review**")
    st.write("View classification, confidence, GA threshold, and PGD-10 robustness status.")

with st.expander("Important use notes"):
    st.markdown(
        """
        - This is a thesis research prototype, not a regulated medical device.
        - Protected data access is role-controlled inside the application.
        - Camera capture remains optional and consent-gated on research/demo pages only.
        - No EHR integration, treatment recommendation, prescription, or medical advice is provided.
        """
    )
