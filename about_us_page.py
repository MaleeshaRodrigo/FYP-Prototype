"""
About Us page for the HARE prototype.
"""

from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="About Us",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("About Us")
st.caption("Project author information for the HARE melanoma screening prototype.")

hero_col, info_col = st.columns([0.9, 1.1], gap="large")

with hero_col:
    st.markdown("### Project Author")
    st.markdown(
        """
        **Mr. Maleesha Rodrigo**

        BEng in Software Engineering degree at the University of Westminster
        """
    )

with info_col:
    st.markdown("### Academic Details")
    st.markdown(
        """
        - University of Westminster ID: `w1902272`
        - IIT ID: `20212187`
        - Contact No: `+94767963545`
        - Email: `maleeshaachintha@gmail.com`
        """
    )

st.markdown("---")

st.markdown("### About The Project")
st.markdown(
    """
    HARE is a research-focused melanoma screening prototype developed as part of a software engineering degree project.
    The system aligns a Streamlit-based clinical screening workflow with thesis-driven adversarial robustness research,
    allowing non-technical users to access simple screening guidance while technical users can inspect metrics,
    architecture details, Grad-CAM visualisations, and attack simulations separately.
    """
)

st.markdown("### Contact")
st.markdown(
    """
    For academic, project, or demonstration inquiries:

    - Email: [maleeshaachintha@gmail.com](mailto:maleeshaachintha@gmail.com)
    - Phone: `+94767963545`
    """
)
