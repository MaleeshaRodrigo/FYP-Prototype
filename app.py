"""
Streamlit navigation shell for the HARE prototype.
"""

from __future__ import annotations

import streamlit as st


home_page = st.Page(
    "home_page.py",
    title="Home",
    icon="🩺",
    default=True,
)

technical_page = st.Page(
    "pages/1_Technical_Research.py",
    title="Technical Research",
    icon="🔬",
)

about_page = st.Page(
    "about_us_page.py",
    title="About Us",
    icon="👤",
)

navigation = st.navigation(
    {
        "HARE Prototype": [home_page, technical_page, about_page],
    }
)

navigation.run()
