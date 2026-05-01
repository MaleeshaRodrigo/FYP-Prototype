"""
Shared cached model loader for all Streamlit pages.
"""

from __future__ import annotations

import streamlit as st

from utils import load_model_runtime


@st.cache_resource(show_spinner=False)
def load_model():
    try:
        from blob_storage_utils import download_model_from_blob

        download_model_from_blob("stage2_best.pth")
        download_model_from_blob("hare_stage2_robust.pth")
    except RuntimeError:
        pass

    return load_model_runtime(".")
