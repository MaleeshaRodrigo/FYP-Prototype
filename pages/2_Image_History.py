"""
Authenticated image upload and history management.
"""

from __future__ import annotations

import streamlit as st

from auth_utils import require_login
from image_storage import list_user_images, prepare_uploaded_image, save_uploaded_image, soft_delete_image


st.set_page_config(page_title="HARE Image History", page_icon="🖼️", layout="wide")

user = require_login()

st.title("Image History")
st.caption("Upload dermoscopy images and manage records owned by your account.")

with st.expander("Upload a new image", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload JPEG, PNG, or DICOM image",
        type=["jpg", "jpeg", "png", "dcm", "dicom"],
        help="DICOM files are converted to a model-ready RGB image for analysis.",
    )
    if uploaded_file is not None:
        try:
            prepared = prepare_uploaded_image(uploaded_file)
            st.image(prepared.image, caption=f"Preview: {prepared.original_filename}", width=320)
            if st.button("Save image to history", type="primary"):
                image_id = save_uploaded_image(int(user["id"]), prepared)
                st.session_state["selected_image_id"] = image_id
                st.success("Image saved. It is now selected for analysis.")
                st.rerun()
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

images = list_user_images(int(user["id"]))
if not images:
    st.info("No uploaded images yet.")
    st.stop()

st.markdown("### Your Uploaded Images")
for record in images:
    with st.container(border=True):
        cols = st.columns([2, 1, 1, 1])
        cols[0].markdown(f"**{record['original_filename']}**")
        cols[0].caption(f"Uploaded: {record['created_at']} | Format: {record['image_format']}")
        cols[1].metric("Size", f"{record['file_size'] / 1024:.1f} KB")
        if cols[2].button("Select for analysis", key=f"select_{record['id']}"):
            st.session_state["selected_image_id"] = int(record["id"])
            st.success("Selected for analysis.")
        if cols[3].button("Delete", key=f"delete_{record['id']}"):
            soft_delete_image(int(record["id"]), int(user["id"]))
            if st.session_state.get("selected_image_id") == int(record["id"]):
                st.session_state.pop("selected_image_id", None)
            st.success("Image deleted from active history.")
            st.rerun()
