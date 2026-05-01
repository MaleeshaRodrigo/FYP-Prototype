"""
Robust analysis report for a stored image.
"""

from __future__ import annotations

import streamlit as st

from analysis_service import run_robust_analysis
from auth_utils import require_login
from image_storage import get_image_for_user, list_user_images, load_image
from model_service import load_model
from utils import prepare_image, risk_banner


st.set_page_config(page_title="HARE Analysis Report", page_icon="📋", layout="wide")

user = require_login()

st.title("Analysis Report")
st.caption("Run the CNN/ViT model and PGD-10 robustness check on a stored image.")

images = list_user_images(int(user["id"]))
if not images:
    st.info("Upload an image in Image History before requesting analysis.")
    st.stop()

selected_image_id = st.session_state.get("selected_image_id") or int(images[0]["id"])
option_labels = [f"{row['original_filename']} | ID {row['id']}" for row in images]
option_ids = [int(row["id"]) for row in images]
selected_index = option_ids.index(int(selected_image_id)) if int(selected_image_id) in option_ids else 0
selected_label = st.selectbox("Selected image", option_labels, index=selected_index)
image_id = option_ids[option_labels.index(selected_label)]
st.session_state["selected_image_id"] = image_id

record = get_image_for_user(image_id, user)
if record is None:
    st.error("Selected image is unavailable or you do not have access.")
    st.stop()

image = load_image(record)

left_col, right_col = st.columns([1.0, 1.2], gap="large")
with left_col:
    st.markdown("### Image")
    st.image(image, use_container_width=True)
    st.caption(f"Source: {record['original_filename']} | Format: {record['image_format']}")

with right_col:
    st.markdown("### Robust Analysis")
    st.info("The final decision combines classification with a PGD-10 robustness check.")
    if st.button("Run robust analysis", type="primary", use_container_width=True):
        try:
            model, device, settings, _weight_message = load_model()
            image_tensor = prepare_image(image, device)
            with st.spinner("Running classification and PGD-10 robustness check..."):
                prediction, adversarial_prediction, robustness_status, result_id = run_robust_analysis(
                    model,
                    device,
                    settings,
                    image_tensor,
                    image_id,
                    int(user["id"]),
                )

            banner_text, banner_style = risk_banner(prediction)
            if banner_style == "error":
                st.error(banner_text)
            else:
                st.success(banner_text)

            st.markdown(f"**Final robust decision:** {prediction.summary} ({robustness_status})")
            st.metric("Melanoma probability", f"{prediction.melanoma_probability * 100:.1f}%")
            st.metric("Confidence", f"{prediction.screening_confidence * 100:.1f}%")
            st.metric("GA threshold", f"{prediction.threshold:.4f}")
            st.caption(f"Analysis ID: {result_id}; PGD-10 label: {adversarial_prediction.summary}")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
