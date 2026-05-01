"""
Public screening-first Streamlit interface for HARE.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
from PIL import Image

from utils import (
    image_from_upload,
    image_to_png_bytes,
    load_model_runtime,
    predict,
    prepare_image,
    risk_banner,
)


st.set_page_config(
    page_title="HARE Home",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


SESSION_IMAGE_KEY = "hare_active_image_bytes"
SESSION_SOURCE_KEY = "hare_active_image_source"


@st.cache_resource(show_spinner=False)
def load_model():
    # Attempt to download models from Azure Blob Storage (if configured)
    try:
        from blob_storage_utils import download_model_from_blob
        download_model_from_blob("stage2_best.pth")
        download_model_from_blob("hare_stage2_robust.pth")
    except RuntimeError:
        # Blob storage not configured or download failed
        # Models should exist locally for local development
        pass
    
    return load_model_runtime(".")


def store_active_image(image: Image.Image, source_label: str) -> None:
    st.session_state[SESSION_IMAGE_KEY] = image_to_png_bytes(image)
    st.session_state[SESSION_SOURCE_KEY] = source_label


def read_active_image() -> Tuple[Optional[Image.Image], Optional[str]]:
    image_bytes = st.session_state.get(SESSION_IMAGE_KEY)
    if not image_bytes:
        return None, None
    return Image.open(BytesIO(image_bytes)).convert("RGB"), st.session_state.get(SESSION_SOURCE_KEY)


def resolve_input_image() -> Tuple[Optional[Image.Image], Optional[str]]:
    upload_tab, camera_tab = st.tabs(["Upload Image", "Take Photo"])
    selected_image: Optional[Image.Image] = None
    source_label: Optional[str] = None

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload a lesion image",
            type=["jpg", "jpeg", "png"],
            help="Best results come from clear, well-lit close-up lesion images.",
        )
        if uploaded_file is not None:
            selected_image = image_from_upload(uploaded_file)
            source_label = "Uploaded image"

    with camera_tab:
        camera_file = st.camera_input(
            "Capture a lesion image",
            help="Camera captures are supported for screening exploration, but the thesis metrics were validated on dermoscopic images.",
        )
        if camera_file is not None:
            selected_image = image_from_upload(camera_file)
            source_label = "Camera capture"

    if selected_image is not None and source_label is not None:
        store_active_image(selected_image, source_label)
        return selected_image, source_label

    previous_image, previous_source = read_active_image()
    return previous_image, previous_source


st.title("HARE Melanoma Screening Support")
st.caption(
    "Research prototype for clinician and patient-friendly melanoma screening support. "
    "Technical metrics, Grad-CAM, and attack simulation live on the Technical Research page."
)

st.info(
    "This app provides screening support only. It does not diagnose melanoma and it was evaluated "
    "on dermoscopic ISIC 2019 images rather than a separate consumer phone-photo validation set."
)

try:
    model, device, settings, weight_message = load_model()
    st.success(weight_message)
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

st.markdown("### Start With an Image")
image, source = resolve_input_image()

if image is None:
    st.markdown(
        """
        Use either option above to begin:

        - Upload a clear skin-lesion image
        - Take a photo directly in the browser

        The screening page keeps the explanation simple. The separate Technical Research page includes
        thesis metrics, architecture details, Grad-CAM, and adversarial attack simulation.
        """
    )
    st.stop()

image_tensor = prepare_image(image, device)
prediction = predict(model, image_tensor, settings)
banner_text, banner_style = risk_banner(prediction)

left_col, right_col = st.columns([1.1, 1.0], gap="large")

with left_col:
    st.markdown("### Image Review")
    st.image(image, use_container_width=True)
    if source:
        st.caption(f"Input source: {source}")

with right_col:
    st.markdown("### Screening Result")
    if banner_style == "error":
        st.error(banner_text)
    else:
        st.success(banner_text)

    st.markdown(f"**Screening summary:** {prediction.summary}")
    st.metric("Melanoma screening probability", f"{prediction.melanoma_probability * 100:.1f}%")
    st.write(prediction.recommendation)

st.markdown("---")

next_step_col, guide_col = st.columns(2, gap="large")

with next_step_col:
    st.markdown("### Suggested Next Step")
    if prediction.predicted_label == "MEL":
        st.markdown(
            """
            - Arrange prompt dermatologist or clinician review.
            - Compare against dermoscopic examination if available.
            - Do not use this app as a final diagnosis or treatment decision.
            """
        )
    else:
        st.markdown(
            """
            - Continue routine clinical review.
            - Reassess if the lesion is evolving, irregular, symptomatic, or clinically suspicious.
            - Escalate to specialist review whenever history or examination warrants it.
            """
        )

with guide_col:
    st.markdown("### How This Helps")
    st.markdown(
        """
        - Converts the uploaded or captured image into a melanoma screening signal.
        - Prioritizes early melanoma detection in line with the thesis screening objective.
        - Keeps technical attack analysis and model internals off the main page so non-technical users can focus on action-oriented guidance.
        """
    )

with st.expander("Important use notes"):
    st.markdown(
        """
        - This is a research prototype, not a regulated medical device.
        - The thesis evaluation used dermoscopic ISIC 2019 images.
        - Camera captures are available for usability, but they should not be interpreted as separately validated consumer-photo performance.
        - The Technical Research page contains the thesis operating point, architecture, and attack simulation details.
        """
    )
