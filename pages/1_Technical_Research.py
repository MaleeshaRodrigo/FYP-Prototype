"""
Technical and research-focused Streamlit page for HARE.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
import torch
from PIL import Image

from auth_utils import require_role
from database import db, utc_now
from image_storage import load_image
from model_service import load_model
from utils import (
    THESIS_ARCHITECTURE_POINTS,
    THESIS_FAILURE_MODE_POINTS,
    THESIS_METRICS_ROWS,
    GradCAM,
    denormalize_image,
    fgsm_attack,
    image_from_upload,
    image_to_png_bytes,
    overlay_heatmap,
    pgd_attack,
    predict,
    prepare_image,
)


st.set_page_config(
    page_title="HARE Technical Research",
    page_icon="🔬",
    layout="wide",
)


SESSION_IMAGE_KEY = "hare_active_image_bytes"
SESSION_SOURCE_KEY = "hare_active_image_source"
SESSION_IMAGE_ID_KEY = "hare_active_image_id"


user = require_role("researcher")


def store_active_image(image: Image.Image, source_label: str, image_id: Optional[int] = None) -> None:
    st.session_state[SESSION_IMAGE_KEY] = image_to_png_bytes(image)
    st.session_state[SESSION_SOURCE_KEY] = source_label
    if image_id is None:
        st.session_state.pop(SESSION_IMAGE_ID_KEY, None)
    else:
        st.session_state[SESSION_IMAGE_ID_KEY] = image_id


def read_active_image() -> Tuple[Optional[Image.Image], Optional[str], Optional[int]]:
    image_bytes = st.session_state.get(SESSION_IMAGE_KEY)
    if not image_bytes:
        return None, None, None
    return (
        Image.open(BytesIO(image_bytes)).convert("RGB"),
        st.session_state.get(SESSION_SOURCE_KEY),
        st.session_state.get(SESSION_IMAGE_ID_KEY),
    )


def resolve_demo_image() -> Tuple[Optional[Image.Image], Optional[str], Optional[int]]:
    existing_image, existing_source, existing_image_id = read_active_image()
    if existing_image is not None:
        st.info(f"Using the latest image from the screening workflow: {existing_source}. You can replace it below.")

    stored_col, upload_col, camera_col = st.columns(3)
    selected_image = existing_image
    source_label = existing_source
    image_id = existing_image_id

    with stored_col:
        records = db.fetch_all("SELECT * FROM image_records WHERE status = 'active' ORDER BY created_at DESC LIMIT 100")
        if records:
            labels = [f"{row['original_filename']} | user {row['user_id']} | ID {row['id']}" for row in records]
            selected = st.selectbox("Use stored image", ["No stored image"] + labels)
            if selected != "No stored image":
                index = labels.index(selected)
                record = records[index]
                selected_image = load_image(record)
                source_label = f"Stored image ID {record['id']}"
                image_id = int(record["id"])
        else:
            st.info("No stored images are available yet.")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a technical demo image",
            type=["jpg", "jpeg", "png"],
            key="technical_upload",
        )
        if uploaded_file is not None:
            selected_image = image_from_upload(uploaded_file)
            source_label = "Technical upload"
            image_id = None

    with camera_col:
        st.caption(
            "Camera access is optional and only requested after you enable capture for this session."
        )
        camera_enabled = st.checkbox(
            "Enable camera for demo capture",
            key="technical_camera_consent",
            help="Uploaded images work without camera permission.",
        )
        if camera_enabled:
            camera_file = st.camera_input(
                "Capture a technical demo image",
                key="technical_camera",
            )
            if camera_file is not None:
                selected_image = image_from_upload(camera_file)
                source_label = "Technical camera capture"
                image_id = None
        else:
            st.info("Camera capture is off. Upload an image instead, or enable camera capture when needed.")

    if selected_image is not None and source_label is not None:
        store_active_image(selected_image, source_label, image_id)

    return selected_image, source_label, image_id


st.title("HARE Technical Research View")
st.caption(
    "Architecture notes, thesis metrics, Grad-CAM, and single-image attack simulation for the "
    "Hybrid Adversarially Robust Ensemble."
)

try:
    model, device, settings, weight_message = load_model()
    st.success(weight_message)
except Exception as exc:
    st.error(f"Model loading failed: {exc}")
    st.stop()

runtime_cols = st.columns(4)
runtime_cols[0].metric("Stage 1 clean AUC", "0.9001")
runtime_cols[1].metric("Stage 2 clean AUC", "0.8856")
runtime_cols[2].metric("GA threshold theta", f"{settings.ga_theta:.4f}")
runtime_cols[3].metric("Adversarial MEL sensitivity", "0.7204")

with st.expander("Active runtime configuration", expanded=False):
    st.markdown(
        f"""
        - Active checkpoint: `{settings.active_checkpoint}`
        - GA alpha: `{settings.ga_alpha:.4f}`
        - GA tau: `{settings.ga_tau:.4f}`
        - GA theta: `{settings.ga_theta:.4f}`
        - White-box thesis evaluation: `PGD-20`, `epsilon = 0.03`
        """
    )

metrics_col, architecture_col = st.columns([1.1, 0.9], gap="large")

with metrics_col:
    st.markdown("### Thesis Evaluation Summary")
    st.table(THESIS_METRICS_ROWS)
    st.markdown(
        "The technical page reports the thesis operating point directly: GA-calibrated screening "
        "with aggressive melanoma sensitivity preservation and transparent specificity loss."
    )

with architecture_col:
    st.markdown("### Architecture Snapshot")
    for point in THESIS_ARCHITECTURE_POINTS:
        st.markdown(f"- {point}")

    st.markdown("### Failure Mode Interpretation")
    for point in THESIS_FAILURE_MODE_POINTS:
        st.markdown(f"- {point}")

st.markdown("---")
st.markdown("### Single-Image Technical Demo")
st.info(
    "This section is an interactive single-image demonstration. It is not the full dataset-level "
    "PGD-20 evaluation reported in the thesis."
)

image, source, image_id = resolve_demo_image()
if image is None:
    st.stop()

image_tensor = prepare_image(image, device)
prediction = predict(model, image_tensor, settings)

demo_left, demo_right = st.columns([0.9, 1.1], gap="large")

with demo_left:
    st.image(image, use_container_width=True)
    if source:
        st.caption(f"Input source: {source}")

with demo_right:
    st.markdown("#### GA-Calibrated Screening Output")
    st.markdown(f"**Screening label:** {prediction.summary}")
    st.metric("Melanoma probability", f"{prediction.melanoma_probability * 100:.1f}%")
    st.metric("Distance from GA threshold", f"{(prediction.melanoma_probability - settings.ga_theta) * 100:+.1f} pp")
    st.markdown(
        f"""
        - CNN branch MEL probability: `{prediction.branch_probabilities["cnn"][1] * 100:.1f}%`
        - ViT branch MEL probability: `{prediction.branch_probabilities["vit"][1] * 100:.1f}%`
        - Fusion-head MEL probability: `{prediction.fusion_probabilities[1] * 100:.1f}%`
        - GA late-fusion MEL probability: `{prediction.late_fusion_probabilities[1] * 100:.1f}%`
        """
    )

st.markdown("---")

cam_col, attack_col = st.columns(2, gap="large")

with cam_col:
    st.markdown("### Grad-CAM")
    try:
        grad_cam = GradCAM(model)
        heatmap, dispersion_score = grad_cam.generate(image_tensor, prediction.predicted_index)
        overlay = overlay_heatmap(image, heatmap, alpha=0.45)
        st.image(overlay, use_container_width=True)
        st.caption("Red regions show stronger contribution from the ResNet branch feature map.")
        st.metric("Attention concentration score", f"{dispersion_score:.3f}")
    except Exception as exc:
        st.error(f"Grad-CAM generation failed: {exc}")

with attack_col:
    st.markdown("### Attack Simulation")
    attack_name = st.selectbox("Attack type", ["FGSM", "PGD-20"], index=1)
    epsilon = st.slider("Epsilon", min_value=0.0, max_value=0.06, value=0.03, step=0.005)
    label_tensor = torch.tensor([prediction.predicted_index], device=device)

    if st.button("Run attack simulation", use_container_width=True):
        with st.spinner("Generating adversarial example..."):
            if attack_name == "FGSM":
                adv_tensor = fgsm_attack(model, image_tensor, label_tensor, epsilon=epsilon)
            else:
                adv_tensor = pgd_attack(
                    model,
                    image_tensor,
                    label_tensor,
                    epsilon=epsilon,
                    alpha=max(epsilon / 3.0, 1e-4),
                    steps=20,
                )

        adv_prediction = predict(model, adv_tensor, settings)
        adv_image = Image.fromarray((denormalize_image(adv_tensor) * 255).astype("uint8"))

        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("#### Original")
            st.image(image, use_container_width=True)
            st.markdown(f"**Label:** {prediction.summary}")
            st.metric("MEL probability", f"{prediction.melanoma_probability * 100:.1f}%")

        with after_col:
            st.markdown("#### Attacked")
            st.image(adv_image, use_container_width=True)
            st.markdown(f"**Label:** {adv_prediction.summary}")
            st.metric("MEL probability", f"{adv_prediction.melanoma_probability * 100:.1f}%")

        if adv_prediction.predicted_label == prediction.predicted_label:
            st.success(
                "The single-image screening label stayed unchanged under the selected attack settings."
            )
        else:
            st.warning(
                "The single-image screening label changed under the selected attack settings."
            )

        st.caption(
            "This interactive demo uses the current image and the model's present prediction as the attack target. "
            "The thesis metrics come from held-out ISIC 2019 evaluation, not from this single example."
        )
        if image_id is not None:
            simulation_id = db.execute_returning_id(
                """
                INSERT INTO attack_simulations (
                    image_id, user_id, attack_type, epsilon, before_label,
                    after_label, changed, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    int(user["id"]),
                    attack_name,
                    epsilon,
                    prediction.predicted_label,
                    adv_prediction.predicted_label,
                    adv_prediction.predicted_label != prediction.predicted_label,
                    utc_now(),
                ),
            )
            db.audit(
                "attack_simulation",
                actor_user_id=int(user["id"]),
                target_resource=f"attack_simulation:{simulation_id}",
                details={"image_id": image_id, "attack": attack_name, "epsilon": epsilon},
            )
