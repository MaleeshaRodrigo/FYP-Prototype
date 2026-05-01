"""
Image upload, DICOM parsing, and Azure/local storage helpers.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None

try:
    from azure.storage.blob import BlobClient, ContentSettings
except ImportError:  # pragma: no cover
    BlobClient = None
    ContentSettings = None

from app_config import load_app_config
from database import db, utc_now
from utils import image_to_png_bytes


@dataclass
class PreparedImage:
    image: Image.Image
    storage_bytes: bytes
    original_filename: str
    content_type: str
    image_format: str
    dicom_metadata: Optional[Dict[str, str]]


def _dicom_to_image(data: bytes) -> tuple[Image.Image, Dict[str, str]]:
    if pydicom is None:
        raise ValueError("DICOM support requires pydicom to be installed.")
    dataset = pydicom.dcmread(BytesIO(data))
    pixel_array = dataset.pixel_array.astype(np.float32)
    pixel_array -= pixel_array.min()
    pixel_array /= pixel_array.max() + 1e-8
    pixel_array = (pixel_array * 255).astype(np.uint8)
    image = Image.fromarray(pixel_array).convert("RGB")
    metadata = {
        "PatientID": str(getattr(dataset, "PatientID", ""))[:64],
        "StudyDate": str(getattr(dataset, "StudyDate", ""))[:32],
        "Modality": str(getattr(dataset, "Modality", ""))[:32],
        "Rows": str(getattr(dataset, "Rows", "")),
        "Columns": str(getattr(dataset, "Columns", "")),
    }
    return image, metadata


def prepare_uploaded_image(upload) -> PreparedImage:
    original_filename = upload.name
    raw = upload.getvalue()
    suffix = Path(original_filename).suffix.lower()
    content_type = getattr(upload, "type", "") or "application/octet-stream"

    if suffix in {".dcm", ".dicom"} or content_type == "application/dicom":
        image, metadata = _dicom_to_image(raw)
        return PreparedImage(
            image=image,
            storage_bytes=image_to_png_bytes(image),
            original_filename=original_filename,
            content_type="application/dicom",
            image_format="DICOM",
            dicom_metadata=metadata,
        )

    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Only JPEG, PNG, and DICOM files are supported.")
    image = Image.open(BytesIO(raw)).convert("RGB")
    return PreparedImage(
        image=image,
        storage_bytes=image_to_png_bytes(image),
        original_filename=original_filename,
        content_type=content_type,
        image_format=suffix.lstrip(".").upper().replace("JPG", "JPEG"),
        dicom_metadata=None,
    )


def _store_bytes(blob_path: str, data: bytes, content_type: str) -> str:
    config = load_app_config()
    if config.azure_storage_connection_string and BlobClient is not None:
        client = BlobClient.from_connection_string(
            config.azure_storage_connection_string,
            container_name=config.azure_image_container,
            blob_name=blob_path,
        )
        client.upload_blob(
            data,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type) if ContentSettings else None,
        )
        return "azure_blob"

    target = config.local_upload_dir / blob_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return "local_file"


def _load_bytes(blob_path: str, storage_backend: str) -> bytes:
    config = load_app_config()
    if storage_backend == "azure_blob":
        if not config.azure_storage_connection_string or BlobClient is None:
            raise RuntimeError("Azure Blob Storage is not configured.")
        client = BlobClient.from_connection_string(
            config.azure_storage_connection_string,
            container_name=config.azure_image_container,
            blob_name=blob_path,
        )
        return client.download_blob().readall()
    return (config.local_upload_dir / blob_path).read_bytes()


def save_uploaded_image(user_id: int, prepared: PreparedImage) -> int:
    blob_path = f"user-{user_id}/{uuid.uuid4().hex}.png"
    storage_backend = _store_bytes(blob_path, prepared.storage_bytes, "image/png")
    image_id = db.execute_returning_id(
        """
        INSERT INTO image_records (
            user_id, original_filename, content_type, blob_path, storage_backend,
            file_size, image_format, dicom_metadata, status, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
        """,
        (
            user_id,
            prepared.original_filename,
            prepared.content_type,
            blob_path,
            storage_backend,
            len(prepared.storage_bytes),
            prepared.image_format,
            json.dumps(prepared.dicom_metadata) if prepared.dicom_metadata else None,
            utc_now(),
        ),
    )
    db.audit(
        "image_uploaded",
        actor_user_id=user_id,
        target_resource=f"image:{image_id}",
        details={"filename": prepared.original_filename, "format": prepared.image_format, "storage": storage_backend},
    )
    return image_id


def list_user_images(user_id: int) -> list[Dict]:
    return db.fetch_all(
        "SELECT * FROM image_records WHERE user_id = ? AND status = 'active' ORDER BY created_at DESC",
        (user_id,),
    )


def get_image_for_user(image_id: int, user: Dict) -> Optional[Dict]:
    if user["role"] == "researcher":
        return db.fetch_one("SELECT * FROM image_records WHERE id = ? AND status = 'active'", (image_id,))
    return db.fetch_one(
        "SELECT * FROM image_records WHERE id = ? AND user_id = ? AND status = 'active'",
        (image_id, int(user["id"])),
    )


def load_image(record: Dict) -> Image.Image:
    return Image.open(BytesIO(_load_bytes(record["blob_path"], record["storage_backend"]))).convert("RGB")


def soft_delete_image(image_id: int, user_id: int) -> None:
    db.execute("UPDATE image_records SET status = 'deleted', deleted_at = ? WHERE id = ? AND user_id = ?", (utc_now(), image_id, user_id))
    db.audit("image_deleted", actor_user_id=user_id, target_resource=f"image:{image_id}")
