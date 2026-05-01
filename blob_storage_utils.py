"""
Azure Blob Storage utilities for downloading HARE models dynamically.

This module enables downloading pre-trained model weights from Azure Blob Storage
instead of bundling them in the Docker image, reducing image size significantly.
"""

from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

try:
    from azure.storage.blob import BlobClient
except ImportError:
    BlobClient = None


def get_blob_connection_string() -> Optional[str]:
    """Get Azure Blob Storage connection string from environment."""
    return os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


def download_model_from_blob(
    blob_name: str,
    container_name: str = "hare-models",
    local_cache_dir: str = "./models",
) -> str:
    """
    Download a model file from Azure Blob Storage.

    Args:
        blob_name: Name of the blob (e.g., "hare_stage2_robust.pth")
        container_name: Name of the Blob Storage container
        local_cache_dir: Local directory to cache downloaded models

    Returns:
        Local file path to the downloaded model

    Raises:
        RuntimeError: If Azure Blob Storage is not configured or download fails
    """
    if BlobClient is None:
        raise RuntimeError(
            "azure-storage-blob is not installed. "
            "Install it with: pip install azure-storage-blob"
        )

    # Create cache directory if it doesn't exist
    Path(local_cache_dir).mkdir(parents=True, exist_ok=True)
    local_path = os.path.join(local_cache_dir, blob_name)

    # Return if already cached
    if os.path.exists(local_path):
        return local_path

    # Get connection string
    conn_str = get_blob_connection_string()
    if not conn_str:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING environment variable not set. "
            "Configure it in your Azure App Service."
        )

    # Download from blob storage
    try:
        blob_client = BlobClient.from_connection_string(
            conn_str,
            container_name=container_name,
            blob_name=blob_name,
        )
        print(f"Downloading {blob_name} from Azure Blob Storage...")
        with open(local_path, "wb") as file:
            file.write(blob_client.download_blob().readall())
        print(f"✓ Downloaded to {local_path}")
        return local_path
    except Exception as e:
        raise RuntimeError(f"Failed to download {blob_name}: {e}")


def list_available_models(
    container_name: str = "hare-models",
) -> list[str]:
    """
    List all available model files in Blob Storage.

    Args:
        container_name: Name of the Blob Storage container

    Returns:
        List of blob names (model filenames)

    Raises:
        RuntimeError: If Azure Blob Storage is not configured
    """
    if BlobClient is None:
        return []

    conn_str = get_blob_connection_string()
    if not conn_str:
        return []

    try:
        from azure.storage.blob import ContainerClient

        container_client = ContainerClient.from_connection_string(
            conn_str,
            container_name=container_name,
        )
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs if blob.name.endswith(".pth")]
    except Exception:
        return []


def is_blob_storage_configured() -> bool:
    """Check if Azure Blob Storage is configured."""
    return bool(get_blob_connection_string() and BlobClient is not None)
