from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image


def preprocess_image(image_path: str) -> str:
    """Load original image and convert to JPEG base64."""
    path = Path(image_path)
    if not path.exists():
        # Include path in the exception message for easier diagnostics.
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        image = img.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded
