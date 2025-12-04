from __future__ import annotations
import io
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from . import config


def load_class_names(path: Path | None = None, num_classes: int | None = None) -> List[str]:
    p = Path(path) if path else config.CLASS_NAMES_PATH
    if p.exists():
        try:
            data = json.loads(Path(p).read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass
    # Fallback to generic names; recommend updating file
    n = num_classes if num_classes is not None else 4
    return [f"Class {i}" for i in range(n)]


def read_image_to_rgb(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def pil_to_model_array(img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    img_resized = img.resize(target_size)
    arr = np.asarray(img_resized).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr
