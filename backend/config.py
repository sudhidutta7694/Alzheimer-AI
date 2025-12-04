import os
from pathlib import Path

# Defaults
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "trained_model" / "Alzheimer_Detection_model.h5"
DEFAULT_CLASS_NAMES_PATH = Path(__file__).resolve().parent / "class_names.json"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", str(DEFAULT_CLASS_NAMES_PATH))).resolve()

INPUT_SIZE = (224, 224)

