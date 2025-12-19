from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CROPPED_DIR = DATA_DIR / "cropped"
GALLERY_DIR = DATA_DIR / "gallery"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CROPPED_DIR.mkdir(parents=True, exist_ok=True)
GALLERY_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PATH = (
    PROJECT_ROOT
    / "weights"
    / "FaceNet_MobileNetV3_P100_Epoch100.pth"
)