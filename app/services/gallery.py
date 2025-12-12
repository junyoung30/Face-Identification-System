from pathlib import Path
import numpy as np
from typing import Dict, List

from app.config import PROJECT_ROOT


GALLERY_DIR = PROJECT_ROOT / "data" / "gallery"
GALLERY_DIR.mkdir(parents=True, exist_ok=True)


def save_embedding(
    person_id: str, 
    embedding: np.ndarray
) -> Path:

    person_dir = GALLERY_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    idx = len(list(person_dir.glob("*.npy")))
    path = person_dir / f"emb_{idx:03d}.npy"

    np.save(path, embedding)
    return path


def load_embeddings(
    person_id: str
) -> List[np.ndarray]:
    
    person_dir = GALLERY_DIR / person_id
    if not person_dir.exists():
        return []
    
    return [np.load(p) for p in sorted(person_dir.glob("*.npy"))]


def load_gallery():
    gallery = {}
    for person_dir in GALLERY_DIR.iterdir():
        if person_dir.is_dir():
            gallery[person_dir.name] = load_embeddings(person_dir.name)
    return gallery


def compute_prototype(
    embeddings: List[np.ndarray]
) -> np.ndarray:
    
    if len(embeddings) == 0:
        raise ValueError("No embeddings.")

    proto = np.mean(embeddings, axis=0)
    proto = proto / np.linalg.norm(proto)
    return proto