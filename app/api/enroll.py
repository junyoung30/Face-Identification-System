from fastapi import APIRouter, UploadFile, File
import cv2

from app.services.pipeline import process_uploaded_image
from app.services.embedder import FaceEmbedder
from app.services.gallery import save_embedding
from app.config import PROJECT_ROOT

router = APIRouter()

WEIGHT_PATH = PROJECT_ROOT / "weights" / "FaceNet_MobileNetV2_Epoch100.pth"
embedder = FaceEmbedder(weight_path=WEIGHT_PATH)

@router.post("/enroll/{person_id}")
async def enroll_person(
    person_id: str,
    file: UploadFile = File(...)
):
    
    result = process_uploaded_image(file, person_id)
    
    if result["status"] != "ok":
        return result
    
    cropped_path = result["cropped_path"]
    img = cv2.imread(cropped_path)
    
    if img is None:
        return {
            "status": "read_failed",
            "path": cropped_path
        }
    
    embedding = embedder(img)
    emb_path = save_embedding(person_id, embedding)
    
    return {
        "status": "enrolled",
        "person_id": person_id,
        "embedding_path": str(emb_path),
        "cropped_path": cropped_path
    }

