import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from app.services.embedder import FaceEmbedder
from app.services.detector import detect_face_from_image
from app.services.pipeline import identify_from_embedding
from app.config import WEIGHT_PATH


router = APIRouter()

embedder = FaceEmbedder(weight_path=WEIGHT_PATH)

@router.post("/identify")
async def identify_person(
    file: UploadFile = File(...)
):
    
    image_bytes = await file.read()
    
    img = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )
    if img is None:
        return {
            "status": "read failed"
        }
    
    cropped = detect_face_from_image(img)
    if cropped is None:
        return {
            "status": "no face detected"
        }
    
    emb = embedder(cropped)
    
    return identify_from_embedding(emb)