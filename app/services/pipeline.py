from fastapi import UploadFile
from pathlib import Path
import cv2
from datetime import datetime

from app.config import RAW_DIR, CROPPED_DIR
from app.services.storage import save_image
from app.services.detector import detect_face



def process_uploaded_image(
    file: UploadFile,
    person_id: str = None
) -> dict:
    
    # --------------------
    # 1. raw image 저장
    # --------------------
    ext = (
        file.filename.split(".")[-1]
        if file.filename and "." in file.filename
        else "jpg"
    )
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.{ext}"
    
    raw_path = save_image(
        file=file,
        save_dir=RAW_DIR,
        filename=filename
    )
    
    # --------------------
    # 2. 얼굴 검출 + crop
    # --------------------
    cropped_face = detect_face(raw_path)
    if cropped_face is None:
            return {
                "status": "no_face_detected",
                "raw_path": str(raw_path)
            }
        
    # --------------------
    # 3. cropped 이미지 저장
    # --------------------
    target_person = person_id if person_id is not None else "unknown"
    save_dir = CROPPED_DIR / target_person
    save_dir.mkdir(parents=True, exist_ok=True)
    
    cropped_path = save_dir / filename
    cv2.imwrite(str(cropped_path), cropped_face)
    
    return {
        "status": "ok",
        "raw_path": str(raw_path),
        "cropped_path": str(cropped_path),
        "person_id": target_person
    }
    
