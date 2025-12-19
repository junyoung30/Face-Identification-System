## 등록 목록 조회 / 삭제 / 통계

from pathlib import Path
from typing import List, Dict
import shutil

from app.config import CROPPED_DIR, GALLERY_DIR


def list_persons():
    persons = []
    
    for person_dir in sorted(CROPPED_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
            
        persons.append({
            "person_id": person_dir.name,
            "thumbnail_url": get_latest_cropped_image(person_dir)
        })
    
    return persons

        
def get_latest_cropped_image(
    person_dir: Path
):
    images = sorted(person_dir.glob("*.jpg"))
    if len(images) == 0:
        return None
    
    latest = images[-1]
    return f"/cropped/{person_dir.name}/{latest.name}"

def delete_person(
    person_id: str
):
    cropped_dir = CROPPED_DIR / person_id
    gallery_dir = GALLERY_DIR / person_id
    
    if not cropped_dir.exists() and not gallery_dir.exists():
        return False
    
    if cropped_dir.exists():
        shutil.rmtree(cropped_dir)
        
    if gallery_dir.exists():
        shutil.rmtree(gallery_dir)
        
    return True
    