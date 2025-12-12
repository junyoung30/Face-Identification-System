import shutil
from pathlib import Path
from fastapi import UploadFile

def save_image(
    file: UploadFile,  # 카메라로 찍어서 서버로 전송된 raw 이미지
    save_dir: Path,    # raw 이미지 저장 경로
    filename: str      # raw 이미지 저장할 파일명
) -> Path:
    """
    file:       카메라로 찍어서 서버로 전송된 raw 이미지
    save_dir:   raw 이미지 저장 경로
    filename:   raw 이미지 저장할 파일명
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / filename
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return save_path