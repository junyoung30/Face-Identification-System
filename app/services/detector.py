from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from typing import Optional, Tuple



_model_path = hf_hub_download(
    repo_id="AdamCodd/YOLOv11n-face-detection", 
    filename="model.pt"
)
_model = YOLO(_model_path)


def _expand_bbox(
    x1: int, 
    y1: int, 
    x2: int, 
    y2: int,
    img_w: int, 
    img_h: int,
    margin_x: float = 0.3,
    margin_y: float = 0.2
) -> Tuple[int, int, int, int]:
    
    w = x2 - x1
    h = y2 - y1

    pad_w = int(w * margin_x)
    pad_h = int(h * margin_y)

    new_x1 = max(0, x1 - pad_w)
    new_y1 = max(0, y1 - pad_h)
    new_x2 = min(img_w, x2 + pad_w)
    new_y2 = min(img_h, y2 + pad_h)

    return new_x1, new_y1, new_x2, new_y2



def detect_face_from_image(
    img: np.ndarray,
    margin_x: float = 0.3,
    margin_y: float = 0.2
):

    h, w, _ = img.shape
    
    results = _model.predict(
        source=img,
        save=False,
        verbose=False,
    )
    
    for r in results:
        if len(r.boxes) == 0:
            return None
        
        boxes = r.boxes
        scores = boxes.conf.cpu().numpy()
        max_idx = int(np.argmax(scores))
        box = boxes[max_idx]
        
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    
        px1, py1, px2, py2 = _expand_bbox(
            x1, y1, x2, y2,
            img_w=w, img_h=h,
            margin_x=margin_x,
            margin_y=margin_y
        )
    
        cropped = img[py1:py2, px1:px2]
        return cropped
    
    return None
    

def detect_face(
    img_path: Path,
    margin_x: float = 0.3,
    margin_y: float = 0.2
):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    return detect_face_from_image(
        img,
        margin_x=margin_x,
        margin_y=margin_y
    )
    
    