import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from typing import Optional

from app.models.facenet import FaceNet_MobileNetV3Large



class FaceEmbedder:
    def __init__(
        self,
        weight_path: Path,
        embedding_size: int = 512,
        device: str = "cuda"
    ):
        
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.model = FaceNet_MobileNetV3Large(embedding_size=embedding_size)
        self.model.load_state_dict(
            torch.load(weight_path, map_location=self.device)['model_state_dict']
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
    
    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.transform(image).unsqueeze(0).to(self.device)
        
        emb = self.model(x)
        emb = emb.squeeze(0).cpu().numpy()
        return emb
    
    