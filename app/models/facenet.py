import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

import math
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# =============================== FaceNet =====================================
# =============================================================================
class FaceNet_MobileNetV2(nn.Module):
    def __init__(self, embedding_size:int, seed:int=42):
        super(FaceNet_MobileNetV2, self).__init__()
        base_model = models.mobilenet_v2(pretrained=True)
        
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        torch.manual_seed(seed)
        self.fc = nn.Linear(base_model.last_channel, embedding_size)
                
    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class FaceNet_MobileNetV3Large(nn.Module):
    def __init__(self, embedding_size:int, seed:int=42):
        super(FaceNet_MobileNetV3Large, self).__init__()
        base_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")

        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        torch.manual_seed(seed)
        self.fc = nn.Linear(base_model.classifier[0].in_features, embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class FaceNet_EfficientNet(nn.Module):
    def __init__(self, version: str = "b0", embedding_size: int = 512, seed: int = 42):

        super(FaceNet_EfficientNet, self).__init__()

        if version == "b0":
            base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
            in_features = 1280
        elif version == "b1":
            base_model = models.efficientnet_b1(weights="IMAGENET1K_V1")
            in_features = 1280
        elif version == "b2":
            base_model = models.efficientnet_b2(weights="IMAGENET1K_V1")
            in_features = 1408
        else:
            raise ValueError(f"Unknown EfficientNet version: {version}")

        # Feature extractor 부분만 사용 (classifier 제거)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        torch.manual_seed(seed)
        self.fc = nn.Linear(in_features, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x