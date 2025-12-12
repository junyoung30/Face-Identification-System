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