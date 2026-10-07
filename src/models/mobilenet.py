import torch.nn as nn
from torchvision import models

def get_mobilenet_v3(num_classes=101):
    """
    Initializes pre-trained MobileNetV3.
    Chosen for high inference speed and mobile compatibility.
    """
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    
    # Freeze early layers to retain generic feature extraction
    for param in model.parameters():
        param.requires_grad = False
        
    # MobileNetV3 uses a different classifier block structure than ResNet
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    
    return model