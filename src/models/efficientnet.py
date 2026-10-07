import torch.nn as nn
from torchvision import models

def get_efficientnet_b1(num_classes=101):
    """
    Initializes pre-trained EfficientNet-B1.
    Chosen for optimal balance between accuracy and computational efficiency.
    """
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
        
    # EfficientNet classifier structure modification
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    return model
