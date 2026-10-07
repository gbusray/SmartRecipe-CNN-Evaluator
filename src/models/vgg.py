import torch.nn as nn
from torchvision import models

def get_vgg16(num_classes=101):
    """Initializes VGG16. A classic, heavy architecture for baseline depth comparison."""
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for param in model.parameters(): 
        param.requires_grad = False
    
    # VGG16's final classification layer is at index 6
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model