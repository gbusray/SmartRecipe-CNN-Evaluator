import torch.nn as nn
from torchvision import models

def get_densenet121(num_classes=101):
    """Initializes DenseNet121. Highly efficient feature reuse through dense blocks."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for param in model.parameters(): 
        param.requires_grad = False
        
    # DenseNet has a simple, single-layer classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model