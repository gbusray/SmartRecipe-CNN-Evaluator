import torch.nn as nn
from torchvision import models

def get_efficientnet_b1(num_classes, pretrained=True, freeze_backbone=False):
    """
    Initializes EfficientNet-B1 with dynamic backbone freezing.
    Architectural rationale: Decoupling the freezing logic allows this same 
    function to be used for both initial transfer learning (frozen) and 
    deep fine-tuning (unfrozen).
    """
    
    weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b1(weights=weights)

    # Toggle gradients for the feature extractor
    for param in model.parameters():
        param.requires_grad = not freeze_backbone
        
    # Reconfigure the final classification head for our specific dataset
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model
