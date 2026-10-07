import torch.nn as nn
from torchvision import models

def get_resnet50_model(num_classes=101):
    """
    Initializes a pre-trained ResNet-50 model and modifies the final 
    fully connected layer for our specific classification task (Food-101).
    """
    # Load the pre-trained ResNet-50 model (Transfer Learning)
    # Using weights='DEFAULT' loads the best available pre-trained weights
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze the early convolutional layers so we retain the pre-trained feature extraction
    # This drastically reduces training time and computational cost
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer (classifier)
    # Original ResNet-50 outputs 1000 classes; we need it to output 101 for our dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Initialize the model and verify the output dimension
    model = get_resnet50_model()
    print(f"ResNet-50 architecture successfully built! Output classes: {model.fc.out_features}")