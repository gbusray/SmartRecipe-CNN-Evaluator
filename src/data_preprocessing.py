import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Standardize dimensions for CNN architectures (ResNet, MobileNet, EfficientNet)
IMAGE_SIZE = 224
BATCH_SIZE = 32

def get_data_loaders(data_dir="./data"):
    """
    Downloads the Food-101 dataset and applies preprocessing transformations.
    We use data augmentation for training to prevent overfitting, and standard 
    normalization for validation.
    """
    
    # Training transformations: Resize, Augment (Flip, Rotation), and Normalize
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transformations: Resize, CenterCrop, and Normalize
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Download and load the Food-101 dataset natively
    print("Downloading and preparing Food-101 training dataset (this may take a while)...")
    train_dataset = datasets.Food101(root=data_dir, split="train", download=True, transform=train_transforms)
    
    print("Downloading and preparing Food-101 validation dataset...")
    val_dataset = datasets.Food101(root=data_dir, split="test", download=True, transform=val_transforms)

    # Create DataLoaders for efficient batch processing during model training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader

if __name__ == "__main__":
    # Execute the function to initiate the download and verify the pipeline
    train_loader, val_loader = get_data_loaders()
    print(f"Data loaders successfully created! Training batches ready: {len(train_loader)}")