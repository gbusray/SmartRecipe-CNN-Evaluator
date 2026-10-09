import argparse
import os
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_preprocessing import get_data_loaders
from models.resnet import get_resnet50_model
from models.mobilenet import get_mobilenet_v3
from models.efficientnet import get_efficientnet_b1
from models.vgg import get_vgg16
from models.densenet import get_densenet121

# Mapping model names to architecture initializers
MODEL_REGISTRY = {
    "ResNet50": get_resnet50_model,
    "MobileNetV3": get_mobilenet_v3,
    "EfficientNetB1": get_efficientnet_b1,
    "VGG16": get_vgg16,
    "DenseNet121": get_densenet121
}

def parse_args():
    """Parses command line arguments for flexible execution across Mac and Cloud."""
    parser = argparse.ArgumentParser(description="Food-101 Model Evaluation & Overfitting Analysis")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=list(MODEL_REGISTRY.keys()),
        help="Name of the architecture to evaluate."
    )
    parser.add_argument(
        "--weights", 
        type=str, 
        required=True, 
        help="Path to the trained .pth checkpoint file."
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64, 
        help="Inference batch size."
    )
    return parser.parse_args()

def evaluate():
    args = parse_args()
    
    # Prioritize CUDA on Colab, then Apple Silicon MPS, falling back to CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"🚀 Evaluation initialized for: {args.model} on device [{device}]")

    # Verify weight checkpoint existence
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Checkpoint not found at: {args.weights}")

    # Initialize model and load trained state dictionary
    model_fn = MODEL_REGISTRY[args.model]
    model = model_fn(num_classes=101)
    
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load validation data
    _, val_loader = get_data_loaders(batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    print(f"🔍 Running inference over validation set...")
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Evaluating Batches"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    
    # Calculate detailed performance metrics
    print("\n" + "=" * 55)
    print(f"📊 PERFORMANCE REPORT: {args.model}")
    print("=" * 55)
    print(f"Validation Loss: {avg_loss:.4f}")
    
    # Macro F1 accounts for class balance; Weighted F1 accounts for support
    report = classification_report(
        all_targets, 
        all_predictions, 
        digits=4, 
        zero_division=0
    )
    print(report)

if __name__ == "__main__":
    evaluate()