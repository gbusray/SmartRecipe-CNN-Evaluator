import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import get_data_loaders

# Import all modular architectures (ResNet import is fully fixed!)
from models.resnet import get_resnet50_model
from models.mobilenet import get_mobilenet_v3
from models.efficientnet import get_efficientnet_b1
from models.vgg import get_vgg16
from models.densenet import get_densenet121

def get_optimizer_target(model_name, model):
    """
    Returns the specific classification layer parameters for the optimizer.
    Since we are using Transfer Learning and freezing the base layers, 
    we only train these final classification heads.
    """
    if model_name == "ResNet50": return model.fc.parameters()
    elif model_name == "MobileNetV3": return model.classifier[3].parameters()
    elif model_name == "EfficientNetB1": return model.classifier[1].parameters()
    elif model_name == "VGG16": return model.classifier[6].parameters()
    elif model_name == "DenseNet121": return model.classifier.parameters()

def run_batch_training(num_epochs=10):
    # Utilize Apple Silicon MPS (Metal Performance Shaders) for local GPU acceleration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Batch Training initialized on: {device}\n")

    train_loader, val_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()

    # Dictionary of all models to train sequentially
    architectures = {
        "ResNet50": get_resnet50_model,
        "MobileNetV3": get_mobilenet_v3,
        "EfficientNetB1": get_efficientnet_b1,
        "VGG16": get_vgg16,
        "DenseNet121": get_densenet121
    }

    results = {}

    for name, model_init in architectures.items():
        print("=" * 40)
        print(f"✨ INITIALIZING: {name}")
        print("=" * 40)
        
        # Initialize a fresh model instance to guarantee a fair baseline
        model = model_init(num_classes=101).to(device)
        optimizer = optim.Adam(get_optimizer_target(name, model), lr=0.001)

        for epoch in range(num_epochs):
            # --- TRAINING PHASE ---
            # Enable training mode (activates layers like Dropout and BatchNorm)
            model.train()
            running_loss = 0.0
            print(f"[{name}] --- Epoch {epoch+1}/{num_epochs} ---")
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if i % 50 == 49:
                    print(f"Batch {i+1} | Loss: {running_loss / 50:.4f}")
                    running_loss = 0.0
                    
            # --- VALIDATION PHASE ---
            # Enable evaluation mode (freezes Dropout/BatchNorm for stable testing)
            model.eval()
            correct, total = 0, 0
            print(f"🔍 Validating {name}...")
            
            # Disable gradient calculation to save memory and speed up validation
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"✅ [{name}] Epoch {epoch+1} Validation Accuracy: {accuracy:.2f}%\n")
            
            # Store the final epoch's accuracy metric for the comparative leaderboard
            if epoch == num_epochs - 1:
                results[name] = accuracy

        # --- SAVE THE MODEL ---
        # Persist weights to the local data directory
        save_path = f"data/{name.lower()}_food101.pth"
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved {name} weights to {save_path}\n")

    # --- FINAL COMPARISON REPORT ---
    print("🏆 FINAL BATCH TRAINING RESULTS 🏆")
    for name, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"{name}: {acc:.2f}%")

if __name__ == "__main__":
    # Ready for the overnight 10-epoch run!
    run_batch_training(num_epochs=10)