import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import get_data_loaders
from models.efficientnet import get_efficientnet_b1

def train_and_evaluate(num_epochs=2):
    """
    Executes the training loop for EfficientNet-B1.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training & Evaluating initialized on: {device}")

    train_loader, val_loader = get_data_loaders()
    model = get_efficientnet_b1(num_classes=101).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # CRITICAL: EfficientNet-B1 classification layer is at index 1
    optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

    for epoch in range(num_epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
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
        model.eval()
        correct = 0
        total = 0
        print("🔍 Running Validation...")
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"✅ Epoch {epoch+1} Validation Accuracy: {accuracy:.2f}%")

    # --- SAVE THE MODEL ---
    save_path = "data/efficientnetb1_food101.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model weights successfully saved to {save_path}")

if __name__ == "__main__":
    train_and_evaluate(num_epochs=2)