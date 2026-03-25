import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# --- CNN Architecture (Math Fixed for 64x64) ---
class StaticCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.fc(self.conv(x))

# Aggressive Augmentation to solve "Only detects two" issue
transform = T.Compose([
    T.Resize((64, 64)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.4, contrast=0.4),
    T.ToTensor(),
])

if __name__ == "__main__":
    dataset = ImageFolder("dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticCNN(len(dataset.classes)).to(device)
    
    # Lower learning rate (0.0001) for better convergence
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    print(f"Training classes: {dataset.classes}")
    for epoch in range(50): # Increased epochs
        total_loss = 0
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/50 | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "static_model.pth")
    with open("labels.txt", "w") as f: f.write("\n".join(dataset.classes))
    print("Success: Model and Labels saved.")