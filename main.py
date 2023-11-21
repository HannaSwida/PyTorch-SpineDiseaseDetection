import pandas as pd
from data_loader import SpineDicomDataset
from models import SimpleCNN, CustomCNN
import pydicom
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.Resize((512, 512)), 
    transforms.ToTensor(),
])
parent_path = "./data/files"
dataset = SpineDicomDataset(csv_file='./data/xray_spine.csv', parent_path=parent_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# Define the size of your test set
test_size = int(0.2 * len(dataset))  # 20% of the dataset for testing
train_size = len(dataset) - test_size  # The rest for training

# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("success")
num_classes = 6 
model = CustomCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 10  
for epoch in range(num_epochs):
    model.train()  
    running_loss = 0.0
    for images, labels in train_loader:
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}')
    # Testing loop
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {100 * correct / total:.2f}%')
