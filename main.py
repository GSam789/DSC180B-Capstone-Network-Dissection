from getdata import get_data
from train import train, validate
from resnet18 import ResNet, BasicBlock
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# Load Model & Data
model = ResNet(img_channels = 3, num_layers = 18, block = BasicBlock, num_classes = 10)
train_loader, valid_loader = get_data()

# Train Data
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()
for _ in range(10):
    train(model, train_loader, optimizer, loss, 'cpu')
print("Training Done!")
validate(model, valid_loader, loss, 'cpu')

# Save Model
filename = "resnet18_cifar10"
pickle.dump(model, open(filename, "wb"))
print(f"Model saved on {filename}")