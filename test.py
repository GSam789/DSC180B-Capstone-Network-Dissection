from getdata import get_data
from train import train, validate
import resnet18 
import dropout
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

# Load Model & Data

# ResNet
# filename = "models/resnet18_cifar10"
filename = "models/resnet18_cifar10_dropout0.2"
# filename = "models/resnet18_cifar10_dropout0.5"

model = pickle.load(open(filename, "rb"))
train_loader, valid_loader = get_data()

# Test Data
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()

val_loss, val_acc = validate(model, valid_loader, loss, 'cpu')
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")