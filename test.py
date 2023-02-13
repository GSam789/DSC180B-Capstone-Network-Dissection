from getdata import get_data
from train import train, validate
import resnet18 
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Load Model & Data

# ResNet
# filename = "models/resnet18_cifar10"
filename = "models/resnet18_cifar10_dropout0.2"
# filename = "models/resnet18_cifar10_dropout0.5"

# Load Test Data (First 100 CIFAR10 validation data)
model = pickle.load(open(filename, "rb"))
test_data = torch.load("test_data.pt")
loss = nn.CrossEntropyLoss()

valid_loader = DataLoader(
        test_data, 
        batch_size=16,
        shuffle=False)

# Test model accuracy
model_name = filename.split("/")[1]
print(f"Model: {model_name}")
val_loss, val_acc = validate(model, valid_loader, loss, 'cpu')
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")