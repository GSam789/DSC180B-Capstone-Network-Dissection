from getdata import get_data
from train import train, validate
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

############################################ Load Model to Validate #############################################

# Unmodified ResNet18
# filename = "models/resnet18_cifar10"

# ResNet18 with Dropout
filename = "models/cifar10/resnet/resnet18_cifar10_dropout0.2"
# filename = "models/resnet18_cifar10_dropout0.5"

# ResNet18 with FocusedDropout
# filename = "models/resnet18_cifar10_focuseddropout0.01sgd"
# filename = "models/resnet18_cifar10_focuseddropout0.05sgd"
# filename = "models/resnet18_cifar10_focuseddropout0.1sgd"

# ResNet18 with AntiFocusedDropout
# filename = "models/resnet18_cifar10_antifocuseddropout0.1sgd"

model = pickle.load(open(filename, "rb"))

# Load Test Data (First 100 CIFAR10 validation data)
test_data = torch.load("test_data.pt")
loss = nn.CrossEntropyLoss()

# _, valid_loader = get_data(128)
valid_loader = DataLoader(
        test_data, 
        batch_size=16,
        shuffle=False)

# Test model accuracy
model_name = filename.split("/")[1]
print(f"Model: {model_name}")
val_loss, val_acc = validate(model, valid_loader, loss, 'cuda')
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")