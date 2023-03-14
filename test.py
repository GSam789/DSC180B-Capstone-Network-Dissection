from getdata import get_data
from train import train, validate
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import vgg16
from utils import vgg16_focuseddropout

############################################ Load Model to Validate #############################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Baseline VGG16
filename = "models/vgg16_cifar100.pth"
model = vgg16.VGG("VGG16")

# VGG16 with FocusedDropout
# filename = "models/vgg16_cifar100_focuseddropout0.1.pth"
# model = vgg16_focuseddropout.VGG("VGG16")


model.load_state_dict(torch.load(filename))
model.to(device)

# Load Test Data (First 100 CIFAR100 validation data)
test_data = torch.load("test_data.pt")
loss = nn.CrossEntropyLoss()
loss.to(device)

valid_loader = DataLoader(
        test_data, 
        batch_size=16,
        shuffle=False)

# Test model accuracy
model_name = filename.split("/")[1][:-4]
print(f"Model: {model_name}")
val_loss, val_acc = validate(model, valid_loader, loss, device)
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")