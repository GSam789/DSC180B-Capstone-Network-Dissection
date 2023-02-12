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
# model = resnet18.ResNet(img_channels = 3, num_layers = 18, block = resnet18.BasicBlock, num_classes = 10)

# ResNet with dropout
model = dropout.ResNet(img_channels = 3, num_layers = 18, block = dropout.BasicBlock, num_classes = 10)
train_loader, valid_loader = get_data()

# Train Data
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()
for _ in range(10):
    train_loss, train_acc = train(model, train_loader, optimizer, loss, 'cpu')
    print(f"Training loss: {train_loss}, Training accuracy: {train_acc}")
print("Training Done!")

val_loss, val_acc = validate(model, valid_loader, loss, 'cpu')
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

# Save Model
# filename = "models/resnet18_cifar10"
# filename = "models/resnet18_cifar10_dropout0.2"
filename = "models/resnet18_cifar10_dropout0.5"
pickle.dump(model, open(filename, "wb"))
print(f"Model saved on {filename}")