from getdata import get_data
from train import train, validate
import resnet18 as rn
import dropout_model as dm
import focuseddropout_model as fm
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from plotting import plot

# Load Model & Data

# model_name = "models/resnet18_cifar10"
# model_name = "models/resnet18_cifar10_dropout0.2"
model_name = "resnet18_cifar10_dropout0.5"
filename = "models/" + model_name

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet
# model = rn.ResNet(img_channels = 3, num_layers = 18, block = resnet18.BasicBlock, num_classes = 10)

# ResNet with dropout
# model = dm.ResNet(img_channels = 3, num_layers = 18, block = dropout.BasicBlock, num_classes = 10, dropout_rate = 0.2)

# ResNet with FocusedDropout
model = fm.ResNet(img_channels = 3, num_layers = 18, block = dropout.BasicBlock, num_classes = 10, par_rate = 0.1)

# Train Data
epochs = 100
train_loader, valid_loader = get_data()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
loss = nn.CrossEntropyLoss()
train_losses, train_accs, val_losses, val_accs = [], [], [], []

for _ in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, loss, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss, val_acc = validate(model, valid_loader, loss, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Training loss: {train_loss}, Training accuracy: {train_acc}")
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
print("Training Done!")
plot(train_losses, train_accs, val_losses, val_accs, model_name)

# Save Model
pickle.dump(model, open(filename, "wb"))
print(f"Model saved on {filename}")