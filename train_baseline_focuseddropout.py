from dataloader import get_data
from utils.vgg16 import VGG
from utils import vgg16_dropout as vd
from utils import vgg16_focuseddropout as vfd
import torch
import torch.nn as nn
import torch.optim as optim
from plotting import plot
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    # epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    epoch_acc =  (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


##########################################Load Model & Data##################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############# VGG16
# model = VGG('VGG16', num_classes = 100)
# model_name = "vgg16_cifar100"

# ############# VGG16 with Dropout
# model = vd.VGG('VGG16', num_classes = 100)
# model_name = "vgg16_cifar100_dropout0.2"

# ############# VGG16 with FocusedDropout
model = vfd.VGG('VGG16', num_classes = 100)
model_name = "vgg16_cifar100_focuseddropout0.1"

filename = "models/" + model_name + '.pth'
########################################Train & Validate#####################################################
epochs = 150
train_loader, valid_loader = get_data(batch_size=128)

model.to(device)
loss = nn.CrossEntropyLoss()
loss.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

train_losses, train_accs, val_losses, val_accs = [], [], [], []

for _ in range(epochs):
    print(f"Epoch {_}")
    train_loss, train_acc = train(model, train_loader, optimizer, loss, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    val_loss, val_acc = validate(model, valid_loader, loss, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    scheduler.step()
    print(f"Training loss: {train_loss}, Training accuracy: {train_acc}")
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
    
print("Training Done!")
print("Best training accuracy: ", max(train_accs))
print("Best validation accuracy: ", max(val_accs))
plot(train_losses, train_accs, val_losses, val_accs, model_name)

# Save Model
torch.save(model.state_dict(), filename)
print(f"Model saved on {filename}")