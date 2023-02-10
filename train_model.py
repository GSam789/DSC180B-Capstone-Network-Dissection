from getdata import get_data
from train import train, validate
from resnet18 import ResNet
import pickle

# Load Data & Model
train_loader, valid_loader = get_data()
model = resnet18(weights = 'DEFAULT')

# Training
for _ in range(10):
    train(resnet18, train_loader, optimizer, loss, 'cpu')
print("Training Done!")
validate(resnet18, valid_loader, loss, 'cpu')

filename = "resnet18_cifar10"
pickle.dump(model, open("resnet18_cifar10", "wb"))
print(f"Model saved on {filename}")