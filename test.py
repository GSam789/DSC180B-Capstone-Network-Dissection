import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import vgg16
from utils import vgg16_focuseddropout

############################################ Validation Function ################################################
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

def test(filename, model):
    model.load_state_dict(torch.load(filename))
    model.to(device)

    # Load Test Data (First 100 CIFAR100 validation data)
    valid_loader = DataLoader(
            test_data, 
            batch_size=16,
            shuffle=False)

    # Test model accuracy
    model_name = filename.split("/")[1][:-4]
    print(f"Model: {model_name}")
    val_loss, val_acc = validate(model, valid_loader, loss, device)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

############################################ Load Model to Validate #############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filenames = ["models/vgg16_cifar100.pth", "models/vgg16_cifar100_focuseddropout0.1.pth", "models/vgg16_cifar100_input_grad_reg.pth"]
models = [vgg16.VGG("VGG16"), vgg16_focuseddropout.VGG("VGG16"), vgg16.VGG("VGG16")]

test_data = torch.load("test_data.pt")
loss = nn.CrossEntropyLoss()
loss.to(device)

for filename, model in zip(filenames, models):
    test(filename, model)