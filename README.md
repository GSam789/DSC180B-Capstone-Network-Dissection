# DSC180B-Capstone-Network-Dissection

This branch is used to implement FocusedDropout onto Resnet-18 trained on CIFAR-10. So far, I have implemented regular dropout with rates 0.2 and 0.5 on the model. 

Files:
- resnet18.py: skeleton of ResNet-18 without dropout
- dropout.py: Skeleton of ResNet-18 with dropout (0.2 for default)
- getdata.py: load training and testing data of CIFAR10
- train.py: train and test functions
- main.py: Trains and tests the selected model on CIFAR10 and saves the result in the models/ folder
- test.py: Tests one of the specified model in the models/ folder on the first 128 CIFAR10 test data points.
- test_data.pt : First 100 data points in the CIFAR10 test data to test on

Simply run ```python3 test.py``` to test the ResNet-18 model (with 0.2 dropout rate) on the CIFAR-10 test data (first 128 points).
