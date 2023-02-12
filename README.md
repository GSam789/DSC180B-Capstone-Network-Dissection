# DSC180B-Capstone-Network-Dissection

This branch is used to implement FocusedDropout onto Resnet-18 trained on CIFAR-10. So far, I have implemented regular dropout with rates 0.2 and 0.5 on the model. 

Files:
- resnet18.py: skeleton of ResNet-18 without dropout
- dropout.py: Skeleton of ResNet-18 with dropout (0.2 for default)
- getdata.py: load training and testing data of CIFAR10
- train.py: train and test functions
- main.py: Trains and tests the selected model on CIFAR10 and saves the result in the models/ folder
- test.py: Tests one of the specified model in the models/ folder on the CIFAR10 test data
