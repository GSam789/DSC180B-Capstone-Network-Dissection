# Improving Network Accuracy through Network Manipulation


## To run our code:
Simply run ```python3 test.py``` to find the accuracies of our 3 trained VGG16 models on the first 100 test data points of CIFAR-100.

## Project Background & Context
Deep learning has been growing rapidly over the past couple decades due to its ability in solving extremely complex problems. However, this machine learning
method is often considered as a "black box" since it is unclear how the neurons of a deep learning model work together to arrive at the final output. A recently found
method called Network Dissection has solved this interpretability issue by coming up with a visual that shows what each neuron looks for and why. Given that we have 
information regarding neuron activities, we want to investigate if we can use this information to further improve our network's performance.

Thus, in this project, we explored 3 methods:
1. Network Dissection Intervention
2. FocusedDropout
3. Input Gradient Regularization

We implemented these network dissection intervention on VGG16 that was pre-trained on Places365.
We implemented FocusedDropout and Input Gradient Regularization separately on VGG16 on CIFAR-100 dataset. This was done due to our limited resources since training on 
Places365 would take 50 days.

## Rundown of Folders & Files
In this Github repository, you will find the following folders and files:

- Folders:
  - models: Contains all the state dictionaries of the trained models
  - utils: Contains all the model skeletons and FocusedDropout implementation

- Files:
  - dataloader.py: Loads the CIFAR-100 dataset into a DataLoader
  - plotting.py: Plots the resulting accuracies and losses over epochs
  - test_data.pt: The first 100 test data points of CIFAR-100
  - test.py: Runs our 3 VGG16 on CIFAR-100 models (Baseline, VGG16 with FocusedDropout, VGG16 with Input Gradient Regularization) on test_data.pt 
  - train_baseline_focuseddropout.py: When run, trains either baseline model (Plain VGG16) or VGG16 with FocusedDropout from scratch based on selected model in file and 
saves final model's state dictionary into the models/ folder
  - train_input_grad.py: When run, trains VGG16 with Input Gradient Regularization and saves final model's state dictionary into the models/ folder

