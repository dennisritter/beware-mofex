from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pathlib import Path
import mofex.models.resnet as resnet
import mofex.model_loader as model_loader
import mofex.model_saver as model_saver
import mofex.model_plotter as model_plotter
import mofex.model_trainer as model_trainer

# ----- Training Parameters

# Models to choose from [resnet18, resnet50, resnet101]
model_name = 'resnet101'
# Dataset name. Choose from [hdm05-122_90-10]
dataset_name = 'mka-beware-1.1'
# Top level data directory. Here we assume the format of the directory conforms to the ImageFolder structure
data_dir = "./data/motion_images/mka-beware-1.1"
# The squared input size of motion images
input_size = 256
# Number of classes in the dataset
num_classes = 7
# Number of epochs to train for
num_epochs = 5
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Flag for feature extracting. When False, we finetune the whole model,
train_last_layer_only = False

# ----- Training Setup

# Initialize the model
model, input_size = model_loader.initialize_model(model_name, num_classes, input_size=input_size)

# Data augmentation and normalization for training and validation repectively
# TODO: What else could we do to optimize the data for training?
data_transforms = {
    'train':
    transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val':
    transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
# TODO: Check how to use more num_workers subprocesses.
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in ['train', 'val']}

# Observe that all parameters are being optimized
# TODO: Understand Optimizer and parameters
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Set Loss function
criterion = nn.CrossEntropyLoss()

# ----- The actual training

# Train and evaluate
# TODO: Support training for multiple models in one run
model, val_acc_history = model_trainer.train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# ----- Post Training
trained_models_path = Path(f'./data/trained_models/{dataset_name}').resolve()

# Save model state
model_saver.save_model(model, trained_models_path, model_name, dataset_name, num_epoch=num_epochs)
# Plot the training curves of validation accuracy vs. number of epochs
val_acc_history = [acc.cpu().numpy() for acc in val_acc_history]
model_plotter.plot_val_acc_on_batch(val_acc_history, model_name, dataset_name, path=trained_models_path, num_pretrained_epochs=0, show=False, save=True)
