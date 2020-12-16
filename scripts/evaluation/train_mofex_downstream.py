# The idea is to firstly train mofex (with motion images) on HDM05 with a classification task,
# then fine-tune on downstream repetition identification task.

# 1. Setup HDM05 dataset
#     a. Full reps -> Classification
#     b. Split full reps into random chunk sizes -> Classification
#         1. 1, 2, 4, 8, ..., frames?
#         2. Generate different chunk sizes for each class not one random size foreach example
#     c. Implement suitable DataLoader
# 2. Pre-Train MOFEX
#     a. Pre-pare MOFEX model with additional (easy to remove) classification layer1
#     b. Pre-train ResNet on new Dataset
#     c. save best model
# 3. Setup MKA dataset
#     a. Full reps -> rep identification
#     b. Split full reps into stream-like chunks
#         1. 1 frame, 1+1 frame, 1+1+1 frame, etc...
#         2. Two folders: successful & fail rep
#         3. There will be a lot of ! reps and less correct/ full ones
#     c. Implement suitable DataLoader
# 4. Fine-tune MOFEX
#     a. Add fine-tune layer to pre-trained ResNet
#     b. Fine-tune on MKA dataset
#     c. save best model
# 5. (OPTIONAL) Evaluate against old RepNet implementation
#     a. Probably re-train ResNet on MKA with splitting logic from 1.b.
#     b. How to evaluate?

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from pathlib import Path
import mofex.model_loader as model_loader
import mofex.model_saver as model_saver
import mofex.model_plotter as model_plotter
import mofex.model_trainer as model_trainer


def pre_train():
    # Models to choose from [resnet18, resnet50, resnet101]
    _models = ['resnet101']
    _datasets = {
        'hdm05-122': "./data/hdm05-122/motion_images/hdm05-122_90-10_downstream"
    }

    # The squared input size of motion images
    input_size = 256
    # Number of epochs to train for
    num_epochs = 50
    # Batch size for training (change depending on how much memory you have)
    batch_size = 8

    # ----- Training Setup
    for dataset_name in _datasets.keys():
        # Number of classes in the dataset
        num_classes = 122

        for model_name in _models:
            for opti in ['sgd', 'adam']:
                # Initialize the model
                model, input_size = model_loader.initialize_model(
                    model_name,
                    num_classes,
                    input_size=input_size,
                    pretrained=False)

                # Data augmentation and normalization for training and validation repectively
                data_transforms = {
                    'train':
                    transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ]),
                    'val':
                    transforms.Compose([
                        transforms.Resize(input_size),
                        transforms.CenterCrop(input_size),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
                    ]),
                }

                # Create training and validation datasets
                image_datasets = {
                    x: datasets.ImageFolder(
                        os.path.join(_datasets[dataset_name], x),
                        data_transforms[x])
                    for x in ['train', 'val']
                }
                # Create training and validation dataloaders
                dataloaders_dict = {
                    x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=12)
                    for x in ['train', 'val']
                }

                # Observe that all parameters are being optimized
                if opti == 'sgd':
                    optimizer = optim.SGD(model.parameters(),
                                          lr=0.001,
                                          momentum=0.9)
                else:
                    optimizer = optim.Adam(model.parameters())

                # Set Loss function
                criterion = nn.CrossEntropyLoss()

                # ----- The actual training

                # Train and evaluate
                model, val_acc_history = model_trainer.train_model(
                    model,
                    dataloaders_dict,
                    criterion,
                    optimizer,
                    num_epochs=num_epochs)

                # ----- Post Training
                trained_models_path = Path(
                    f'./output/trained_downstream/{dataset_name}').resolve()

                # Save model state
                model_saver.save_model(model, trained_models_path, model_name,
                                       dataset_name, num_epochs, opti)
                # Plot the training curves of validation accuracy vs. number of epochs
                val_acc_history = [acc.cpu().numpy() for acc in val_acc_history]
                model_plotter.plot_val_acc_on_batch(val_acc_history,
                                                    model_name,
                                                    dataset_name,
                                                    opti,
                                                    path=trained_models_path,
                                                    num_pretrained_epochs=0,
                                                    show=True,
                                                    save=True)


def fine_tune():
    # TODO: load model without last layer + add new sequential for downstream
    pass


if __name__ == "__main__":
    pre_train()
    fine_tune()