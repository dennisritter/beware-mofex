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


class CNN1(nn.Module):
    def __init__(self, input_size, channels, kernel_size, output_size):
        super(CNN1, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        padding = 1 if kernel_size == 3 else 2
        self.conv = nn.Conv2d(3, channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(int(channels * (input_size / 2) * (input_size / 2)), output_size)

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        x = x.view(-1, int(channels * (input_size / 2) * (input_size / 2)))
        x = self.fc(x)
        return x


class CNN2(nn.Module):
    def __init__(self, input_size, channels, kernel_size, output_size):
        super(CNN2, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.kernel_size = kernel_size
        self.output_size = output_size

        padding = 1 if kernel_size == 3 else 2
        self.conv1 = nn.Conv2d(3, channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.fc = nn.Linear((channels * 2 * int(input_size / 4) * int(input_size / 4)), output_size)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, channels * 2 * int(input_size / 4) * int(input_size / 4))
        x = self.fc(x)
        return x


# ----- Training Parameters

# Models to choose from [resnet18, resnet50, resnet101]
_models = [
    'cnn1_c6_k3',
    'cnn1_c10_k3',
    'cnn1_c20_k3',
    'cnn1_c6_k5',
    'cnn1_c10_k5',
    'cnn1_c20_k5',
    'cnn2_c6_k3',
    'cnn2_c10_k3',
    'cnn2_c20_k3',
    'cnn2_c6_k5',
    'cnn2_c10_k5',
    'cnn2_c20_k5',
    'resnet18',
    'resnet50',
    'resnet101',
]

_datasets = {'hdm05-122': "./data/hdm05-122/motion_images/hdm05-122_90-10", 'mka-beware-1.1': "./data/mka-beware-1.1/motion_images/mka-beware-1.1"}

# The squared input size of motion images
input_size = 256
# Number of epochs to train for
num_epochs = 50
# Batch size for training (change depending on how much memory you have)
batch_size = 8
# Flag for feature extracting. When False, we finetune the whole model,
train_last_layer_only = False

# ----- Training Setup
for dataset_name in _datasets.keys():
    # Number of classes in the dataset
    num_classes = 122 if dataset_name == 'hdm05-122' else 7

    for model_name in _models:
        for opti in ['sgd', 'adam']:
            # Initialize the model
            if 'cnn' in model_name:
                channels, kernel_size = int(model_name.split('_')[1][1:]), int(model_name.split('_')[2][1:])
                if 'cnn1' in model_name:
                    model = CNN1(input_size, channels, kernel_size, num_classes)
                else:
                    model = CNN2(input_size, channels, kernel_size, num_classes)
            else:
                model, input_size = model_loader.initialize_model(model_name, num_classes, input_size=input_size)

            # Data augmentation and normalization for training and validation repectively
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
            image_datasets = {x: datasets.ImageFolder(os.path.join(_datasets[dataset_name], x), data_transforms[x]) for x in ['train', 'val']}
            # Create training and validation dataloaders
            dataloaders_dict = {
                x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=12)
                for x in ['train', 'val']
            }

            # Observe that all parameters are being optimized
            if opti == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            else:
                optimizer = optim.Adam(model.parameters())

            # Set Loss function
            criterion = nn.CrossEntropyLoss()

            # ----- The actual training

            # Train and evaluate
            model, val_acc_history = model_trainer.train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

            # ----- Post Training
            trained_models_path = Path(f'./output/trained_models/{dataset_name}').resolve()

            # Save model state
            model_saver.save_model(model, trained_models_path, model_name, dataset_name, num_epochs, opti)
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
