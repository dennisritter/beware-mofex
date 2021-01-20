""" Saves a models state on harddrive. """
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_val_acc_on_batch(val_accuracies,
                          model_name,
                          dataset_name,
                          optimizer,
                          num_pretrained_epochs=0,
                          path='data',
                          show=False,
                          save=True):
    """ Plots the validation accuracies on batch number. 

        Args:
            val_accuracies (list): A list of accuracy values to plot. Reach value represents the result of one epoch.
            model_name (str): The name of the trained model that is used in the file name.
            dataset_name (str): The name of the dataset used for training that is used in the file name.
            optimizer(str): The name of the optimizer.
            num_pretrained_epochs (int): The number of epochs this model has been trained before.
            path (str): The path where the figure saved
            show (bool): Show the plot or not
            save (bool): Save plot on disk or not
    """
    num_epochs = len(val_accuracies) + num_pretrained_epochs
    plot_name = f'{model_name}_{dataset_name}_{optimizer}_e{num_pretrained_epochs}-{num_epochs}'
    plt.title(f'Val Acc vs. Epochs for {plot_name}')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label=model_name)
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    if save:
        plt.savefig(f'{path}/{plot_name}.png')
        print(f'Saved Acc/Epoch training Plot to: {path}/{plot_name}.png')
    if show:
        plt.show()
    plt.clf()