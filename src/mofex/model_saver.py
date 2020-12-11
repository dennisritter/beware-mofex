""" Saves a models state on harddrive. """
import os
import torch


def save_model(model, save_dir, model_name, dataset_name, num_epoch, optimizer):
    """ Saves a models state to disc.

        Args:
            model:  nn model
            save_dir: save model direction
            model_name:  model name
            epoch:  epoch
            optimizer: name of optimizer
    """
    # Create Folder(s) in path if doesn't exist already
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(
        save_dir,
        f'{model_name}_{dataset_name}_{optimizer}_e{num_epoch}.pt',
    )
    output_path = open(save_path, mode="wb")
    torch.save(model.state_dict(), output_path)
    output_path.close()
    print(f'Saved model state to: {output_path.name}')