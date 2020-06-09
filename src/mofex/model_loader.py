""" Loads Models with the desired parameter properties. """
import torch
import torch.nn as nn
import mofex.models.resnet as resnet


def initialize_model(model_name, num_classes, input_size=256, feature_extract=False, pretrained=True):
    """ Returns a CNN model with the specified model_name with respect to the parameter settings.

        Args:
            model_name (str): The name of the model you want to initialize.
            num_classes (int): Sets the number of output classes/outputs of the last layer. 
            feature_extract (Boolean): Decides whether to freeze all but the last layer (True) or train all layers (False).
            pretrained (Boolean): Decides whether to load a pretrained model or give birth to a new one. 
    """
    model = None
    if model_name == "resnet18":
        model = resnet.load_resnet18()
        set_parameter_requires_grad(model, feature_extract)
        set_output_layer(model, num_classes)
        input_size = input_size
    elif model_name == "resnet50":
        model = resnet.load_resnet50()
        set_parameter_requires_grad(model, feature_extract)
        set_output_layer(model, num_classes)
        input_size = input_size
    elif model_name == "resnet101":
        model = resnet.load_resnet101()
        set_parameter_requires_grad(model, feature_extract)
        set_output_layer(model, num_classes)
        input_size = input_size

    else:
        print("Invalid model name, exiting...")
        exit()

    return model, input_size


def load_trained_model(model_name, remove_last_layer=True, state_dict_path='./data/trained_models/hdm05-122_90-10/resnet18_hdm05-122_90-10.pt'):
    model = None
    if model_name == "resnet18_hdm05-122_90-10":
        model = resnet.load_resnet18_finetuned_hdm05_122_9010(state_dict_path=state_dict_path)
    elif model_name == "resnet50_hdm05-122_90-10":
        model = resnet.load_resnet50_finetuned_hdm05_122_9010(state_dict_path=state_dict_path)
    elif model_name == "resnet101_hdm05-122_90-10":
        model = resnet.load_resnet101_finetuned_hdm05_122_9010(state_dict_path=state_dict_path)
    elif model_name == "resnet101_cmu-30_80-20_256":
        model = resnet.load_resnet101_finetuned_cmu30_8020(state_dict_path=state_dict_path)
    else:
        print("Invalid model name, exiting...")
        exit()
    return model


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def set_output_layer(model, num_classes):
    num_input_last_layer = model.fc.in_features
    model.fc = nn.Linear(num_input_last_layer, num_classes)