""" Helper functions for CNN models """


def print_training_layers(model):
    """Prints what layers will be trained for the model"""
    params_to_update = model.parameters()
    print("Model layers that will be trained:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("\t", name)