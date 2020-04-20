import torchvision.models as models
import torch


def load_resnet18(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet18(pretrained=True, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet34(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet34(pretrained=True, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet50(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet50(pretrained=True, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet101(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet101(pretrained=True, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model