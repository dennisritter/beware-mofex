import torchvision.models as models
import torch
import os


def load_resnet18(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet18(pretrained=pretrained, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet34(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet34(pretrained=pretrained, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet50(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet50(pretrained=pretrained, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet101(pretrained: bool = True, remove_last_layer=False):
    model = models.resnet101(pretrained=pretrained, progress=True)
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet18_finetuned_hdm05_122_9010(remove_last_layer=True,
                                           state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet18_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet18_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 122)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet50_finetuned_hdm05_122_9010(remove_last_layer=True,
                                           state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet50_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet50_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, 122)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet101_finetuned_hdm05_122_9010(remove_last_layer=True,
                                            state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet101(pretrained=False)
    model.fc = torch.nn.Linear(2048, 122)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet101_finetuned_cookie_downstream(remove_last_layer=False,
                                               state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet101(pretrained=False)
    model.fc = torch.nn.Linear(2048, 2)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model


def load_resnet101_finetuned_cmu30_8020(remove_last_layer=True,
                                        state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet101(pretrained=False)
    model.fc = torch.nn.Linear(2048, 30)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])


def load_resnet101_finetuned_mka7_9010(remove_last_layer=True,
                                       state_dict_path=None):
    if not os.path.isfile(state_dict_path):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a file. Exiting program.'
        )
        exit()
    if not str(state_dict_path).endswith('.pt'):
        print(
            f'The state_dict_path parameter of load_resnet101_finetuned function does not lead to a .pt file. Exiting program.'
        )
        exit()
    model = models.resnet101(pretrained=False)
    model.fc = torch.nn.Linear(2048, 7)
    model.load_state_dict(torch.load(state_dict_path))
    if remove_last_layer:
        model = torch.nn.Sequential(*list(model.children())[:-1])

    return model