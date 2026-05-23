import torch
import torchvision
import torch.nn as nn


def backbone(name: str,
             pretrained: bool = False,
             finetuning: bool = False,
             num_classes: int = 10):

    model_families = {
        'densenet': ['densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'inception': ['googlenet', 'inception_v3'],
        'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
        'resnext': ['resnext101'],
        'efficient': ['efficientnet_b0', 'efficientnet_b4',
                      'efficientnet_widese_b0', 'efficientnet_widese_b4'],
        'vit': ['vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32', 'vit_h_14']
    }

    if not any(name in models for models in model_families.values()):
        print('name must belong to one of the following families: \n')
        for k, v in model_families.items():
            print(f'{k}: {v}')
        raise ValueError(f"Unsupported backbone: {name}")

    weights = "DEFAULT" if pretrained else None

    if name in model_families['densenet']:
        model = getattr(torchvision.models, name)(weights=weights)
        if finetuning:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)

    elif name in model_families['inception']:
        model = getattr(torchvision.models, name)(weights=weights)
        if finetuning:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    elif name in model_families['resnet']:
        model = getattr(torchvision.models, name)(weights=weights)
        if finetuning:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    elif name == 'resnext101':
        model = torchvision.models.resnext101_32x8d(weights=weights)
        if finetuning:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)

    elif name in model_families['efficient']:
        tv_name = name.replace('_widese', '')
        if not hasattr(torchvision.models, tv_name):
            raise ValueError(
                f"Model {name} (mapped to {tv_name}) not found in torchvision.models"
            )
        model = getattr(torchvision.models, tv_name)(weights=weights)
        if finetuning:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif name in model_families['vit']:
        model = getattr(torchvision.models, name)(weights=weights)
        if finetuning:
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Model {name} not handled")

    return model