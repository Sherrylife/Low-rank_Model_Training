"""
Configuration file for experiments
"""

import string
import torchvision.transforms as transforms

TRANSFORM_CONFIG = {
    "cifar10_train": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ],
    "cifar10_test": [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ],
    "cifar100_train": [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ],
    "cifar100_test": [
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ]
}

WIKITEXT2_TRANSFORMER_CONFIG = {
    "num_tokens": 33279,
    "embedding_size": 128,
    "num_heads": 8,
    "hidden_size": 2048,
    "num_layers": 3,
    "dropout_rate": 0.1,
    "bptt": 64,
    "mask_rate": 0.15,
}

IMAGE_RESNET_CONFIG = {
    "input_size": 3,
    "hidden_size": [64, 128, 256, 512],
    "output_size": {
        "cifar10": 10,
        "cifar100": 100,
        "tinyImageNet": 200,
        "svhn": 10,
        "ImageNet": 1000
    }
}

IMAGE_VGG_CONFIG = {
    "input_size": 3,
    "hidden_size": 512,
    "output_size": {
        "cifar10": 10,
        "cifar100": 100,
        "tinyImageNet": 200,
        "svhn": 10,
        "ImageNet": 1000
    }
}

CIFAR_VIT_CONFIG = {
    "input_channels": 3,
    "image_size": 32,
    "patch_size": 4,
    "num_classes": {
        "cifar10": 10,
        "cifar100": 100,
    },
    "embed_dim": 512,
    "depth": 6,
    "head": 8,
    "dim_head": 64,
    "mlp_dim": 512,
    "dropout": 0.1,
    "embed_dropout": 0.1,
}

SHAKESPEARE_CONFIG = {
    "input_size": len(string.printable),
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": len(string.printable),
    "n_layers": 2,
    "chunk_len": 80
}

OPTIMIZER_CONFIG = {
    "momentum": 0.9,
    "weight_decay": 5.0e-4,
    "min_lr": 1.0e-5,
    "factor": 0.1,
    "warmup_round": 10,
}

CHARACTERS_WEIGHTS = {
    '\n': 0.43795308843799086,
    ' ': 0.042500849608091536,
    ',': 0.6559597911540539,
    '.': 0.6987226398690805,
    'I': 0.9777491725556848,
    'a': 0.2226022051965085,
    'c': 0.813311655455682,
    'd': 0.4071860494572223,
    'e': 0.13455606165058104,
    'f': 0.7908671114133974,
    'g': 0.9532922255751889,
    'h': 0.2496906467588955,
    'i': 0.27444893060347214,
    'l': 0.37296488139109546,
    'm': 0.569937324017103,
    'n': 0.2520734570378263,
    'o': 0.1934141300462555,
    'r': 0.26035705948768273,
    's': 0.2534775933879391,
    't': 0.1876471355731429,
    'u': 0.47430062920373184,
    'w': 0.7470615815733715,
    'y': 0.6388302610200002
}