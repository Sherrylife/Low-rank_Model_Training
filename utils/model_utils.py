import torch
from torch import optim, nn
from model.LSTM import NextCharacterLSTM, LowRankNextCharacterLSTM
from model.ResNet import *
from model.VGG import *
from model.Transformer import FillTextTransformer
from model.LowRankTransformer import LowRankFillTextTransformer
from model.ViT import VisionTransformer
from model.LowRankViT import LowRankVisionTransformer
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import string
from utils.experiment_config import *
from utils.metrics import *

def generate_model(args):
    model_name = args['model']
    method = args['method']
    low_rank_ratio = args['ratio_LR']
    device = args['device']

    if model_name == 'lstm':
        if method == 'original':
            model = NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"],
                device=device,
            ).to(device)
        else:
            model = LowRankNextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"],
                low_rank_ratio=low_rank_ratio,
                device=device,
            ).to(device)
    elif model_name == 'transformer':
        if method == 'original':
            model = FillTextTransformer(args=args).to(device)
        else:
            model = LowRankFillTextTransformer(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'],
                                               args=args).to(device)
    elif model_name == 'vit':
        if method == 'original':
            model = VisionTransformer(args=args).to(device)
        else:
            model = LowRankVisionTransformer(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'],
                                             args=args).to(device)
    elif model_name == 'resnet18':
        if method == 'original':
            model = ResNet18(args=args).to(device)
        else:
            model = LowRankResNet18(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'],
                                    args=args).to(device)
    elif model_name == 'resnet101':
        if method == 'original':
            model = ResNet101(args=args).to(device)
        else:
            model = LowRankResNet101(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'],
                                     args=args).to(device)
    elif model_name == 'vgg11':
        if method == 'original':
            model = VGG11(args).to(device)
        else:
            model = LowRankVGG11(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'], args=args).to(device)
    elif model_name == 'vgg16':
        if method == 'original':
            model = VGG16(args).to(device)
        else:
            model = LowRankVGG16(ratio_LR=args['ratio_LR'], decom_rule=args['decom_rule'], args=args).to(device)
    else:
        raise NotImplementedError("Model name is invalid.")

    return model


def generate_optimizer(model, args):
    optim_name = args['optim']
    if optim_name == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args['lr'],
            momentum=OPTIMIZER_CONFIG["momentum"],
            weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        )
    elif optim_name == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args['lr'],
            momentum=OPTIMIZER_CONFIG["momentum"],
            weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        )
    elif optim_name == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args['lr'],
            weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        )
    elif optim_name == 'Adamax':
        optimizer = optim.Adamax(
            model.parameters(),
            lr=args['lr'],
            weight_decay=OPTIMIZER_CONFIG["weight_decay"],
        )
    else:
        raise NotImplementedError('Non-valid optimizer name.')

    return optimizer


def generate_scheduler(optimizer, args):
    scheduler_name = args['lr_scheduler']
    n_rounds = args['T']

    if scheduler_name == "None":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda train_round: 1
        )
    elif scheduler_name == "Sqrt":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda train_round: 1 / np.sqrt(train_round) if train_round > 0 else 1
        )

    elif scheduler_name == "Linear":
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda train_round: 1 / train_round if train_round > 0 else 1
        )

    elif scheduler_name == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_rounds,
            eta_min=OPTIMIZER_CONFIG["min_lr"]
        )

    elif scheduler_name == "MultiStepLR":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=OPTIMIZER_CONFIG["factor"]
        )
    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")

    if args['warmup']:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=OPTIMIZER_CONFIG["warmup_round"],
                                                  after_scheduler=scheduler)
    return scheduler


def generate_metric(args):
    dataset_name = args['dataset']
    if dataset_name in ['shakespeare', 'mnist', 'cifar10', 'cifar100', 'svhn',
                        'tinyImageNet', 'ImageNet', 'tinyShakespeare']:
        metric = accuracy
    elif dataset_name in ['wikiText2']:
        metric = perplexity
    else:
        raise NotImplementedError("There is non-valid metric to support this dataset")

    return metric


def generate_criterion(args):

    dataset_name = args['dataset']
    device = args['device']

    if dataset_name in ['shakespeare']:
        all_characters = string.printable
        labels_weight = torch.ones(len(all_characters), device=device)
        for char in CHARACTERS_WEIGHTS:
            labels_weight[all_characters.index(char)] = CHARACTERS_WEIGHTS[char]

        labels_weight = labels_weight * 8

        criterion = nn.CrossEntropyLoss(reduction="none", weight=labels_weight).to(device)
    elif dataset_name in ['mnist', 'cifar10', 'cifar100', 'svhn', 'tinyImageNet',
                          'ImageNet', 'wikiText2', 'tinyShakespeare']:
        criterion = nn.CrossEntropyLoss(reduction="none").to(device)
    else:
        raise NotImplementedError("There is non-valid criterion to support this dataset")

    return criterion