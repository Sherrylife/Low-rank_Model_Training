'''
Modified from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
'''
import torch
import math
import torch.nn as nn
from utils.experiment_config import *
from model.low_rank_base import *
"""
number: means the number of the output channels for current layer
M: means MaxPool2d
L1: means the first linear layer
L2: means the second linear layer
L3 means the last linear layer
"""
vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'L1', 'L2', 'L3'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 'L1', 'L2', 'L3'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 'L1', 'L2', 'L3'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M', 'L1', 'L2', 'L3'],
}


class LowRankVGG(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, config,
                 ratio_LR, start_decom_idx, bias=True, batch_norm=True, args=None):
        super(LowRankVGG, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.args = args
        self.config = config
        self.ratio_LR = ratio_LR
        self.start_decom_idx = start_decom_idx
        self.bias = bias
        self.device = args['device']

        feature_layers = []
        classifier_layers = []
        cur_in_channels = self.input_size

        cur_layer = 0
        for v in self.config:
            if v != 'M':
                cur_layer = cur_layer + 1

            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v in ['L1', 'L2']:
                if cur_layer < self.start_decom_idx:
                    newLinear = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                else:
                    newLinear = FactorizedLinear(self.hidden_size, self.hidden_size, self.ratio_LR, self.bias)

                classifier_layers += [nn.Dropout(), newLinear, nn.ReLU(True)]
            elif v in ['L3']: # the last linear layer is not decomposed
                newLinear = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
                classifier_layers += [newLinear]
            else:
                if cur_layer < self.start_decom_idx:
                    newConv2d = nn.Conv2d(cur_in_channels, v, kernel_size=3, padding=1, stride=1, bias=self.bias)
                else:
                    newConv2d = FactorizedConv(cur_in_channels, v, kernel_size=3, padding=1, stride=1, bias=self.bias)

                if batch_norm:
                    feature_layers += [newConv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    feature_layers += [newConv2d, nn.ReLU(inplace=True)]

                cur_in_channels = v

        self.feature_layers = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)
        self._init_model_parameters()

    def _init_model_parameters(self):
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def recover_low_rank_layer(self):
        for idx, layer in enumerate(self.feature_layers):
            if isinstance(layer, FactorizedConv):
                W = self.feature_layers[idx].recover()
                self.feature_layers[idx] = nn.Conv2d(
                    layer.in_channels, layer.out_channels, kernel_size=3, padding=1, bias=self.bias)
                self.feature_layers[idx].weight.data = W

        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, FactorizedLinear):
                W = self.classifier[idx].recover()
                self.classifier[idx] = nn.Linear(
                    layer.input_size, layer.output_size, bias=layer.bias)
                self.classifier[idx].weight.data = W

    def decom_original_layer(self, ratio_LR=0.2):
        for idx, layer in enumerate(self.feature_layers):
            if isinstance(layer, nn.Conv2d):
                a, b, c, d = layer.weight.shape # (out_size, in_size, k, k)
                dim1, dim2 = a * c, b * d
                rank = int(min(dim1, dim2) * ratio_LR)
                W = layer.weight.data.reshape(dim1, dim2)
                U, S, V = torch.svd(W)
                sqrtS = torch.diag(torch.sqrt(S[:rank]))
                new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
                self.feature_layers[idx] = FactorizedConv(
                    in_channels=a,
                    out_channels=b,
                    kernel_size=c,
                    low_rank_ratio=ratio_LR,
                    stride=1,
                    bias=self.bias,
                    padding=1,
                )
                self.feature_layers[idx].conv[0].weight.data = new_V.reshape(b, c, 1, rank).permute(3, 0, 2, 1)
                self.feature_layers[idx].conv[1].weight.data = new_U.reshape(a, c, rank, 1).permute(0, 2, 1, 3)

        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, nn.Linear) and idx < len(self.classifier) - 1: # the last linear layer will be not factorized
                a, b = layer.weight.shape # (out_size, in_size)
                rank = int(min(a, b) * ratio_LR)
                W = layer.weight.data
                U, S, V = torch.svd(W)
                sqrtS = torch.diag(torch.sqrt(S[:rank]))
                new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
                self.classifier[idx] = FactorizedLinear(
                    input_size=b,
                    output_size=a,
                    low_rank_ratio=ratio_LR,
                    bias=self.bias,
                )
                self.classifier[idx].linear[0].weight.data = new_V
                self.classifier[idx].linear[1].weight.data = new_U



    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for idx, layer in enumerate(self.feature_layers):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.frobenius_loss()
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.frobenius_loss()
        return loss

    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for idx, layer in enumerate(self.feature_layers):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.kronecker_loss()
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.kronecker_loss()
        return loss

    def L2_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for idx, layer in enumerate(self.feature_layers):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.L2_loss()
        for idx, layer in enumerate(self.classifier):
            if isinstance(layer, FactorizedConv) or isinstance(layer, FactorizedLinear):
                loss += layer.L2_loss()
        return loss



    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class OriginalVGG(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, config,
                 batch_norm=True, args=None):
        super(OriginalVGG, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.args = args
        self.config = config

        all_layers = []
        feature_layers = []
        classifier_layers = []
        cur_in_channels = self.input_size

        for v in self.config:
            if v == 'M':
                feature_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v in ['L1', 'L2']:
                classifier_layers += [
                    nn.Dropout(),
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.ReLU(True),
                ]
            elif v == 'L3':
                classifier_layers += [nn.Linear(self.hidden_size, self.output_size)]
            else:
                conv2d = nn.Conv2d(cur_in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    feature_layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    feature_layers += [conv2d, nn.ReLU(inplace=True)]

                cur_in_channels = v

        all_layers = feature_layers + classifier_layers
        self._init_model_parameters()
        self.feature_layers = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(*classifier_layers)

    def _init_model_parameters(self):
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def LowRankVGG11(ratio_LR, decom_rule, args):
    start_decom_layer_idx = decom_rule[1]
    dataset_name = args['dataset']
    model = LowRankVGG(
        input_size=IMAGE_VGG_CONFIG["input_size"],
        hidden_size=IMAGE_VGG_CONFIG["hidden_size"],
        output_size=IMAGE_VGG_CONFIG["output_size"][str(dataset_name)],
        config=vgg_cfg['A'],
        ratio_LR=ratio_LR,
        start_decom_idx=start_decom_layer_idx,
        batch_norm=True,
        args=args,
    )

    return model


def LowRankVGG16(ratio_LR, decom_rule, args):
    start_decom_layer_idx = decom_rule[1]
    dataset_name = args['dataset']
    return LowRankVGG(
        input_size=IMAGE_VGG_CONFIG["input_size"],
        hidden_size=IMAGE_VGG_CONFIG["hidden_size"],
        output_size=IMAGE_VGG_CONFIG["output_size"][str(dataset_name)],
        config=vgg_cfg['D'],
        ratio_LR=ratio_LR,
        start_decom_idx=start_decom_layer_idx,
        batch_norm=True,
        args=args,
    )

def VGG11(args):
    """VGG 11-layer model (configuration "A")"""
    dataset_name = args['dataset']
    return OriginalVGG(
        input_size=IMAGE_VGG_CONFIG["input_size"],
        hidden_size=IMAGE_VGG_CONFIG["hidden_size"],
        output_size=IMAGE_VGG_CONFIG["output_size"][str(dataset_name)],
        config=vgg_cfg['A'],
        batch_norm=True,
        args=args,
    )



def VGG16(args):
    """VGG 16-layer model (configuration "D")"""
    dataset_name = args['dataset']
    return OriginalVGG(
        input_size=IMAGE_VGG_CONFIG["input_size"],
        hidden_size=IMAGE_VGG_CONFIG["hidden_size"],
        output_size=IMAGE_VGG_CONFIG["output_size"][str(dataset_name)],
        config=vgg_cfg['D'],
        batch_norm=True,
        args=args,
    )





