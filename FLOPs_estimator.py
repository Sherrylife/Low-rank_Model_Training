import torch

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from utils.model_utils import generate_model

import argparse







"""
NOTE: You may suffer a warning "Unsupported operator aten::add encountered 8 time(s)", according to
https://github.com/facebookresearch/fvcore/issues/98, this happens when some formula is not implemented 
(the model you use is a separate operator). In ResNet, this formula is "out += shortcut"
"""



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='centralized training')
    parser.add_argument('--method', default='low_rank',
                        help='the method to be used, possible choices are '
                             'training the original full model or the low-rank model',
                        choices=['original', 'low_rank'], type=str)
    parser.add_argument('--model', default='resnet101',
                        help='the model architecture to be used, possible choices are '
                             '`mlp`, `cnn`, `vgg16`, `resnet18`, `transformer` and so on',
                        choices=['mlp', 'cnn', 'vgg11', 'vgg16', 'resnet18',
                                 'resnet101', 'transformer', 'lstm', 'vit'], type=str)
    parser.add_argument('--dataset', default='cifar10', help='the dataset to be used',
                        choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'wikiText2',
                                 'reddit', 'tinyShakespeare', 'shakespeare'], type=str)
    parser.add_argument('--device', default='cuda:1', type=str, choices=['cpu', 'cuda', 'cuda:0',
                                                                         'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4',
                                                                         'cuda:5', 'cuda:6', 'cuda:7'],
                        help='device for training model')
    # the hyper-parameter for low-rank model
    parser.add_argument('--ratio_LR', default=0.2, type=float,
                        help='low-rank ratio, a hyper parameter for low-rank model')
    parser.add_argument('--decom_rule', default=[0, 0], nargs='+', type=int,
                        help='a hyper parameter for the technique of "hybrid model architecture", '
                             'where the first K layers in the model are not decomposed with low-rank technique. '
                             'decom_rule is a 2-tuple like (block_index, layer_index). '
                             'For resnet18, block_index is selected from [0,1,2,3] and layer_index is selected from [0,1].'
                             'In resnet18 model which has 18 layers, the 1st layer and the 18th layer'
                             'will be not decomposed, and the value of decom_rule can be: '
                             '`0 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the first blocks (actually the 2nd layer in the original model), '
                             '`0 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the first blocks (actually the 4th layer in the original model), '
                             '`1 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the second blocks (actually the 6th layer in the original model), '
                             '`1 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the second blocks (actually the 8th layer in the original model), '
                             '...'
                             '`3 0`: which means we start decomposing resnet18 at the first residual block '
                             'in the fourth blocks (actually the 14th layer in the original model), '
                             '`3 1`: which means we start decomposing resnet18 at the second residual block '
                             'in the fourth blocks (actually the 16th layer in the original model), '
                        )

    args = vars(parser.parse_args())

    model = generate_model(args)

    model.train()

    if args['model'] == 'lstm' and args['method'] != 'original':
        raise NotImplementedError("Don't support the low-rank LSTM model")

    if args['dataset'] in ['cifar10', 'cifar100', 'svhn']:
        inputs = torch.randn((1, 3, 32, 32)).to(args['device']) # (batch, channel, height, weight)
    elif args['dataset'] in ['tinyImageNet', 'ImageNet']:
        inputs = torch.randn((1, 3, 64, 64)).to(args['device']) # (batch, channel, height, weight)
    elif args['dataset'] in ['shakespeare', 'tinyShakespeare']:
        inputs = torch.randint(1, 64, (1, 64)).to(args['device']) # (batch, chunk_length)
    elif args['dataset'] in ['wikiText2']:
        inputs = torch.randint(1, 64, (1, 64)).to(args['device']) # (batch, chunk_length)
    else:
        raise NotImplementedError

    flops = FlopCountAnalysis(model, inputs)
    print(flops.total())
    print(flop_count_table(flops))


