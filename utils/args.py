import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='centralized training')
    parser.add_argument('--method', default='low_rank',
                        help='the method to be used, possible choices are '
                             'training the original full model or the low-rank model',
                        choices=['original', 'low_rank'], type=str)
    parser.add_argument('--model', default='lstm',
                        help='the model architecture to be used, possible choices are '
                             '`mlp`, `cnn`, `vgg16`, `resnet18`, `transformer` and so on',
                        choices=['mlp', 'cnn', 'vgg11', 'vgg16', 'resnet18',
                                 'resnet101', 'transformer', 'lstm', 'vit'], type=str)
    parser.add_argument('--dataset', default='shakespeare', help='the dataset to be used',
                        choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'tinyImageNet', 'wikiText2',
                                 'reddit', 'tinyShakespeare', 'shakespeare'], type=str)
    parser.add_argument('--B', default=128, help='batch size', type=int)
    parser.add_argument('--lr', default=0.01, help='learning rate', type=float)
    parser.add_argument('--T', default=10, help='training round', type=int)
    parser.add_argument('--lr_scheduler', default='CosineAnnealingLR', help='learning rate scheduler',
                        type=str, choices=['None', 'Linear', 'Sqrt', 'MultiStepLR', 'CosineAnnealingLR'])
    parser.add_argument('--optim', default='SGD', type=str,
                        help='learning rate optimizer, possible choices are '
                             'sgd with momentum (the default coefficient of momentum is 0.9), '
                             'adam, rmsProp, adamax',
                        choices=['SGD', 'Adam', 'RMSprop', 'Adamax'])
    parser.add_argument('--warmup', action='store_true')

    parser.add_argument('--eval_freq', default=1, type=int,
                        help='the frequency for evaluating the model on the test dataset')
    parser.add_argument('--seed', default=12345, type=int, help='random seed for experiment')
    parser.add_argument('--device', default='cuda:1', type=str, choices=['cpu', 'cuda', 'cuda:0',
                                                'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'],
                        help='device for training model')
    parser.add_argument('--cuda_id', default=[0], nargs='+', type=int,
                        help='specify the id of the gpu used for model training')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='num_workers for data loader in pytorch')


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
    parser.add_argument('--regularization', default='none', type=str,
                        help='the regularization technique for low-rank model training, '
                             'possible choices can be: '
                             'frobenius decay, kronecker decay, L2 weight decay, or use them both',
                        choices=['none', 'frobenius', 'kronecker', 'L2', 'L2-FD', 'L2-KD', 'FD-KD'])
    parser.add_argument('--init', type=str, default='none', choices=['SI', 'none'],
                        help='initialization scheme for low-rank model parameter, possible choices'
                             'can be spectral initialization scheme (SI) or other popular schemes (none) like '
                             'he initialization or Xavier initialization, which depend on the specific model '
                             'architecture')
    parser.add_argument('--coef_decay', default=0.0001, type=float, help='coefficient when only '
                                                                        'one regularization is used')
    parser.add_argument('--coef_L2', default=0.001, type=float, help='coefficient for L2 decay '
                                                                     'when two regularization methods are used')
    parser.add_argument('--coef_KD', default=0.0001, type=float, help='coefficient for kronecker decay'
                                                                     'when two regularization methods are used')
    parser.add_argument('--coef_FD', default=0.0001, type=float, help='coefficient for frobenius decay'
                                                                     'when two regularization methods are used')

    args = vars(parser.parse_args())

    return args
