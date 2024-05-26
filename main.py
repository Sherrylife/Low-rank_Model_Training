import time

import torch


import numpy as np
import random
import os

from torch.utils.tensorboard import SummaryWriter
from utils.args import parse_args
from utils.logger import LoggerCreator
from utils.data_utils import fetch_dataset
from utils.model_utils import *
from utils.trainer import TrainingObject

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    args = parse_args()

    logs_path = "./logs/mylogs/"
    os.makedirs(logs_path, exist_ok=True)
    log_file_name = f"{args['dataset']}_{args['method']}_{args['model']}_{args['optim']}_T={args['T']}" + \
                     f"_B={args['B']}" + (f"_lr={args['lr']}_decom={args['decom_rule']}_ratioLR={args['ratio_LR']}"
                                          f"_regular={args['regularization']}_coef={args['coef_decay']}_seed={args['seed']}")

    my_logger = LoggerCreator.create_logger(
        log_path=os.path.join(logs_path, log_file_name),
        logging_name="low-rank model training",
    )

    set_seed(args['seed'])
    train_obj = TrainingObject(args=args, logger=my_logger)
    train_obj.train()





