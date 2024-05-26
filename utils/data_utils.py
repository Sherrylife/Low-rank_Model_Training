import os.path
import numpy as np
import torch.utils.data

from utils.datasets import *
from torch.utils.data.sampler import SubsetRandomSampler

from utils.experiment_config import *
from torchvision import datasets
import torchvision.transforms as transforms


def train_val_split_text(total_text, frac):
    """
    splits role text data into a set of training lines (the first `frac` of lines for the role),
     and validation lines (the last 1 - `frac`, rounded up to at least one line)
    :param total_text: raw text data
    :type total_text: str
    :param frac: training fraction
    return `train_text`, `val_text`
    """
    assert 0 < frac < 1 # `frac` should be in (0, 1)
    all_lines = total_text.split('\n')[:-1]
    n_lines = len(all_lines)

    n_test_lines = max(1, int((1-frac)*n_lines))
    n_train_lines = n_lines - n_test_lines

    train_lines = all_lines[:n_train_lines]
    val_lines = all_lines[n_train_lines:]

    train_text = '\n'.join(train_lines)
    val_text = '\n'.join(val_lines)

    return train_text, val_text


def get_loader(dataset_name, path, batch_size, is_train=True, if_need_validation=False,
               val_ratio=0.2, num_workers=0):
    """
    construct a torch.utils.DataLoader object from the given path
    :param dataset_name:
    :param path:
    :param batch_size:
    :param is_train:
    :param if_need_validation:
    :param val_ratio:
    :return:
    """
    if dataset_name in ['shakespeare', 'tinyShakespeare']:
        with open(path, 'r') as f:
            total_text = f.read()
        if is_train is True:
            if if_need_validation is True:
                train_text, val_text = train_val_split_text(total_text, val_ratio)
            else:
                train_text, val_text = total_text, None

            train_set = CharacterDataset(train_text, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                    shuffle=True, drop_last=False, num_workers=num_workers)

            if val_text is not None:
                val_set = CharacterDataset(val_text, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                    shuffle=False, drop_last=False, num_workers=num_workers)
            else:
                val_loader = None
            return train_loader, val_loader
        else:
            test_text = total_text
            test_set = CharacterDataset(test_text, chunk_len=SHAKESPEARE_CONFIG["chunk_len"])
            return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=num_workers)

    elif dataset_name == 'wikiText2':
        if is_train:
            train_data = torch.load(os.path.join(path, 'train.pt'))
            train_data = batchify(train_data, bsz=batch_size) # (sequence, batch_size)
            train_data = train_data.permute(1, 0) # (batch_size, sequence)
            train_data = [train_data.clone(), train_data.clone()] # the input and the label are the same
            train_loader = BatchDataset(dataset=train_data, seq_length=WIKITEXT2_TRANSFORMER_CONFIG["bptt"])
            if if_need_validation:
                val_data = torch.load(os.path.join(path, 'valid.pt'))
                val_data = batchify(val_data, bsz=batch_size)  # (sequence, batch_size)
                val_data = val_data.permute(1, 0)  # (batch_size, sequence)
                val_data = [val_data.clone(), val_data.clone()] # the input and the label are the same
                val_loader = BatchDataset(dataset=val_data, seq_length=WIKITEXT2_TRANSFORMER_CONFIG["bptt"])
            else:
                val_loader = None
            return train_loader, val_loader
        else:
            test_data = torch.load(os.path.join(path, 'test.pt'))
            test_data = batchify(test_data, bsz=batch_size) # (sequence, batch_size)
            test_data = test_data.permute(1, 0) # (batch_size, sequ
            test_data = [test_data.clone(), test_data.clone()]
            test_loader = BatchDataset(dataset=test_data, seq_length=WIKITEXT2_TRANSFORMER_CONFIG["bptt"])
            return test_loader

    elif dataset_name == 'cifar10':
        os.makedirs(path, exist_ok=True)
        if is_train:
            train_data = datasets.CIFAR10(
                root=path,
                train=True,
                download=True,
                transform=transforms.Compose(TRANSFORM_CONFIG["cifar10_train"])
            )
            # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
            if if_need_validation:
                val_data = datasets.CIFAR10(
                    root=path,
                    train=True,
                    download=True,
                    transform=transforms.Compose(TRANSFORM_CONFIG["cifar10_train"])
                )
                num_train = len(train_data)
                indices = list(range(num_train))
                split = int(np.floor(val_ratio * num_train))
                np.random.seed(12345)
                np.random.shuffle(12345)
                train_idx, valid_idx = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers
                )
                val_loader = torch.utils.data.DataLoader(
                    val_data, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers
                )
            else:
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, num_workers=num_workers
                )
                val_loader = None

            return train_loader, val_loader
        else:
            test_data = datasets.CIFAR10(
                root=path,
                train=False,
                download=True,
                transform=transforms.Compose(TRANSFORM_CONFIG["cifar10_test"])
            )
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                      num_workers=num_workers)
            return test_loader

    elif dataset_name == 'cifar100':
        os.makedirs(path, exist_ok=True)
        if is_train:
            train_data = datasets.CIFAR100(
                root=path,
                train=True,
                download=True,
                transform=transforms.Compose(TRANSFORM_CONFIG["cifar100_train"])
            )
            if if_need_validation:
                val_data = datasets.CIFAR100(
                    root=path,
                    train=True,
                    download=True,
                    transform=transforms.Compose(TRANSFORM_CONFIG["cifar100_train"])
                )
                num_train = len(train_data)
                indices = list(range(num_train))
                split = int(np.floor(val_ratio * num_train))
                np.random.seed(12345)
                np.random.shuffle(12345)
                train_idx, valid_idx = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_idx)
                valid_sampler = SubsetRandomSampler(valid_idx)
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers
                )
                val_loader = torch.utils.data.DataLoader(
                    val_data, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers
                )
            else:
                train_loader = torch.utils.data.DataLoader(
                    train_data, batch_size=batch_size, num_workers=num_workers
                )
                val_loader = None

            return train_loader, val_loader
        else:
            test_data = datasets.CIFAR100(
                root=path,
                train=False,
                download=True,
                transform=transforms.Compose(TRANSFORM_CONFIG["cifar100_test"])
            )
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                      num_workers=num_workers)
            return test_loader

    else:
        raise NotImplementedError("Do not support this dataset.")




def fetch_dataset(args):
    """
    constructs lists of 'torch.utils.DataLoader' object from the given dataset name;
    :param args: experiment setting
    :return:
        train_loader, val_loader, test_loader
        (List[torch.utils.DataLoader], List[torch.utils.DataLoader], List[torch.utils.DataLoader])
    """
    dataset_name = args['dataset']
    batch_size = args['B']
    dataloader_num_workers = args['num_workers']

    is_validation = False

    if dataset_name in ['shakespeare', 'tinyShakespeare']:
        train_loader, val_loader = get_loader(
            dataset_name=dataset_name,
            path=os.path.join(f'dataset/{dataset_name}/postProcess_data/train', 'train.txt'),
            batch_size=batch_size,
            is_train=True,
            if_need_validation=is_validation,
            num_workers=dataloader_num_workers
        )

        test_loader = get_loader(
            dataset_name=dataset_name,
            path=os.path.join(f'dataset/{dataset_name}/postProcess_data/test', 'test.txt'),
            batch_size=batch_size,
            is_train=False,
            if_need_validation=False,
            num_workers=dataloader_num_workers
        )
    elif dataset_name == 'wikiText2':
        train_loader, val_loader = get_loader(
            dataset_name=dataset_name,
            path=f'dataset/wikitext2/processed/',
            batch_size=batch_size,
            is_train=True,
            if_need_validation=False,
            num_workers=dataloader_num_workers
        )
        test_loader = get_loader(
            dataset_name=dataset_name,
            path=f'dataset/wikitext2/processed/',
            batch_size=batch_size,
            is_train=False,
            if_need_validation=False,
            num_workers=dataloader_num_workers
        )

    elif dataset_name in ['cifar10', 'cifar100']:
        train_loader, val_loader = get_loader(
            dataset_name=dataset_name,
            path=f'dataset/{dataset_name}/raw_data/',
            batch_size=batch_size,
            is_train=True,
            if_need_validation=is_validation,
            num_workers=dataloader_num_workers,
        )
        test_loader = get_loader(
            dataset_name=dataset_name,
            path=f'dataset/{dataset_name}/raw_data/',
            batch_size=batch_size,
            is_train=False,
            if_need_validation=False,
            num_workers=dataloader_num_workers,
        )
    else:
        raise NotImplementedError(f"{dataset_name} not recognized dataset; please check the dataset name")

    return train_loader, val_loader, test_loader
