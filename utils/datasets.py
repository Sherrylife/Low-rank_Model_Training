import os
import pickle
import string
import torch
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    def __init__(self, text, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param text: .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.chunk_len = chunk_len

        self.text = text
        self.n_characters = len(self.text)

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        # return self.inputs[idx], self.targets[idx], idx
        return self.inputs[idx], self.targets[idx]


def batchify(data, bsz):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


class BatchDataset(Dataset):
    """
    For Wikitext2 dataset, the detailed NLP task
    is filling masked text. So the input and the label
    are the same.
    """
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0].size(1)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        """
        input: (batch_size, sequence_length)
        label: (batch_size, sequence_length)
        :param index:
        :return:
        """
        seq_length = min(self.seq_length, self.S - index)
        input = self.dataset[0][:, self.idx[index]:self.idx[index] + seq_length]
        label = self.dataset[1][:, self.idx[index]:self.idx[index] + seq_length]
        return input, label

