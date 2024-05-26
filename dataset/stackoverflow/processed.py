import numpy as np
import h5py
import os
import torch
from tqdm.auto import tqdm

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

file = h5py.File('./stackoverflow_train.h5')

with open('./stackoverflow.word_count', 'r') as f:
    word_counts = list(map(lambda x: (x.split('\t')[0], x.split('\t')[1]), f.readlines()))

class Vocab:
    def __init__(self):
        self.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        self.symbol_to_index = {u'<ukn>': 0, u'<eos>': 1}
        self.index_to_symbol = [u'<ukn>', u'<eos>']
        self.symbol_list = set(self.index_to_symbol)
        self.vocab_size = 10000

    def add(self, symbol):
        if symbol not in self.symbol_list and symbol not in self.filters:
            self.index_to_symbol.append(symbol)
            self.symbol_to_index[symbol] = len(self.index_to_symbol) - 1
            self.symbol_list.add(symbol)
        return

    def delete(self, symbol):
        if symbol in self.symbol_to_index:
            self.index_to_symbol.remove(symbol)
            self.symbol_to_index.pop(symbol, None)
        return

    def __len__(self):
        return len(self.index_to_symbol)

    def __getitem__(self, input):
        if input not in self.symbol_list:
            output = self.symbol_to_index[u'<ukn>']
        else:
            output = self.symbol_to_index[input]
        return output

    def __contains__(self, input):
        if isinstance(input, int):
            exist = len(self.index_to_symbol) > input >= 0
        elif isinstance(input, str):
            exist = input in self.symbol_to_index
        else:
            raise ValueError('Not valid data type')
        return exist

vocab = Vocab()
for symbol in word_counts:
    if len(vocab.index_to_symbol) >= 10000:
        break
    vocab.add(symbol[0])

print(len(vocab.index_to_symbol))

all_data = []
for k in tqdm(file['examples'].keys()):
    client_data = []
    for i in range(len(file['examples'][k]['tokens'])):
        symbols = list(map(lambda x: vocab[x], str(file['examples'][k]['tokens'][i]).split(' ') + [u'<eos>']))
        client_data += symbols
    all_data.append(torch.tensor(client_data))

torch.save(all_data, 'stackoverflow_train.pt')
# torch.save(vocab, 'train_meta.pt')
