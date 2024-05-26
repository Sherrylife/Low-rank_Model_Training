import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from model.rnn_base import *


class LowRankLinear(nn.Module):
    def __init__(self, input_size, output_size, low_rank_ratio):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.low_rank_ratio = low_rank_ratio
        self.r = int(self.output_size * self.low_rank_ratio)

        assert self.r > 0
        self.linear_VT = nn.Linear(in_features=input_size, out_features=self.r)
        self.linear_U = nn.Linear(in_features=self.r, out_features=self.output_size)

    def forward(self, input_):
        return self.linear_U(self.linear_VT(input_))


class LowRankNextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers,
                 low_rank_ratio, bias=True, device="cpu"):
        super(LowRankNextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.low_rank_ratio = low_rank_ratio

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn = LSTM_low_rank(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            low_rank_ratio=self.low_rank_ratio
        )
        self.reset_rnn_parameters()

        self.decoder = LowRankLinear(
            input_size=self.hidden_size,
            output_size=self.output_size,
            low_rank_ratio=self.low_rank_ratio
        )

    def reset_rnn_parameters(self):
        """
        initialize the model parameters of rnn
        :return:
        """
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.rnn.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_):
        encoded = self.encoder(input_) # (batch size, sequence length, embedding size): (128,80,8)
        """Using the custom LSTM based on torchScript"""
        states = [
            LSTMState(
                torch.zeros(encoded.size(0), self.hidden_size, dtype=encoded.dtype, device=encoded.device),
                torch.zeros(encoded.size(0), self.hidden_size, dtype=encoded.dtype, device=encoded.device)
            )
            for _ in range(self.n_layers)
        ]
        encoded = encoded.permute(1, 0, 2)
        output, output_state = self.rnn(encoded, states) # (seq_len, batch_size, hidden_size)
        output = self.decoder(output) # (seq_len, batch_size, output_size)
        output = output.permute(1, 2, 0) # (batch_size, output_size, seq_len)

        return output


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers,
                 bias=True, device="cpu"):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=self.embed_size,
                hidden_size=self.hidden_size,
                num_layers=self.n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_) # (batch size, sequence length, embedding size): (128,80,8)
        """Using the naive LSTM in Pytorch"""
        output, _ = self.rnn(encoded) # (batch size, sequence length, hidden size): (128, 80, 256)
        output = self.decoder(output) # (batch size, sequence length, output_size): (128,80,100)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output








