import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from utils.experiment_config import *


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.positional_embedding = nn.Embedding(WIKITEXT2_TRANSFORMER_CONFIG["bptt"], embedding_size)

    def forward(self, x):
        N, S = x.size()
        position = torch.arange(S, dtype=torch.long, device=x.device).unsqueeze(0).expand((N, S))
        x = self.positional_embedding(position)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, num_tokens, embedding_size, dropout):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.positional_embedding = PositionalEmbedding(embedding_size)
        self.embedding = nn.Embedding(num_tokens + 1, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = self.embedding(src) + self.positional_embedding(src)
        src = self.dropout(self.norm(src))
        return src


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        seq_len = scores.shape[-1]
        h = scores.shape[0]
        mask = torch.tril(torch.ones((h, seq_len, seq_len))).to(str(scores.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.linear_q = nn.Linear(embedding_size, embedding_size)
        self.linear_k = nn.Linear(embedding_size, embedding_size)
        self.linear_v = nn.Linear(embedding_size, embedding_size)
        self.linear_o = nn.Linear(embedding_size, embedding_size)
        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, seq_len, self.num_heads, sub_dim).permute(0, 2, 1, 3) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v, mask)
        q = self._reshape_from_batches(q)
        q = self.linear_o(q)
        return q, attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.mha = MultiheadAttention(embedding_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear1 = nn.Linear(embedding_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size, embedding_size)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.activation = nn.GELU()
        self._init_param()

    def _init_param(self):
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Decoder(nn.Module):
    def __init__(self, num_tokens, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, embedding_size)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)
        self.linear2 = nn.Linear(embedding_size, num_tokens)

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.linear1(src))))
        return out


class FillMaskTextTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size, num_layers, dropout):
        super(FillMaskTextTransformer, self).__init__()
        self.num_tokens = num_tokens
        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.decoder = Decoder(num_tokens, embedding_size)
        self._init_model_parameters()

    def _init_model_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, input, mask_rate=0.15):

        src = input.clone()
        N, S = src.size() # (batch_size, sequence)
        d = torch.distributions.bernoulli.Bernoulli(probs=WIKITEXT2_TRANSFORMER_CONFIG['mask_rate'])
        mask = d.sample((N, S)).to(src.device)
        src = src.masked_fill(mask == 1, self.num_tokens).detach()
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src)
        out = self.decoder(src) # (batch_size, sequence_length, output_size)
        out = out.permute(0, 2, 1) #(batch_size, output_size, sequence_length)
        return out


class LowRankFillMaskTextTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size,
                 num_layers, dropout, ratio_LR, decom_rule=None):
        super(LowRankFillMaskTextTransformer, self).__init__()
        self.num_tokens = num_tokens
        self.ratio_LR = ratio_LR

        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

        self.decoder = Decoder(num_tokens, embedding_size)


        self._init_model_parameters()

    def _init_model_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, input, mask_rate=0.15):

        src = input.clone()
        N, S = src.size() # (batch_size, sequence)
        d = torch.distributions.bernoulli.Bernoulli(probs=WIKITEXT2_TRANSFORMER_CONFIG['mask_rate'])
        mask = d.sample((N, S)).to(src.device)
        src = src.masked_fill(mask == 1, self.num_tokens).detach()
        src = self.transformer_embedding(src)
        src = self.transformer_encoder(src)
        out = self.decoder(src) # (batch_size, sequence_length, output_size)
        out = out.permute(0, 2, 1) #(batch_size, output_size, sequence_length)
        return out


def FillTextTransformer(args=None):
    num_tokens = WIKITEXT2_TRANSFORMER_CONFIG["num_tokens"]
    embedding_size = WIKITEXT2_TRANSFORMER_CONFIG["embedding_size"]
    num_heads = WIKITEXT2_TRANSFORMER_CONFIG["num_heads"]
    hidden_size = WIKITEXT2_TRANSFORMER_CONFIG["hidden_size"]
    num_layers = WIKITEXT2_TRANSFORMER_CONFIG["num_layers"]
    dropout_rate = WIKITEXT2_TRANSFORMER_CONFIG["dropout_rate"]
    model = FillMaskTextTransformer(
        num_tokens=num_tokens,
        embedding_size=embedding_size,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_rate,
    )
    return model
