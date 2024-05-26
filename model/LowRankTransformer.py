import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.experiment_config import *
from model.low_rank_base import FactorizedLinear
from model.Transformer import TransformerEncoderLayer, Decoder, MultiheadAttention

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


class LowRankMultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, ratio_LR=0.2, bias=True):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.bias = bias
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.linear_q = FactorizedLinear(embedding_size, embedding_size, self.ratio_LR, self.bias)
        self.linear_k = FactorizedLinear(embedding_size, embedding_size, self.ratio_LR, self.bias)
        self.linear_v = FactorizedLinear(embedding_size, embedding_size, self.ratio_LR, self.bias)
        self.linear_o = FactorizedLinear(embedding_size, embedding_size, self.ratio_LR, self.bias)

        self.attention = ScaledDotProduct(temperature=(embedding_size // num_heads) ** 0.5)
        self.decom_list = ['linear_q', 'linear_k', 'linear_v', 'linear_o']

    def recover(self):
        for obj_name in self.decom_list:
            obj = self.__getattr__(obj_name)
            W = obj.recover()
            new_obj = nn.Linear(in_features=W.size(1), out_features=W.size(0), bias=self.bias)
            new_obj.weight.data = W
            self.__setattr__(obj_name, new_obj)

    def decom(self, ratio_LR=0.2):
        for obj_name in self.decom_list:
            obj = self.__getattr__(obj_name)
            a, b = obj.weight.shape  # (out_size, in_size)
            rank = int(min(a, b) * ratio_LR)
            W = obj.weight.data
            U, S, V = torch.svd(W)
            sqrtS = torch.diag(torch.sqrt(S[:rank]))
            new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
            new_obj = FactorizedLinear(
                input_size=b,
                output_size=a,
                low_rank_ratio=ratio_LR,
                bias=self.bias,
            )
            new_obj.linear[0].weight.data = new_V
            new_obj.linear[1].weight.data = new_U
            self.__setattr__(name=obj_name, value=new_obj)

    def frobenius_loss(self):
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.frobenius_loss()
            else:
                loss += obj.frobenius_loss()
        return loss

    def kronecker_loss(self):
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.kronecker_loss()
            else:
                loss += obj.kronecker_loss()
        return loss


    def L2_loss(self):
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.L2_loss()
            else:
                loss += obj.L2_loss()
        return loss


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

class LowRankTransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout, ratio_LR=0.2, bias=True):
        super().__init__()
        self.ratio_LR = ratio_LR
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_heads = num_heads

        self.mha = LowRankMultiheadAttention(embedding_size, num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        # self.linear1 = nn.Linear(embedding_size, hidden_size, bias=self.bias)
        self.linear1 = FactorizedLinear(
            input_size=embedding_size,
            output_size=hidden_size,
            low_rank_ratio=self.ratio_LR,
            bias=self.bias
        )

        self.dropout1 = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(hidden_size, embedding_size, bias=self.bias)
        self.linear2 = FactorizedLinear(
            input_size=hidden_size,
            output_size=embedding_size,
            low_rank_ratio=self.ratio_LR,
            bias=self.bias
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.activation = nn.GELU()
        self._init_param()
        self.decom_list = ['linear1', 'linear2']


    def _init_param(self):
        self.linear1.linear[0].weight.data.normal_(mean=0.0, std=0.02)
        self.linear1.linear[1].weight.data.normal_(mean=0.0, std=0.02)

        self.linear2.linear[0].weight.data.normal_(mean=0.0, std=0.02)
        self.linear1.linear[1].weight.data.normal_(mean=0.0, std=0.02)

        self.norm1.weight.data.fill_(1.0)
        self.norm1.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()

    def recover(self):
        self.mha.recover()

        W1 = self.linear1.recover()
        W2 = self.linear2.recover()

        self.linear1 = nn.Linear(in_features=W1.size(1), out_features=W1.size(0), bias=self.bias)
        self.linear2 = nn.Linear(in_features=W2.size(1), out_features=W2.size(0), bias=self.bias)
        self.linear1.weight.data = W1
        self.linear2.weight.data = W2

    def decom(self, ratio_LR=0.2):
        self.mha.decom(ratio_LR=ratio_LR)
        for obj_name in self.decom_list:
            obj = self.__getattr__(obj_name)
            a, b = obj.weight.shape # (out_size, in_size)
            rank = int(min(a, b) * ratio_LR)
            W = obj.weight.data
            U, S, V = torch.svd(W)
            sqrtS = torch.diag(torch.sqrt(S[:rank]))
            new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
            new_obj = FactorizedLinear(
                input_size=b,
                output_size=a,
                low_rank_ratio=ratio_LR,
                bias=self.bias,
            )
            new_obj.linear[0].weight.data = new_V
            new_obj.linear[1].weight.data = new_U
            self.__setattr__(name=obj_name, value=new_obj)


    def frobenius_loss(self):
        # note: use `weight` instead of `weight.data`
        loss = self.mha.frobenius_loss()
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            loss += obj.frobenius_loss()

        return loss

    def L2_loss(self):
        loss = self.mha.frobenius_loss()
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            loss += obj.L2_loss()

        return loss

    def kronecker_loss(self):
        loss = self.mha.frobenius_loss()
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            loss += obj.kronecker_loss()

        return loss


    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        attn_output, _ = self.mha(src, src, src, mask=src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class LowRankDecoder(nn.Module):
    def __init__(self, num_tokens, embedding_size, ratio_LR=0.2, bias=True):
        super().__init__()
        self.bias = bias

        self.ratio_LR = ratio_LR
        self.linear1 = FactorizedLinear(
            input_size=embedding_size,
            output_size=embedding_size,
            low_rank_ratio=self.ratio_LR,
            bias=self.bias
        )
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(embedding_size)

        # the last linear layer will not be decomposed
        self.linear2 = nn.Linear(embedding_size, num_tokens)
        self.decom_list = ['linear1']

    def frobenius_loss(self):
        # note: use `weight` instead of `weight.data`
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.frobenius_loss()
            else:
                loss += obj.frobenius_loss()

        return loss

    def L2_loss(self):
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.L2_loss()
            else:
                loss += obj.L2_loss()
        return loss

    def kronecker_loss(self):
        for idx, obj_name in enumerate(self.decom_list):
            obj = self.__getattr__(obj_name)
            if idx == 0:
                loss = obj.kronecker_loss()
            else:
                loss += obj.kronecker_loss()

        return loss


    def recover(self):
        W = self.linear1.recover()
        self.linear1 = nn.Linear(W.size(1), W.size(0), bias=self.bias)
        self.linear1.weight.data = W

    def decom(self, ratio_LR=0.2):
        a, b = self.linear1.weight.shape # (out_size, in_size)
        rank = int(min(a, b) * ratio_LR)
        W = self.linear1.weight.data
        U, S, V = torch.svd(W)
        sqrtS = torch.diag(torch.sqrt(S[:rank]))
        new_U, new_V = torch.matmul(U[:, :rank], sqrtS), torch.matmul(V[:, :rank], sqrtS).T
        self.linear1 = FactorizedLinear(
            input_size=b,
            output_size=a,
            low_rank_ratio=ratio_LR,
            bias=self.bias,
        )
        self.linear1.linear[0].weight.data = new_V
        self.linear1.linear[1].weight.data = new_U

    def forward(self, src):
        out = self.linear2(self.norm1(self.activation(self.linear1(src))))
        return out


class LowRankFillMaskTextTransformer(nn.Module):
    def __init__(self, num_tokens, embedding_size, num_heads, hidden_size,
                 num_layers, dropout, ratio_LR, decom_rule=None, args=None):
        super(LowRankFillMaskTextTransformer, self).__init__()
        self.num_tokens = num_tokens
        self.ratio_LR = ratio_LR
        self.start_decom_idx = decom_rule[1]
        self.device = args['device']

        # the `1` is the number of decoder and the `num_layers` is the number of encoder
        assert self.start_decom_idx <= num_layers, "start decom idx is wrong in Transformer"

        self.transformer_embedding = TransformerEmbedding(num_tokens, embedding_size, dropout)

        personalized_layers = []

        encoder_modules = []
        for idx in range(num_layers):
            if idx < self.start_decom_idx:
                encoder_modules.append(TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout))
            else:
                encoder_modules.append(LowRankTransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout))
                personalized_layers.append(encoder_modules[-1])

        self.transformer_encoder = nn.Sequential(*encoder_modules)

        if self.start_decom_idx <= num_layers:
            decoder = LowRankDecoder(num_tokens, embedding_size)
            self.decoder = decoder
            personalized_layers.append(decoder)
        else:
            self.decoder = Decoder(num_tokens, embedding_size)

        self.personalized_parts = nn.Sequential(*personalized_layers)

        self._init_model_parameters()


    def recover_low_rank_layer(self):
        for idx, block in enumerate(self.personalized_parts):
            if isinstance(block, (LowRankTransformerEncoderLayer, LowRankDecoder, LowRankMultiheadAttention)):
                block.recover()

    def decom_original_layer(self, ratio_LR=0.2):
        for idx, block in enumerate(self.personalized_parts):
            if isinstance(block, (LowRankTransformerEncoderLayer, LowRankDecoder)):
                block.decom(ratio_LR=ratio_LR)

    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for layer in self.personalized_parts:
            if isinstance(layer, (LowRankTransformerEncoderLayer, LowRankDecoder)):
                loss += layer.frobenius_loss()

        return loss

    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for layer in self.personalized_parts:
            if isinstance(layer, (LowRankTransformerEncoderLayer, LowRankDecoder)):
                loss += layer.kronecker_loss()

        return loss

    def L2_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for layer in self.personalized_parts:
            if isinstance(layer, (LowRankTransformerEncoderLayer, LowRankDecoder)):
                loss += layer.L2_loss()

        return loss



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


def LowRankFillTextTransformer(ratio_LR, decom_rule, args=None):

    num_tokens = WIKITEXT2_TRANSFORMER_CONFIG["num_tokens"]
    embedding_size = WIKITEXT2_TRANSFORMER_CONFIG["embedding_size"]
    num_heads = WIKITEXT2_TRANSFORMER_CONFIG["num_heads"]
    hidden_size = WIKITEXT2_TRANSFORMER_CONFIG["hidden_size"]
    num_layers = WIKITEXT2_TRANSFORMER_CONFIG["num_layers"]
    dropout_rate = WIKITEXT2_TRANSFORMER_CONFIG["dropout_rate"]
    model = LowRankFillMaskTextTransformer(
        num_tokens=num_tokens,
        embedding_size=embedding_size,
        num_heads=num_heads,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_rate,
        ratio_LR=ratio_LR,
        decom_rule=decom_rule,
        args=args,
    )

    # model.recover_low_rank_layer()
    # model.decom_original_layer(ratio_LR=ratio_LR)
    return model



