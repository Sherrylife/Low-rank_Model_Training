
"""
Modify the codes according to
https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model.low_rank_base import FactorizedLinear
from model.ViT import Attention, FeedForward

from utils.experiment_config import *

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self._init_param()

    def _init_param(self):
        self.norm.weight.data.fill_(1.0)
        self.norm.bias.data.zero_()

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class LowRankFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., ratio_LR=0.2):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(dim, hidden_dim),
            FactorizedLinear(dim, hidden_dim, low_rank_ratio=ratio_LR),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(hidden_dim, dim),
            FactorizedLinear(hidden_dim, dim, low_rank_ratio=ratio_LR),
            nn.Dropout(dropout)
        )

        self._init_param()

    def _init_param(self):
        self.net[0].linear[0].weight.data.normal_(mean=0.0, std=0.02)
        self.net[0].linear[1].weight.data.normal_(mean=0.0, std=0.02)
        self.net[3].linear[0].weight.data.normal_(mean=0.0, std=0.02)
        self.net[3].linear[1].weight.data.normal_(mean=0.0, std=0.02)


    def forward(self, x):
        return self.net(x)

    def frobenius_loss(self):
        return self.net[0].frobenius_loss() + self.net[3].frobenius_loss()

    def kronecker_loss(self):
        return self.net[0].kronecker_loss() + self.net[3].kronecker_loss()


    def L2_loss(self):
        return self.net[0].L2_loss() + self.net[3].L2_loss()


class LowRankAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., ratio_LR=0.2):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv = FactorizedLinear(dim, inner_dim * 3, bias=False, low_rank_ratio=ratio_LR)

        self.to_out = nn.Sequential(
            # nn.Linear(inner_dim, dim),
            FactorizedLinear(inner_dim, dim, low_rank_ratio=ratio_LR),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def _init_param(self):
        self.to_qkv.linear[0].weight.data.normal(mean=0.0, std=0.02)
        self.to_qkv.linear[1].weight.data.normal(mean=0.0, std=0.02)
        self.to_out[0].linear[0].weight.data.normal(mean=0.0, std=0.02)
        self.to_out[0].linear[1].weight.data.normal(mean=0.0, std=0.02)


    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def frobenius_loss(self):
        return self.to_qkv.frobenius_loss() + self.to_out[0].frobenius_loss()

    def kronecker_loss(self):
        return self.to_qkv.kronecker_loss() + self.to_out[0].kronecker_loss()

    def L2_loss(self):
        return self.to_qkv.L2_loss() + self.to_out[0].L2_loss()



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., decom_rule=None, ratio_LR=0.2, args=None):
        super().__init__()

        self.start_decom_idx = decom_rule[1]
        self.device = args['device']
        self.ratio_LR = ratio_LR


        assert self.start_decom_idx <= depth, "start decom idx is wrong in Vision Transformer"

        self.layers = nn.ModuleList([])
        for idx in range(depth):
            if idx < self.start_decom_idx:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, LowRankAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, ratio_LR=self.ratio_LR)),
                    PreNorm(dim, LowRankFeedForward(dim, mlp_dim, dropout=dropout, ratio_LR=self.ratio_LR))
                ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def frobenius_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for idx, (attn, ff) in enumerate(self.layers):
            if isinstance(attn.fn, LowRankAttention):
                loss += attn.fn.frobenius_loss()
            if isinstance(ff.fn, LowRankFeedForward):
                loss += ff.fn.frobenius_loss()

        return loss

    def kronecker_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for attn, ff in self.layers:
            if isinstance(attn.fn, LowRankAttention):
                loss += attn.fn.kronecker_decay()
            if isinstance(ff.fn, LowRankFeedForward):
                loss += ff.fn.kronecker_decay()

        return loss

    def L2_decay(self):
        loss = torch.tensor(0.).to(self.device)
        for attn, ff in self.layers:
            if isinstance(attn.fn, LowRankAttention):
                loss += attn.fn.kronecker_decay()
            if isinstance(ff.fn, LowRankFeedForward):
                loss += ff.fn.kronecker_decay()

        return loss

class LowRankViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.,
                 decom_rule=None, ratio_LR=0.2, args=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,
                                       decom_rule=decom_rule, ratio_LR=ratio_LR, args=args)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def frobenius_decay(self):
        return self.transformer.frobenius_decay()

    def kronecker_decay(self):
        return self.transformer.kronecker_decay()

    def L2_decay(self):
        return self.transformer.L2_decay()


def LowRankVisionTransformer(ratio_LR, decom_rule, args=None):

    dataset_name = args["dataset"]
    if dataset_name in ['cifar10', 'cifar100']:
        config_type = CIFAR_VIT_CONFIG
    else:
        NotImplementedError("Current version only support cifar dataset for ViT")

    model = LowRankViT(
        image_size=config_type["image_size"],
        patch_size=config_type["patch_size"],
        num_classes=config_type["num_classes"][dataset_name],
        dim=config_type["embed_dim"],
        depth=config_type["depth"],
        heads=config_type["head"],
        mlp_dim=config_type["mlp_dim"],
        dropout=config_type["dropout"],
        emb_dropout=config_type["embed_dropout"],
        channels=config_type["input_channels"],
        dim_head=config_type["dim_head"],
        decom_rule=decom_rule,
        ratio_LR=ratio_LR,
        args=args,
    )
    return model