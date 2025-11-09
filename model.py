import numpy as np
from torch import nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, cfg, embed_dim, n_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prenorm1 = nn.LayerNorm(embed_dim)
        self.ma1 = MultiAttention(cfg, embed_dim, n_head)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.postnorm1 = nn.LayerNorm(embed_dim)
        self.prenorm2 = nn.LayerNorm(embed_dim)
        self.ma2 = MultiAttention(cfg, embed_dim, n_head)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.postnorm2 = nn.LayerNorm(embed_dim)
        self.lin1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(4 * embed_dim, embed_dim)
        self.postnormlin = nn.LayerNorm(embed_dim)

    def forward(self, x):
        y = self.prenorm1(x)
        y = self.drop1(self.ma1(y)) + x
        y = self.postnorm1(y)
        z = self.prenorm2(y)
        z = self.drop2(self.ma2(z)) + y
        z = self.postnorm2(z)
        a = self.lin1(z)
        a = self.relu(a)
        a = self.lin2(a) + z
        a = self.postnormlin(a)
        return a


class MultiAttention(nn.Module):
    def __init__(self, cfg, feature_size, n_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            feature_size % n_head == 0
        ), f"Feature size should be a multiple of n_head, got feature={feature_size} and n_head={n_head}"
        self.heads = [AttentionHead(cfg, feature_size, n_head) for _ in range(n_head)]

    def forward(self, x):
        out_heads = [head(x) for head in self.heads]
        return torch.cat(out_heads, dim=-1)


class AttentionHead(nn.Module):
    def __init__(self, cfg, feature_size, n_head):
        super().__init__()
        self.key_linear = nn.Linear(feature_size, feature_size // n_head, bias=False)
        self.query_linear = nn.Linear(feature_size, feature_size // n_head, bias=False)
        self.value_linear = nn.Linear(feature_size, feature_size // n_head, bias=False)
        self.register_buffer(
            "triang", torch.tril(torch.ones((cfg.block_size, cfg.block_size)))
        )

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        key = self.key_linear(x)
        query = self.query_linear(x)
        mat = query @ key.transpose(-2, -1)
        value = self.value_linear(x)
        mat = mat.masked_fill(self.triang[:T, :T] == 0, -torch.inf) / np.sqrt(C)
        mat = torch.softmax(mat, dim=-1)
        mat = self.dropout(mat)
        return mat @ value  # B, T, feature_size


class LargeLanguageModel(nn.Module):
    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.cfg = cfg
        self.embedding_table = nn.Embedding(vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(vocab_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            *[AttentionBlock(cfg, cfg.n_embed, cfg.n_head) for _ in range(cfg.n_block)]
        )
        self.norm = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        tok_emb = self.embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T))
        emb = tok_emb + pos_emb
        emb = self.blocks(emb)
        emb = self.norm(emb)
        logits = self.lm_head(emb)

        B, T, C = logits.shape
        if y is None:
            return logits, None
        dbis = logits.view(B * T, C)
        y = y.view(B * T)
        loss = torch.nn.functional.cross_entropy(dbis, y)
        return logits, loss

    def generate(self, x, max_token):

        for _ in range(max_token):
            pred, _ = self(x[:, -self.cfg.block_size :])
            pred = pred[:, -1, :]
            pred = nn.functional.softmax(pred, dim=-1)
            pred = torch.multinomial(pred, 1)
            x = torch.cat((x, pred), dim=-1)
        return x
