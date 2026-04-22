import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, dim_embedding, dim_attention):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_attention = dim_attention

        self.Wq = nn.Linear(dim_embedding, dim_attention, bias=False)
        self.Wk = nn.Linear(dim_embedding, dim_attention, bias=False)
        self.Wv = nn.Linear(dim_embedding, dim_attention, bias=False)

    def forward(self, x, mask=None):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim_attention ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)

        output = torch.matmul(weights, v)

        return output
