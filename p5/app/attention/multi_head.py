import torch
import torch.nn as nn

from .attention import Attention


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_embedding, dim_attention, num_heads):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_attention = dim_attention
        self.num_heads = num_heads

        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"

        self.attention_heads = nn.ModuleList([
            Attention(dim_embedding, dim_attention // num_heads) for _ in range(num_heads)
        ])
        
        self.linear = nn.Linear(dim_attention, dim_embedding)

    def forward(self, x, mask=None):
        head_outputs = [head(x, mask=mask) for head in self.attention_heads]
        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.linear(concat_output)

        return output
