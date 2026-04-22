import torch
import torch.nn as nn

from ..attention import MultiHeadAttention


class LLM(nn.Module):
    def __init__(self, vocab_size, dim_embedding, dim_attention, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.position_embedding = nn.Embedding(max_seq_len, dim_embedding)

        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(dim_embedding, dim_attention, num_heads) for _ in range(num_layers)
        ])
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(dim_embedding) for _ in range(num_layers)
        ])

        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_embedding, dim_embedding * 4),
                nn.GELU(),
                nn.Linear(dim_embedding * 4, dim_embedding)
            )
            for _ in range(num_layers)
        ])
        self.feed_forward_norms = nn.ModuleList([
            nn.LayerNorm(dim_embedding) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(dim_embedding)
        self.output = nn.Linear(dim_embedding, vocab_size)

    def forward(self, x):
        _, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        for attn_layer, attn_norm, ff_layer, ff_norm in zip(
            self.attention_layers,
            self.attention_norms,
            self.feed_forward,
            self.feed_forward_norms,
        ):
            x = attn_norm(x + attn_layer(x, mask=causal_mask))
            x = ff_norm(x + ff_layer(x))

        x = self.output_norm(x)
        output = self.output(x)
        return output
