import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.ff = FeedForward(embedding_dim, ff_dim)
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Attention sub-block and residual connection normalization
        attention_out = self.attention(x, x, x)
        x = self.layer_norm1(x + attention_out)

        # Feed-forward sublayer with residual connection normalization
        ff_out = self.ff(x)
        x = self.layer_norm2(x + ff_out)

        return x


class Transformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ff_dim, num_layers):
        super(Transformer, self).__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Pass through each transformer block
        for block in self.blocks:
            x = block(x)
        return x
