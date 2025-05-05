import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Matches each token id to an embedding with lookup table
        # Lookup table will be modified through training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
