import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature

    def forward(self, query, key, value, mask=None):
        # Calculate dot products (Q*K^T) with scaling (temp)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.temperature

        # Masking (optional) OpenAI masks below the diagonal I think
        # (future words don't influence earlier)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax for attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Multiply weights and scores for full attention pattern
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # make sure embedding_dim is divisible by num_heads
        assert embedding_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads"

        # define projection of embeddings in Query, Key, and Val space
        # Do this with pytorch to integrate with training easily
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        # Apply Scaled Dot-Product Attention
        self.attention = ScaledDotProductAttention(
                temperature=self.head_dim ** 0.5)

        # Final layer to combine multihead outputs
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, value, key, query, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]

        # Project embeddings into query, key, and value space
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Reshape inputs for multi-headed attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to get the shape
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Apply Scaled Dot-Product Attention
        output, attention_weights = self.attention(query, key, value, mask)

        # Transpose back then reshape the output
        output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.embedding_dim)

        # Pass through final linear layer
        output = self.fc_out(output)

        return output, attention_weights
