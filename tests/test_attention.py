import torch
from model.attention import MultiHeadAttention


def test_multihead_attention():
    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, embedding_dim)
    attention = MultiHeadAttention(embedding_dim, num_heads)

    output, _ = attention(x, x, x)

    # Check shape
    assert output.shape == (batch_size, seq_len, embedding_dim), \
        f"Expected output shape ({batch_size}, {seq_len}, {embedding_dim})"

    assert not torch.allclose(x, output), \
        "Expected output to differ from input"

    print("MultiHeadAttention test passed")


if __name__ == "__main__":
    test_multihead_attention()
