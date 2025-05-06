import torch
from model.feed_forward import FeedForward


def test_feed_forward():
    embedding_dim = 8
    hidden_dim = 16

    # Init feed forward block
    ff_block = FeedForward(embedding_dim, hidden_dim)

    # Make random tensor to simulate input from attention block
    batch_size = 4
    seq_len = 5
    input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

    # Pass input tensor to feed forward block
    output_tensor = ff_block(input_tensor)

    # Check output shape
    assert output_tensor.shape == (batch_size, seq_len, embedding_dim), \
        f"Expected output shape ({batch_size}, {seq_len}, {embedding_dim})"

    # Check that output was changed
    assert not torch.allclose(input_tensor, output_tensor), \
        "Output tensor is the same as input tensor"


if __name__ == "__main__":
    test_feed_forward()
    print("FeedForward block test passed!")
