import torch
from model.transformer import Transformer


def test_transformer():
    # Parameters
    batch_size = 4
    seq_len = 5
    embedding_dim = 64
    ff_dim = 256
    num_layers = 10
    num_heads = 8

    # Create the model
    transformer = Transformer(embedding_dim, num_heads, ff_dim, num_layers)

    # Create random tensor for test input
    input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

    # Pass input through transformer
    output_tensor = transformer(input_tensor)

    # Check output shape
    assert output_tensor.shape == (batch_size, seq_len, embedding_dim), \
        f"Expected output shape {(batch_size, seq_len, embedding_dim)}"

    assert len(transformer.blocks) == num_layers, \
        f"Expectected {num_layers} but got {len(transformer.blocks)}"

    print("Transformer test passed successfully")


if __name__ == "__main__":
    test_transformer()
