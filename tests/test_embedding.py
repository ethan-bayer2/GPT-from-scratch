import torch
from model.embedding import EmbeddingLayer


def test_embedding():
    vocab_size = 10
    embedding_dim = 5
    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)

    # Example input: a tensor of token indices
    input_tensor = torch.tensor([0, 1, 2, 3, 4])

    # Get the embeddings for the input
    output = embedding_layer(input_tensor)

    # Check if the output is the expected shape
    assert output.shape == (5, embedding_dim), \
        f"Expected output shape (5, {embedding_dim}), but got {output.shape}"
    assert not torch.equal(output[0], output[1]), \
        "Embeddings for different tokens should be different."


print("Embedding layer test passed!")


if __name__ == "__main__":
    test_embedding()
