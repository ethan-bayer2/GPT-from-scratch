from training.train import train_model
from data.tokenizer import SimpleTokenizer
from model.gpt import GPT


def test_train_model_runs():
    text = "hello world. this is a test of the emergency broadcast system."
    tokenizer = SimpleTokenizer(text)

    model = train_model(
        text=text,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        embedding_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=2,
        block_size=8,         # This is max_seq_len for the dataset
        max_seq_len=8,        # This is batch_size
        epochs=1,
        lr=1e-3,
        device="cpu"          # Use CPU for testing
    )

    assert isinstance(model, GPT)
    print("✅ Training test passed")


if __name__ == "__main__":
    test_train_model_runs()
