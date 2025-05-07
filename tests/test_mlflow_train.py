from data.tokenizer import SimpleTokenizer
from training.train_with_ml_flow import train_and_log_model


def test_train_and_log_model_runs():
    # Dummy training data
    text = "hello world! this is a test of mlflow training"

    model = train_and_log_model(
        text=text,
        embedding_dim=32,
        num_heads=2,
        ff_dim=64,
        num_layers=2,
        max_seq_len=8,
        epochs=1,
        lr=1e-3,
    )

    assert model is not None
    print("✅ MLflow training test passed")


if __name__ == "__main__":
    test_train_and_log_model_runs()
