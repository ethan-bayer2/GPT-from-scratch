import mlflow
import torch
from training.train import train_model
from data.tokenizer import SimpleTokenizer


def train_and_log_model(
    text,
    embedding_dim=64,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    max_seq_len=64,
    epochs=5,
    lr=1e-3,
    experiment_name="GPT-From-Scratch"
):
    # Prepare tokenizer and vocab
    tokenizer = SimpleTokenizer(text)
    vocab_size = len(tokenizer)

    # Set experiment
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "embedding_dim": embedding_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "epochs": epochs,
            "learning_rate": lr,
        })

        # Train the model
        model = train_model(
            text=text,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            block_size=max_seq_len,  # same as max_seq_len
            max_seq_len=max_seq_len,
            epochs=epochs,
            lr=lr,
        )

        # Define model name for registry
        registered_model_name = "GPTFromScratch"

        # Log model to MLflow
        input_example = torch.randint(
                0, vocab_size, (1, max_seq_len)).numpy()
        mlflow.pytorch.log_model(
                model,
                artifact_path="gpt_model",
                registered_model_name=registered_model_name,
                input_example=input_example)

        print("✅ Model logged to MLflow")

    return model
