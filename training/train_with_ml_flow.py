import mlflow
import torch
import torch.nn as nn
from training.train import train_model
from data.tokenizer import SimpleTokenizer
from sklearn.model_selection import train_test_split


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

    # Train/Test split
    train_text, test_text = train_test_split(
            text, test_size=0.2, random_state=42)

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

        # Evaluate on test set
        test_loss = evaluate_model(model, test_text, tokenizer, vocab_size, max_seq_len)

        mlflow.log_metric("test_loss", test_loss)

        print("✅ Model logged to MLflow")

    return model


def evaluate_model(model, tokenizer, test_text, max_seq_len, vocab_size):
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss = 0
    count = 0

    with torch.no_grad():
        input_ids = tokenizer.encode(test_text)
        # Loop over the test text in chunks of size `max_seq_len`
        for i in range(0, len(input_ids) - max_seq_len, max_seq_len):
            chunk = input_ids[i:i + max_seq_len]
            input_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0).to(device)

            output = model(input_tensor)
            target = input_tensor[:, 1:]
            logits = output[:, :-1, :]

            logits = logits.reshape(-1, vocab_size)
            target = target.reshape(-1)

            loss = loss_fn(logits, target)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count if count > 0 else float('inf')
    return avg_loss
