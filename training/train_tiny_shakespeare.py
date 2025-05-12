import os
from training.train_with_ml_flow import train_and_log_model

# Load Tiny Shakespeare dataset
data_path = os.path.expanduser(
    "~/Python/GPT-from-scratch/data/tiny_shakespeare.txt")
with open(data_path, "r") as f:
    text = f.read()

# Set model and training hyperparameters
embedding_dim = 64
num_heads = 4
ff_dim = 256
num_layers = 2
max_seq_len = 64
epochs = 5
lr = 1e-3
experiment_name = "GPT-From-Scratch-TinyShakespeare"

# Train the model and log to MLflow
model = train_and_log_model(
    text=text,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers,
    max_seq_len=max_seq_len,
    epochs=epochs,
    lr=lr,
    experiment_name=experiment_name
)
