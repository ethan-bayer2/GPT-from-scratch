# config.py

# Model hyperparameters
MODEL_CONFIG = {
    "embedding_dim": 256,       # Dimension of the token embeddings
    "num_heads": 8,             # Number of attention heads
    "num_layers": 12,           # Number of transformer layers
    "feed_forward_dim": 1024,   # Dimension of the feed-forward network
    "vocab_size": 30522,        # Vocabulary size (for GPT-2, it's around 50k tokens)
    "max_seq_len": 512,         # Maximum sequence length
    "dropout_rate": 0.1         # Dropout rate
}

# Training hyperparameters
TRAINING_CONFIG = {
    "batch_size": 16,           # Batch size for training
    "learning_rate": 1e-4,      # Learning rate
    "epochs": 10,               # Number of training epochs
    "weight_decay": 0.01        # Weight decay for regularization
}

# File paths for saving models and logs
PATHS = {
    "save_model": "./models/",
    "log_dir": "./logs/"
}
