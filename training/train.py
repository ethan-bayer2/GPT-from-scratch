import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.gpt import GPT


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_seq_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.token_ids = tokenizer.encode(text)

    def __len__(self):
        return len(self.token_ids) - self.max_seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[
            idx: idx + self.max_seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[
            idx + 1: idx + self.max_seq_len + 1], dtype=torch.long)
        return x, y


# train model defined at top level
def train_model(
        text,
        tokenizer,
        vocab_size,
        embedding_dim,
        num_heads,
        ff_dim,
        num_layers,
        block_size,
        max_seq_len,
        epochs,
        lr,
        device="cuda" if torch.cuda.is_available() else "cpu"):
    dataset = TextDataset(text, tokenizer, block_size)
    dataloader = DataLoader(dataset, batch_size=max_seq_len, shuffle=True)

    model = GPT(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            logits = model(x)
            # Reshape for cross entropy
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))

            optimizer.zero_grad()  # Zero gradients
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return model
