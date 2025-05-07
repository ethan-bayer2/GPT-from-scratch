import torch
import torch.nn as nn

from model.embedding import EmbeddingLayer
from model.transformer import Transformer
from data.tokenizer import SimpleTokenizer


class GPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads,
                 ff_dim, num_layers, max_seq_len, tokenizer: SimpleTokenizer):
        super(GPT, self).__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Positional embedding (we assume being passed tokenized text)
        self.token_embedding = EmbeddingLayer(vocab_size, embedding_dim)
        self.pos_embedding = nn.Parameter(
                torch.zeros(1, max_seq_len, embedding_dim))

        # Transformer blocks
        self.transformer = Transformer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_layers=num_layers
            )

        # Final layerNorm (standard practice in LLMs)
        self.norm = nn.LayerNorm(embedding_dim)

        # Project back to vocab size
        self.output_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        seq_len = input_ids.size(1)

        # Embed tokens and add position embeddings
        x = self.token_embedding(input_ids) + self.pos_embedding[
                :, :seq_len, :]

        # Pass through transformer
        x = self.transformer(x)

        # Normalize and project to vocabulary (in logits)
        x = self.norm(x)
        logits = self.output_head(x)

        return logits

    def generate(self, prompt: str, tokenizer: SimpleTokenizer,
                 max_new_tokens: int = 50):
        self.eval()
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token_logits = logits[:, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        generated_tokens = input_ids[0].tolist()
        return self.tokenizer.decode(generated_tokens)
