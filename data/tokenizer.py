class SimpleTokenizer:
    def __init__(self, text: str):
        """Build vocab from provided text"""
        self.vocab = sorted(set(text))  # Step 1
        self.token_to_id = {ch: i for i, ch in enumerate(self.vocab)}  # Step 2
        self.id_to_token = {
                i: ch for ch, i in self.token_to_id.items()
                }  # Step 3

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        """Convert string into list of token IDs"""
        return [self.token_to_id[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        """Convert list of token IDs back to string"""
        return ''.join(self.id_to_token[token] for token in tokens)
