from model.gpt import GPT
from data.tokenizer import SimpleTokenizer


def test_gpt_generate():
    # Create sample text and tokenize
    sample_text = "Hi! my name is"
    tokenizer = SimpleTokenizer(sample_text)

    # Model config
    vocab_size = len(tokenizer.vocab)
    embedding_dim = 32
    num_heads = 4
    num_layers = 2
    ff_dim = 128
    max_seq_len = 20

    # Initialize model
    model = GPT(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer
        )

    # Run generation
    prompt = "Hi!"
    output = model.generate(prompt, tokenizer=tokenizer, max_new_tokens=5)

    # Test
    assert isinstance(output, str), \
        "Output not a string"
    assert len(output) > len(prompt), \
        "Output not longer than input"
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")


if __name__ == "__main__":
    test_gpt_generate()
