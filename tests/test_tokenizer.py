from data.tokenizer import SimpleTokenizer


def test_tokenizer():
    text = "hello world"
    tokenizer = SimpleTokenizer(text)

    encoded = tokenizer.encode("hello")
    decoded = tokenizer.decode(encoded)

    assert encoded == [
            3, 2, 4, 4, 5
            ], f"Expected [3, 2, 4, 4, 5], got {encoded}"
    assert decoded == "hello", f"Expected 'hello', got {decoded}"
    print("Tokenizer test passed!")


if __name__ == "__main__":
    test_tokenizer()
