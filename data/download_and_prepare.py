import requests
from pathlib import Path

# Define where to save the dataset
data_dir = Path.home() / "Python" / "GPT-from-scratch" / "data"

# Download Tiny Shakespeare
url = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
output_path = data_dir / "tiny_shakespeare.txt"

if not output_path.exists():
    print("Downloading Tiny Shakespeare dataset...")
    response = requests.get(url)
    response.raise_for_status()
    output_path.write_text(response.text, encoding="utf-8")
    print(f"Saved dataset to {output_path}")
else:
    print(f"Dataset already exists at {output_path}")
