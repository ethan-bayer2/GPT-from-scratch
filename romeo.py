import torch
import mlflow.pytorch
from data.tokenizer import SimpleTokenizer


# Load the raw training text from file
with open("data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Rebuild tokenizer from training text
tokenizer = SimpleTokenizer(text)

# MLflow run ID or path (update this!)
RUN_ID = "c5a15240463f4b45a764f5310b310a9d"  # Replace with your run id
MODEL_URI = f"runs:/{RUN_ID}/gpt_model"

# Load model
print("Loading model from MLflow...")
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Model loaded and ready.\n")


# Inference function
def generate(
        model, tokenizer, prompt,
        max_new_tokens=50, max_seq_len=128, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[-max_seq_len:]  # truncate
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            output = model(input_tensor)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).unsqueeze(0)

        input_tensor = torch.cat(
                [input_tensor, next_token.unsqueeze(0)], dim=1)

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids)


# CLI loop
def main():
    print("Welcome to Romeo 🤖. Type your prompt (type 'exit' to quit):")
    while True:
        prompt = input("You > ")
        if prompt.lower() in {"exit", "quit"}:
            break
        output = generate(model, tokenizer, prompt, device=device)
        print("Romeo >", output)
        print()


if __name__ == "__main__":
    main()
