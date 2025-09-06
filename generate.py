import torch, json
from model import TinyGPT

# Cargar vocab
with open("checkpoints/vocab.json", "r", encoding="utf-8") as f:
    itos = json.load(f)
stoi = {ch:i for i,ch in itos.items()}
vocab_size = len(itos)

def encode(s): return [stoi[c] for c in s]
def decode(ids): return "".join(itos[str(i)] if isinstance(itos, dict) else itos[i] for i in ids)

# Cargar modelo
model = TinyGPT(vocab_size=vocab_size)
model.load_state_dict(torch.load("checkpoints/tiny_llm.pt", map_location="cpu"))
model.eval()

prompt = input("Prompt: ")
idx = torch.tensor([encode(prompt)], dtype=torch.long)
with torch.no_grad():
    out = model.generate(idx, max_new_tokens=100)[0].tolist()

print("\n--- Generado ---")
print(decode(out))
