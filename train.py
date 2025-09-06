import torch, json, os
from model import TinyGPT

# ---------------------------
# Leer dataset
# ---------------------------
with open("data/tiny.txt", "r", encoding="utf-8") as f:
    text = f.read().lower()  # todo a minúsculas

# ---------------------------
# Char-level tokenizer
# ---------------------------
chars = sorted(list(set(text)))           # lista de caracteres únicos
stoi = {ch:i for i,ch in enumerate(chars)}  # caracter -> id
itos = {i:ch for i,ch in enumerate(chars)}  # id -> caracter
vocab_size = len(chars)

def encode(s): return [stoi[c] for c in s if c in stoi]
def decode(ids): return "".join(itos[i] for i in ids)

# ---------------------------
# Convertir texto a tensor
# ---------------------------
data = torch.tensor(encode(text), dtype=torch.long)

# ---------------------------
# Función para mini batch
# ---------------------------
block_size = 128
def get_batch(batch_size=16):
    ix = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# ---------------------------
# Configurar dispositivo y modelo
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TinyGPT(vocab_size=vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

# ---------------------------
# Entrenamiento rápido
# ---------------------------
for step in range(1, 201):
    xb, yb = get_batch()
    xb, yb = xb.to(device), yb.to(device)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step}/200 - loss {loss.item():.3f}")

# ---------------------------
# Guardar modelo y vocab
# ---------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/tiny_llm.pt")
with open("checkpoints/vocab.json", "w", encoding="utf-8") as f:
    json.dump(itos, f, ensure_ascii=False)

print("✅ Entrenamiento completado")
