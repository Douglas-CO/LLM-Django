# LLM-Django/api/views.py

import torch
import json
from django.http import JsonResponse
from model import TinyGPT
from .verificar_frase import verificar_frase  # 👈 importar la función de verificación

# ---------------------------
# Cargar vocabulario
# ---------------------------
with open("checkpoints/vocab.json", "r", encoding="utf-8") as f:
    itos = json.load(f)  # id -> caracter
stoi = {ch:int(i) for i,ch in itos.items()}  # caracter -> id
vocab_size = len(itos)

# ---------------------------
# Cargar modelo
# ---------------------------
model = TinyGPT(vocab_size=vocab_size)
model.load_state_dict(torch.load("checkpoints/tiny_llm.pt", map_location="cpu"))
model.eval()

# ---------------------------
# Funciones de codificación
# ---------------------------
def encode(s):
    """Convierte string en lista de IDs; ignora caracteres desconocidos"""
    s = s.lower()
    return [stoi[c] for c in s if c in stoi]

def decode(ids):
    """Convierte lista de IDs a string"""
    return "".join(itos[str(i)] if isinstance(itos, dict) else itos[i] for i in ids)

# ---------------------------
# Endpoint /api/generate/
# ---------------------------
@torch.no_grad()
def generate(request):
    prompt = request.GET.get("prompt", "")
    if prompt == "":
        return JsonResponse({"error": "No se proporcionó prompt"}, status=400)

    idx_list = encode(prompt)
    if len(idx_list) == 0:
        return JsonResponse({"error": "El prompt no contiene caracteres válidos"}, status=400)

    # Convertir a tensor
    idx = torch.tensor([idx_list], dtype=torch.long)

    try:
        out = model.generate(idx, max_new_tokens=100)[0].tolist()
        completion = decode(out)
    except Exception as e:
        return JsonResponse({"error": f"Error generando texto: {str(e)}"}, status=500)

    return JsonResponse({"prompt": prompt, "completion": completion})

# ---------------------------
# Endpoint /api/verificar_frase/
# ---------------------------
def verificar_frase_view(request):
    prompt = request.GET.get("prompt", "")
    if not prompt:
        return JsonResponse({"error": "No se proporcionó prompt"}, status=400)
    
    respuesta = verificar_frase(prompt)
    return JsonResponse({"respuesta": respuesta})
