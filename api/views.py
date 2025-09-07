# LLM-Django/api/views.py

import torch
import json
from django.http import JsonResponse
from model import TinyGPT

# ---------------------------
# Cargar vocabulario para el LLM
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
# Nueva función para verificar frase y devolver respuesta humana
# ---------------------------
def buscar_frases_por_autor(autor, max_resultados=3):
    autor = autor.lower().strip()
    frases_encontradas = []

    try:
        with open("data/famosas.txt", "r", encoding="utf-8") as f:
            for line in f:
                if " - " not in line:
                    continue
                frase, autor_linea = line.strip().split(" - ")
                if autor in autor_linea.lower():
                    frases_encontradas.append(frase)
    except FileNotFoundError:
        return "No se encontró el archivo de frases famosas."

    if not frases_encontradas:
        return f"No se encontraron frases de {autor.title()}."
    
    cantidad = len(frases_encontradas)
    muestra = frases_encontradas[:max_resultados]
    return f"{autor.title()} dijo {cantidad} frases célebres, siendo algunas: {', '.join(muestra)}."

def verificar_frase_view(request):
    prompt = request.GET.get("prompt", "")
    if not prompt:
        return JsonResponse({"error": "No se proporcionó prompt"}, status=400)
    
    # Extraer autor del prompt (heurística simple)
    palabras = prompt.lower().split()
    autor = None
    autores_posibles = ["aristóteles", "einstein", "descartes", "shakespeare"]  # ampliar según tus datos
    for palabra in palabras:
        if palabra in autores_posibles:
            autor = palabra
            break

    if not autor:
        return JsonResponse({"respuesta": "No pude identificar el autor en tu pregunta."})

    respuesta = buscar_frases_por_autor(autor)
    return JsonResponse({"respuesta": respuesta})
