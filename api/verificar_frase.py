# LLM-Django/api/verificar_frase.py

from difflib import SequenceMatcher

def verificar_frase(prompt):
    """
    Verifica si el prompt coincide con alguna frase famosa.
    Retorna un string con la coincidencia o mensaje de no encontrada.
    """
    prompt = prompt.lower().strip()
    try:
        with open("data/famosas.txt", "r", encoding="utf-8") as f:
            for line in f:
                if " - " not in line:
                    continue
                frase, autor = line.strip().split(" - ")
                if SequenceMatcher(None, prompt, frase.lower()).ratio() > 0.8:
                    return f"¡Oh! Tal vez tu frase es: '{frase}' dicha por {autor}"
    except FileNotFoundError:
        return "No se encontró el archivo de frases famosas."
    
    return "No encontramos coincidencias."
