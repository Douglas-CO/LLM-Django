from django.http import JsonResponse
from fuzzywuzzy import fuzz

FRASES_FAMOSAS = [
    "Pienso, luego existo",
    "La imaginación es más importante que el conocimiento",
    "Sé el cambio que quieres ver en el mundo"
]

def verificar_frase(request):
    texto = request.GET.get("texto", "")
    resultado = False
    frase_detectada = None
    umbral = 80
    for frase in FRASES_FAMOSAS:
        if fuzz.ratio(texto.lower(), frase.lower()) >= umbral:
            resultado = True
            frase_detectada = frase
            break
    return JsonResponse({
        "input": texto,
        "es_famosa": resultado,
        "frase_coincidente": frase_detectada
    })
