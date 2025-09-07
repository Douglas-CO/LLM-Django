# Mini LLM (char-level) + Django API

Proyecto educativo que entrena un **mini LLM** (Transformer a nivel de caracteres) en un dataset pequeño y expone un endpoint en **Django** para generar texto.

```bash
source venv/bin/activate
```

## Requisitos
```bash
pip install -r requirements.txt
```

## Entrenamiento
```bash
python train.py
```
Esto creará `checkpoints/tiny_llm.pt`.

## Generación rápida por consola
```bash
python generate.py
# escribe un prompt cuando lo pida
```

## Servidor Django (API)
```bash
cd minillm
python manage.py migrate   # base de datos vacía (solo sistema)
python manage.py runserver
```
Endpoint:
- GET/POST `http://127.0.0.1:8000/api/verificar_frase/?prompt=la vida es sueño`

Respuesta JSON:
```json
{"prompt":"La vida ","completion":"La vida ... (texto generado)"}
```

> Nota: Es un modelo muy pequeño y entrenado con poco texto. Su objetivo es **didáctico**.
