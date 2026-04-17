# Práctica 5

Base del proyecto para la práctica 5 de PLN.

Ahora mismo se ha dejado sólo la parte de tokenización, con una versión sencilla de `MiniBPETokenizer` pensada para que un alumno la pueda leer y modificar con facilidad.

> Asignatura: Procesamiento del Lenguaje Natural, UCM  
> Curso: 2025-2026

## Estructura

```text
p5/
├── README.md
├── pyproject.toml
├── uv.lock
├── artifacts/
│   └── .gitkeep
└── p5/
    ├── __init__.py
    └── app/
        ├── __init__.py
        └── tokenizer.py
```

## Qué deja preparado esta base

- proyecto `uv` independiente, con su propio `pyproject.toml` y `uv.lock`
- paquete Python instalable para la práctica
- implementación inicial de un tokenizer BPE sencillo y didáctico

## Instalación

Desde la raíz del monorepo:

```bash
uv sync --project p5
```

## Uso rápido

Desde Python:

```python
from p5.app.tokenizer import MiniBPETokenizer

tokenizer = MiniBPETokenizer()
tokenizer.train("hola hola quijote", vocab_size=64)
ids = tokenizer.encode("hola quijote")
texto = tokenizer.decode(ids)
```

Para generar texto con un modelo entrenado:

```bash
uv run --project p5 python -m app.inference.inference \
  --prompt "en un lugar de la mancha" \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-k 40
```

## Siguiente trabajo recomendado

1. Probar el tokenizer con un corpus más grande.
2. Añadir comentarios o tests si queréis usarlo como base de entrega.
3. Cuando la parte de tokenización esté clara, construir encima el modelo.
