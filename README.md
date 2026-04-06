# PLN Monorepo

Repositorio único para las cuatro prácticas de Procesamiento del Lenguaje Natural.

## Integrantes

- Héctor García Rincón
- Pablo Manuel Rodríguez Sosa

## Estructura

```text
.
├── p1/  # Bot negociador autónomo
├── p2/  # Materiales y memoria de audio
├── p3/  # Conversor UTF-8 <-> PLNCG26
└── p4/  # Buscador clásico + semántico + RAG sobre el Quijote
```

## Cómo usar uv sin romper el repo

Cada práctica tiene su propio `pyproject.toml`. La raíz ya no actúa como proyecto único para evitar que las dependencias de una práctica contaminen a otra.

La regla es:

```bash
uv sync --project pX
uv run --project pX ...
```

Eso hace que:

- `p1` use su entorno para `requests`, `pydantic-ai`, `ollama`, `rich`, etc.
- `p2` quede como entrega de materiales sin imponer dependencias.
- `p3` mantenga un entorno mínimo con `typer`.
- `p4` tenga su propio stack con `spacy`, `textual`, `dynaconf` y `numpy`.

## Uso rápido por práctica

### P1

```bash
uv sync --project p1
uv run --project p1 python p1/main.py --help
```

Detalles en `p1/README.md`.

### P2

No requiere ejecución con `uv`. La práctica está documentada en `p2/README.md`.

### P3

```bash
uv sync --project p3
uv run --project p3 python p3/main.py --help
```

Detalles en `p3/README.md`.

### P4

```bash
uv sync --project p4
uv run --project p4 python -m spacy download es_core_news_lg
uv run --project p4 fdi-pln-2602-p4
```

Detalles en `p4/README.md`.

## Objetivo del refactor

El objetivo de este refactor es que las cuatro prácticas coexistan en un solo repositorio de GitHub sin mezclar requisitos, entornos ni comandos. Ahora cada carpeta puede sincronizarse, ejecutarse y evolucionar por separado.
