# Practica 5

Implementacion de un LLM pequeno desde cero, con entrenamiento, exploracion de hiperparametros, comparacion de artefactos, inferencia y chat.

> Asignatura: Procesamiento del Lenguaje Natural, UCM  
> Curso: 2025-2026

## Estructura

```text
p5/
├── README.md
├── pyproject.toml
├── uv.lock
├── app/
│   ├── attention/
│   ├── chat/
│   ├── inference/
│   ├── model/
│   ├── ner/
│   ├── tokenizer/
│   └── train/
├── analysis/
├── data/
└── artifacts/
    ├── llm/
    │   ├── best/
    │   ├── best_general/
    │   ├── runs/
    │   ├── artifact_comparison.json
    │   └── hyperparam_results.json
    └── ner/
        ├── best/
        ├── runs/
        └── hyperparam_results.json
```

## Instalacion

Desde la raiz del proyecto:

```bash
uv sync
```

## Artefactos

- `artifacts/llm/runs/<timestamp>/`: runs individuales del modelo de lenguaje
- `artifacts/llm/best/`: mejor run global del LLM
- `artifacts/llm/best_general/`: run elegido como mejor compromiso general
- `artifacts/llm/hyperparam_results.json`: historial acumulado del grid search del LLM
- `artifacts/llm/artifact_comparison.json`: comparacion de runs del LLM
- `artifacts/ner/runs/<timestamp>/`: runs individuales de NER
- `artifacts/ner/best/`: mejor run global de NER
- `artifacts/ner/hyperparam_results.json`: historial acumulado del grid search de NER

## Entrenamiento LLM

Entrena una unica configuracion y guarda el run en `artifacts/llm/runs/`. Si mejora el mejor global, actualiza `artifacts/llm/best/`.

Comando basico:

```bash
uv run python -m app.train.train
```

Ejemplo con parametros explicitos:

```bash
uv run python -m app.train.train \
  --data-dir data \
  --vocab-size 95 \
  --seq-len 96 \
  --dim-embedding 105 \
  --dim-attention 140 \
  --num-heads 2 \
  --num-layers 2 \
  --batch-size 16 \
  --epochs 5 \
  --learning-rate 1e-4 \
  --train-split 0.9 \
  --device cpu \
  --output-dir artifacts/llm
```

## Hyperparameter Train LLM

Lanza un grid search y acumula resultados en `artifacts/llm/hyperparam_results.json`.

Comando basico:

```bash
uv run python -m app.train.hyperparam_train
```

Ejemplo con grid pequeno:

```bash
uv run python -m app.train.hyperparam_train \
  --data-dir data \
  --output-dir artifacts/llm \
  --device cpu \
  --vocab-sizes 64,95 \
  --seq-lens 96 \
  --dim-embeddings 70,105 \
  --dim-attentions 140,210 \
  --num-heads 2 \
  --num-layers 2 \
  --batch-sizes 16 \
  --epochs-list 5 \
  --learning-rates 1e-4
```

## Comparacion De Artefactos

Evalua los runs de `artifacts/llm/runs/` y genera `artifacts/llm/artifact_comparison.json`.

```bash
uv run python -m app.train.compare_artifacts
```

Ejemplo con rutas explicitas:

```bash
uv run python -m app.train.compare_artifacts \
  --artifacts-dir artifacts/llm/runs \
  --data-dir data \
  --batch-size 16 \
  --device cpu \
  --output artifacts/llm/artifact_comparison.json
```

## Inference

Usa por defecto `artifacts/llm/best_general/`.

```bash
uv run python -m app.inference.inference \
  --prompt "Hola" \
  --max-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

Ejemplo en CPU y con semilla fija:

```bash
uv run python -m app.inference.inference \
  --prompt "Alice" \
  --max-tokens 60 \
  --temperature 0.7 \
  --top-k 40 \
  --device cpu \
  --seed 42
```

## Chat

Abre un chat interactivo usando el modelo de `artifacts/llm/best_general/`.

```bash
uv run python -m app.chat.chat
```

Escribe `salir` para terminar.

## NER

Entrenamiento simple sobre el backbone preentrenado:

```bash
uv run python -m app.ner.train
```

Grid search de NER:

```bash
uv run python -m app.ner.hyperparam_train
```

## Notas

- Los notebooks de `analysis/` consumen los JSON actuales de `artifacts/llm`.
- `hyperparam_results.json` acumula resultados de ejecuciones previas.
- `chat` e `inference` cargan por defecto desde `artifacts/llm/best_general`.
