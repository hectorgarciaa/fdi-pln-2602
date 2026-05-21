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

## CLI Principal

El ejecutable principal del proyecto es:

```bash
uv run fdi-pln-2602-p5 --help
```

Subcomandos disponibles:

- `train-llm`
- `train-hp-llm`
- `generate-llm`
- `train-ner`
- `train-hp-ner`
- `detect-ner`

## Entrenamiento LLM

Entrena una unica configuracion y guarda el run en `artifacts/llm/runs/`. Si mejora el mejor global, actualiza `artifacts/llm/best/`.

Comando basico:

```bash
uv run fdi-pln-2602-p5 train-llm
```

Ejemplo con parametros explicitos:

```bash
uv run fdi-pln-2602-p5 train-llm \
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
uv run fdi-pln-2602-p5 train-hp-llm
```

Ejemplo con grid pequeno:

```bash
uv run fdi-pln-2602-p5 train-hp-llm \
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

## Generacion LLM

Generacion puntual a partir de un prompt:

```bash
uv run fdi-pln-2602-p5 generate-llm \
  --artifacts-dir artifacts/llm/best_general \
  --prompt "Hola" \
  --max-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

Modo chat interactivo:

```bash
uv run fdi-pln-2602-p5 generate-llm \
  --artifacts-dir artifacts/llm/best_general \
  --chat
```

## Comparacion De Artefactos

Sigue disponible como utilidad separada:

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

## NER

Entrenamiento simple sobre el backbone preentrenado:

```bash
uv run fdi-pln-2602-p5 train-ner
```

Grid search de NER:

```bash
uv run fdi-pln-2602-p5 train-hp-ner
```

Prediccion de entidades con el mejor modelo NER:

```bash
uv run fdi-pln-2602-p5 detect-ner \
  --artifacts-dir artifacts/ner/best_general \
  --text "Alice went to the garden."
```

Si quieres ver tambien la etiqueta predicha para cada palabra:

```bash
uv run fdi-pln-2602-p5 detect-ner \
  --artifacts-dir artifacts/ner/best_general \
  --text "Alice went to the garden." \
  --show-labels
```

Tambien se puede detectar NER sobre un fichero de texto:

```bash
uv run fdi-pln-2602-p5 detect-ner \
  --artifacts-dir artifacts/ner/best_general \
  --input-file ruta/al/texto.txt
```

## Notas

- Los notebooks de `analysis/` consumen los JSON actuales de `artifacts/llm`.
- `hyperparam_results.json` acumula resultados de ejecuciones previas.
- `chat` e `inference` cargan por defecto desde `artifacts/llm/best_general`.
- Los modulos `python -m app...` siguen existiendo, pero la forma recomendada de uso es el ejecutable `fdi-pln-2602-p5`.
