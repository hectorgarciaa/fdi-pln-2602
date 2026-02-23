# Bot Negociador Autónomo (PLN)

Bot que negocia intercambios de recursos en `fdi-pln-butler`.

## Requisitos
- Python 3.12+
- Ollama en ejecución
- Servidor Butler en ejecución

## Instalación
```bash
uv sync
```

Modelo recomendado:
```bash
ollama pull qwen3:8b
```

## Ejecución rápida
Un bot:
```bash
uv run app/main.py
```

Un bot en automático:
```bash
uv run app/main.py --alias MiBot --debug
```

Varios bots:
```bash
uv run app/test_runner.py -n 3 --consola
```

## Estructura del proyecto
```text
app/
├── main.py                         # Entry point principal
├── test_runner.py                  # Entry point multi-bot
└── pln_bot/
    ├── agente/                     # Paquete renombrado al español
    │   ├── negociador.py
    │   └── ejecutor_ronda.py       # (extraído de negociador.py)
    ├── interfaz/                   # Paquete renombrado al español
    │   ├── main.py
    │   └── test_runner.py
    ├── nucleo/                     # Paquete renombrado al español
    │   └── config.py
    ├── negociacion/                # Paquete renombrado al español
    │   ├── gestor_acuerdos.py      # (extraído de negociador.py)
    │   ├── procesador_buzon.py     # (extraído de negociador.py)
    │   ├── utilidades_mensajes.py  # (extraído de negociador.py)
    │   ├── politica_negociacion.py # (extraído de negociador.py)
    │   ├── constructor_propuestas.py # (extraído de negociador.py)
    │   └── enviador_propuestas.py  # (extraído de negociador.py)
    └── servicios/                  # Paquete renombrado al español
        ├── api_client.py
        ├── ollama_client.py
        └── servicio_analisis.py    # (extraído de negociador.py)
```

## Flujo del bot
En cada ronda:
1. Actualiza estado (`/info`, `/gente`).
2. Procesa buzón.
3. Decide aceptar/rechazar/contraofertar.
4. Envía propuestas nuevas.
5. Espera y pasa a la siguiente ronda.

## Configuración
Archivo central:
- `app/pln_bot/nucleo/config.py`

Variables clave:
- `FDI_PLN__BUTLER_ADDRESS`
- `api_base_url`
- `ollama_url`
- `modelo_default`

## Opciones de CLI
`app/main.py`:
- `--alias`, `-a`
- `--modelo`, `-m`
- `--debug`, `-d`
- `--max-rondas`, `-r`
- `--pausa`, `-p`
- `--api-url`

`app/test_runner.py`:
- `-n`
- `--prefijo`
- `--modelo`, `-m`
- `--debug / --no-debug`, `-d`
- `--max-rondas`, `-r`
- `--pausa`, `-p`
- `--consola`
