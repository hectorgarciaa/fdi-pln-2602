# Bot Negociador Autónomo (PLN)

Bot para negociar recursos contra el servidor `fdi-pln-butler`.

El proyecto está organizado por paquetes en español y tiene dos modos de ejecución:
- `app/principal.py`: un bot.
- `app/orquestador_bots.py`: varios bots en paralelo.

## Requisitos
- Python 3.12+
- Ollama en ejecución (`http://127.0.0.1:11434` por defecto)
- Servidor Butler en ejecución

## Instalación
```bash
uv sync
```

Descarga un modelo si no lo tienes:
```bash
ollama pull qwen3:8b
```

## Ejecución rápida
Un bot (menú interactivo):
```bash
uv run app/principal.py
```

Un bot (modo automático):
```bash
uv run app/principal.py --alias MiBot --modelo qwen3:8b --debug
```

Varios bots:
```bash
uv run app/orquestador_bots.py -n 3 --consola
```

Compatibilidad:
- `app/main.py` sigue funcionando como wrapper de `app/principal.py`.
- `app/test_runner.py` sigue funcionando como wrapper de `app/orquestador_bots.py`.

## Estructura actual
```text
app/
├── principal.py                  # Entry point en español (1 bot)
├── orquestador_bots.py           # Entry point en español (N bots)
├── main.py                       # Wrapper legado
├── test_runner.py                # Wrapper legado
└── pln_bot/
    ├── agente/
    │   ├── negociador.py
    │   └── ejecutor_ronda.py
    ├── interfaz/
    │   ├── principal.py
    │   └── orquestador_bots.py
    ├── nucleo/
    │   └── configuracion.py
    ├── negociacion/
    │   ├── utilidades_mensajes.py
    │   ├── politica_negociacion.py
    │   ├── constructor_propuestas.py
    │   ├── enviador_propuestas.py
    │   ├── gestor_acuerdos.py
    │   └── procesador_buzon.py
    └── servicios/
        ├── cliente_api.py
        ├── cliente_ollama.py
        └── servicio_analisis.py
```

## Flujo del bot
En cada ronda:
1. Actualiza estado (`/info`, `/gente`).
2. Procesa cartas del buzón.
3. Acepta/rechaza/contraoferta según reglas.
4. Envía nuevas propuestas.
5. Espera y pasa a la siguiente ronda.

Cuando cumple objetivo de recursos:
1. Cambia a modo de maximización de oro.
2. Vende excedentes.
3. Finaliza al no tener más excedentes.

## Configuración
La configuración central está en:
- `app/pln_bot/nucleo/configuracion.py`

Variables y parámetros relevantes:
- `FDI_PLN__BUTLER_ADDRESS`: URL base de Butler.
- `api_base_url`, `ollama_url`.
- `modelo_default`, `modelos_disponibles`.
- `think_timeout`, `disable_think`.

## CLI principal (`app/principal.py`)
Opciones:
- `--alias`, `-a`: nombre del bot; si se indica, ejecuta directo sin menú.
- `--modelo`, `-m`: modelo de IA.
- `--debug`, `-d`: muestra decisiones y logs en consola.
- `--max-rondas`, `-r`: número máximo de rondas.
- `--pausa`, `-p`: pausa entre rondas (segundos).
- `--api-url`: override explícito de URL de API.

## Orquestador (`app/orquestador_bots.py`)
Opciones:
- `-n`: número de bots.
- `--prefijo`: prefijo de alias (`Bot_1`, `Bot_2`, ...).
- `-m`, `--modelo`: modelo para todos los bots.
- `-d`, `--debug/--no-debug`: debug para todos.
- `-r`, `--max-rondas`: rondas máximas.
- `-p`, `--pausa`: pausa entre rondas.
- `--consola`: salida coloreada en terminal (si no, logs a archivo).

## Dependencias principales
- `requests`
- `ollama`
- `click`
- `rich`
- `loguru`
- `pydantic`
- `pydantic-ai`
