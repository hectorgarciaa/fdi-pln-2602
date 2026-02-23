# 🤖 Bot Negociador Autónomo — PLN

Bot autónomo que negocia intercambios de recursos en el servidor de juego **fdi-pln-butler**.  
Usa un modelo de IA local (Ollama) para analizar mensajes, detectar estafas, aceptar o rechazar ofertas y generar propuestas.

---

<<<<<<< HEAD
## Requisitos
- Python 3.12+
- Ollama en ejecución (`http://127.0.0.1:11434` por defecto)
- Servidor Butler en ejecución
=======
## 📋 Índice

- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Inicio rápido](#-inicio-rápido)
- [Uso detallado](#-uso-detallado)
  - [main.py — Bot individual](#mainpy--bot-individual)
  - [test_runner.py — Orquestador multi-bot](#test_runnerpy--orquestador-multi-bot)
- [Arquitectura](#-arquitectura)
- [Configuración](#-configuración)
- [Librerías utilizadas](#-librerías-utilizadas)

---

## 🔧 Requisitos

| Requisito | Versión mínima |
|-----------|---------------|
| Python    | 3.12+         |
| [Ollama](https://ollama.com) | Cualquiera (con al menos un modelo descargado) |
| Servidor fdi-pln-butler | Corriendo en `http://127.0.0.1:7719` |

### Modelos de IA soportados

| Modelo | Velocidad | Notas |
|--------|-----------|-------|
| `llama3.2:3b` | ⚡⚡⚡ Ultra rápido (3-5s) | Ligero |
| `phi3:mini` | ⚡⚡⚡ Muy rápido (3-5s) | Ligero |
| `qwen3-vl:8b` | ⚡⚡ Balance (5-10s) | Buena calidad |
| `qwen2.5:7b` | ⚡ Calidad (10-15s) | Muy preciso |
| `qwen3:8b` | — Solo texto | **Default**, con control de `<think>` |

---

## 📦 Instalación
>>>>>>> parent of f36904a (refactor por paquetes)

```bash
# Clonar o entrar en el directorio del proyecto
cd pln/

# Instalar dependencias con uv
uv sync
```

Asegúrate de tener Ollama corriendo y al menos un modelo descargado:

```bash
ollama pull qwen3:8b
ollama serve   # si no está corriendo ya
```

---

## 🚀 Inicio rápido

```bash
# Desde la raíz del proyecto:

# Modo interactivo (menú con opciones)
uv run app/main.py

# Lanzar un bot directo por línea de comandos
uv run app/main.py --alias MiBot --debug

# Lanzar 3 bots en paralelo
uv run app/test_runner.py -n 3
```

---

## 📖 Uso detallado

### `main.py` — Bot individual

Punto de entrada principal. Tiene dos modos:

#### Modo interactivo (sin flags)

```bash
uv run app/main.py
```

Abre un menú donde puedes:
1. **Iniciar agente autónomo** — configura modelo, debug, rondas y lanza el bot.
2. **Operaciones API manuales** — ver info, jugadores, enviar cartas/paquetes, etc.

#### Modo automático (con `--alias`)

```bash
uv run app/main.py --alias Bot_1 --modelo llama3.2:3b --debug --max-rondas 15 --pausa 20
```

Lanza el bot directamente sin menú interactivo.

#### Flags de `main.py`

| Flag | Corto | Default | Descripción |
|------|-------|---------|-------------|
| `--alias` | `-a` | *(ninguno)* | Nombre del bot. **Si se da, modo automático.** |
| `--modelo` | `-m` | `qwen3:8b` | Modelo de IA a usar. |
| `--debug` | `-d` | `False` | Activa modo debug: muestra cada decisión del agente (📤📥🔍🧠🔄). |
| `--max-rondas` | `-r` | `10` | Número máximo de rondas de negociación. |
| `--pausa` | `-p` | `30` | Segundos de espera entre rondas (para dar tiempo a respuestas). |
| `--api-url` | — | *(ninguno)* | Override explícito de la URL base de la API (si no se indica, se usa `FDI_PLN__BUTLER_ADDRESS`). |
| `--help` | `-h` | — | Muestra la ayuda. |

Nota para Butler `--monopuesto`: el bot envía automáticamente `agente=<alias_del_bot>` en las llamadas API que lo requieren (`/info`, `/paquete`, `/mail/{uid}`).

#### Ejemplos

```bash
# Bot con modelo rápido y debug
uv run app/main.py -a Bot_1 -m llama3.2:3b -d

# Tomar URL del butler desde variable de entorno
export FDI_PLN__BUTLER_ADDRESS=http://127.0.0.1:7719
uv run app/main.py -a Bot_1

# Bot silencioso, 20 rondas, pausa corta
uv run app/main.py -a Negociador -r 20 -p 10

# Bot conectado a otro servidor
uv run app/main.py -a Bot_1 --api-url http://192.168.1.100:7719
```

---

### `test_runner.py` — Orquestador multi-bot

Lanza N bots en paralelo, cada uno como un subproceso independiente de `main.py`.
Cada bot se identifica por su alias mediante el parámetro `agente`.

```bash
uv run app/test_runner.py -n 5 --consola
```

#### Flags de `test_runner.py`

| Flag | Corto | Default | Descripción |
|------|-------|---------|-------------|
| `-n` | — | `3` | Número de bots a lanzar. |
| `--prefijo` | — | `Bot` | Prefijo para los nombres (`Bot_1`, `Bot_2`, …). |
| `--modelo` | `-m` | `qwen3:8b` | Modelo de IA para todos los bots. |
| `--debug / --no-debug` | `-d` | `True` | Activar/desactivar debug en los bots. |
| `--max-rondas` | `-r` | `10` | Rondas máximas por bot. |
| `--pausa` | `-p` | `30` | Pausa entre rondas (segundos). |
| `--consola` | — | `False` | Mostrar salida coloreada en terminal. Sin este flag, la salida va a archivos `logs/`. |
| `--help` | `-h` | — | Muestra la ayuda. |

#### Modos de salida

- **Modo logs (default):** cada bot escribe en `app/logs/Bot_1_20250101_120000.log`.
- **Modo consola (`--consola`):** salida coloreada en terminal, cada bot con un color distinto.

#### Cierre limpio

Pulsa `Ctrl+C` para detener todos los bots. El orquestador envía SIGTERM a cada proceso y muestra un resumen final.

---

## 🏗️ Arquitectura

```
app/
├── config.py          # Configuración centralizada (pydantic)
├── api_client.py      # Cliente HTTP para la API del juego
├── ollama_client.py   # Cliente para Ollama (IA local)
├── negociador.py      # Lógica del agente negociador
├── main.py            # Punto de entrada (CLI con click)
├── test_runner.py     # Orquestador multi-bot
└── logs/              # Logs de ejecución (modo archivos)
```

### Flujo de datos

```
config.py
    ↓
api_client.py ←→ Servidor fdi-pln-butler (HTTP)
ollama_client.py ←→ Ollama (IA local)
    ↓
negociador.py  ← Usa ambos clientes
    ↓
main.py  ← Punto de entrada (1 bot)
test_runner.py  ← Lanza N bots (subprocesos de main.py)
```

### Loop del agente

Cada ronda el agente:

1. **Actualiza su estado** — consulta `/info` y `/gente`.
2. **Revisa el buzón** — analiza cada carta con IA:
   - ¿Es estafa? → Bloquear + lista negra.
   - ¿Es aceptación? → Ejecutar acuerdo pendiente (enviar paquete).
   - Análisis completo → Aceptar / Contraofertar / Rechazar.
3. **Envía propuestas** — contacta hasta 3 jugadores con ofertas estructuradas (`[OFREZCO]`/`[PIDO]`).
4. **Espera** — pausa configurable para dar tiempo a respuestas.
5. **¿Objetivo completado?** → Cambia a modo *Maximizar Oro* o finaliza.

---

## ⚙️ Configuración

La configuración se gestiona en `app/config.py` con modelos **pydantic**.
La URL del butler se obtiene de `FDI_PLN__BUTLER_ADDRESS` y, si no existe, cae en `http://127.0.0.1:7719`.
El flag `--api-url` tiene prioridad sobre ambos valores.
Con Butler en modo `--monopuesto`, el cliente usa también el alias del bot como identificador `agente`.

Valores principales:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `api_base_url` | `FDI_PLN__BUTLER_ADDRESS` o `http://127.0.0.1:7719` | URL del servidor del juego |
| `ollama_url` | `http://127.0.0.1:11434` | URL de Ollama |
| `modelo_default` | `qwen3:8b` | Modelo de IA por defecto |
| `think_timeout` | `25` | Máx. segundos de bloque `<think>` antes de cortar |
| `disable_think` | `False` | Forzar `/no_think` en modelos qwen3 |

### Parámetros de Ollama

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `temperature` | `0.3` | Baja creatividad, respuestas más deterministas |
| `num_predict` | `100` | Máx. tokens de respuesta |
| `num_ctx` | `2048` | Ventana de contexto |
| `num_gpu` | `1` | Capas en GPU |
| `num_thread` | `8` | Hilos de CPU |

---

## 📚 Librerías utilizadas

| Librería | Uso |
|----------|-----|
| [requests](https://docs.python-requests.org/) | Cliente HTTP para la API del juego |
| [ollama](https://github.com/ollama/ollama-python) | Cliente oficial de Ollama (IA local) |
| [click](https://click.palletsprojects.com/) | CLI moderna (reemplaza argparse) |
| [rich](https://rich.readthedocs.io/) | Tablas, paneles, prompts y colores en terminal |
| [loguru](https://loguru.readthedocs.io/) | Logging estructurado (reemplaza print + logging) |
| [pydantic](https://docs.pydantic.dev/) | Validación de configuración y respuestas de IA |
