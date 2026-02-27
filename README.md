# Bot Negociador Autónomo (PLN) — Práctica 1

Agente autónomo que negocia e intercambia recursos con otros jugadores en el servidor **fdi-pln-butler**, combinando **IA clásica** (regex y heurísticas de texto) con **IA moderna** (LLM local con salida estructurada vía `pydantic-ai`).

El bot opera en un bucle de rondas: lee su buzón, clasifica cada carta (aceptación, rechazo, propuesta o ruido), decide la mejor respuesta y envía propuestas nuevas a otros jugadores hasta cumplir su objetivo de recursos o maximizar su oro.

> **Asignatura:** Procesamiento del Lenguaje Natural — Grado en Ingeniería Informática, UCM  
> **Curso:** 2025-2026

## Equipo — Grupo 02 (2602)

| Integrante | Rol |
|------------|-----|
| **Héctor García Rincón** | Desarrollo del agente negociador |
| **Pablo Manuel Rodriguez Sosa** | Desarrollo del agente negociador |

---

## Índice

1. [Requisitos](#requisitos)  
2. [Instalación](#instalación)  
3. [Variables de entorno](#variables-de-entorno)  
4. [Ejecución](#ejecución)  
5. [Arquitectura](#arquitectura)  
6. [Flujo de una ronda](#flujo-de-una-ronda)  
7. [Estrategia de negociación](#estrategia-de-negociación)  
8. [Pipeline de procesamiento de mensajes](#pipeline-de-procesamiento-de-mensajes)  
9. [Sistema de propuestas y acuerdos](#sistema-de-propuestas-y-acuerdos)  
10. [Configuración avanzada](#configuración-avanzada)  
11. [Opciones CLI](#opciones-cli)  
12. [Logging y persistencia](#logging-y-persistencia)  
13. [Tecnologías utilizadas](#tecnologías-utilizadas)  

---

## Requisitos

| Requisito | Versión mínima |
|-----------|---------------|
| Python | 3.12+ |
| [uv](https://docs.astral.sh/uv/) | cualquiera |
| [Ollama](https://ollama.com/) | en ejecución local (`http://127.0.0.1:11434`) |
| Servidor Butler | en ejecución (`http://127.0.0.1:7719` por defecto) |

## Instalación

```bash
# 1. Instalar dependencias del proyecto
uv sync

# 2. Descargar el modelo de lenguaje recomendado
ollama pull qwen3:8b
```

### Dependencias principales

| Paquete | Uso |
|---------|-----|
| `click` | CLI moderna con opciones y flags |
| `rich` | Interfaz de terminal enriquecida (paneles, tablas, colores) |
| `loguru` | Logging estructurado a consola y fichero |
| `requests` | Cliente HTTP para la API del juego |
| `pydantic` | Validación de datos y modelos tipados |
| `pydantic-ai` | Salida estructurada del LLM (JSON tipado con reintentos) |
| `ollama` | Cliente del servidor Ollama |
| `python-dotenv` | Carga automática de variables desde `.env` |

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `FDI_PLN__BUTLER_ADDRESS` | URL del servidor Butler | `http://127.0.0.1:7719` |
| `FDI_PLN__OLLAMA_URL` | URL del servidor Ollama | `http://127.0.0.1:11434` |
| `FDI_PLN__MODELO` | Modelo de IA por defecto | `qwen3:8b` |

Se pueden definir en un archivo `.env` en la raíz del proyecto (cargado automáticamente vía `python-dotenv`) o como variables de entorno del sistema. Las variables de entorno tienen prioridad sobre el `.env`.

## Ejecución

### Bot individual (modo directo)

```bash
uv run fdi-pln-2602-p1 --alias MiBot --debug
```

### Menú interactivo

```bash
uv run fdi-pln-2602-p1
```

El menú permite:
- **Iniciar agente autónomo** — Seleccionar modelo, activar debug, configurar rondas y pausa.
- **Operaciones API manuales** — Ver información, jugadores, crear/eliminar alias, enviar cartas y paquetes directamente.

### Varios bots en paralelo (orquestador)

```bash
uv run app/test_runner.py -n 3 --consola     # salida coloreada en terminal
uv run app/test_runner.py -n 5 -d            # 5 bots, logs a fichero
```

El orquestador (`test_runner.py`) lanza N subprocesos independientes, cada uno ejecutando `main.py` con un alias único (`Bot_1`, `Bot_2`, …). Soporta cierre limpio con `Ctrl+C`.

---

## Arquitectura

```text
app/
├── main.py                              # CLI principal — 1 bot (click + rich)
├── test_runner.py                       # Orquestador — N bots en paralelo
└── pln_bot/
    ├── agente/
    │   ├── negociador.py                # AgenteNegociador: estado, loop, acciones
    │   └── ronda.py                     # Lógica de una ronda completa
    ├── core/
    │   └── config.py                    # Settings (pydantic) + env + defaults
    ├── negociacion/
    │   ├── procesador_buzon.py          # Pipeline de análisis del buzón
    │   ├── utilidades_mensajes.py       # Regex: clasificación y extracción
    │   ├── constructor_propuestas.py    # Generación de propuestas y contraofertas
    │   ├── enviador_propuestas.py       # Envío de propuestas a jugadores
    │   └── gestor_acuerdos.py           # Acuerdos pendientes, TTL y aceptaciones
    └── services/
        ├── api_client.py                # Cliente HTTP (requests + reintentos)
        └── analysis.py                  # AnalisisMensajesService (pydantic-ai)
```

### Responsabilidades clave

| Módulo | Responsabilidad |
|--------|----------------|
| `negociador.py` | Máquina de estados del agente (`CONSEGUIR_OBJETIVO` → `MAXIMIZAR_ORO` → `COMPLETADO`), persistencia JSON, tracking de acuerdos/rechazos/backoff |
| `ronda.py` | Orquesta una ronda: limpia TTL, actualiza estado, procesa buzón, envía propuestas, detecta paquetes recibidos |
| `procesador_buzon.py` | Pipeline de filtros en cascada para cada carta (sistema → aceptación → rechazo → corto → estructurada → LLM) |
| `utilidades_mensajes.py` | 14 patrones regex de rechazo, 8 de aceptación, extracción de `[tx:id]`, parsing de ofertas `"N recurso"` |
| `constructor_propuestas.py` | Genera propuestas con rotación round-robin, respetando rechazos/backoff/comprometidos; contraofertas y propuestas adaptadas |
| `gestor_acuerdos.py` | Registra acuerdos pendientes, resuelve aceptaciones por `tx_id` o heurística FIFO, valida inventario antes de enviar |
| `analysis.py` | Wrapper de `pydantic-ai`: un `Agent` con system prompt inyecta contexto y devuelve `RespuestaUnificada` (decisión + extracción en 1 llamada) |
| `api_client.py` | Centraliza las llamadas HTTP al servidor Butler con timeout, reintentos y logging |
| `config.py` | `Settings(BaseModel)` con validación Pydantic; resuelve env → `.env` → defaults |

---

## Flujo de una ronda

```
┌─────────────────────────────────────────────────────────┐
│                    INICIO DE RONDA                      │
├─────────────────────────────────────────────────────────┤
│ 1. Limpiar acuerdos expirados (TTL dinámico)            │
│ 2. Purgar backoff y propuestas obsoletas                │
│ 3. Actualizar estado vía API (recursos, objetivo, gente)│
│ 4. ¿Objetivo completado? → cambiar a MAXIMIZAR_ORO     │
│ 5. Procesar buzón (pipeline de filtros + LLM)           │
│ 6. Detectar paquetes recibidos (diff de inventario)     │
│ 7. Enviar propuestas a jugadores no contactados         │
│ 8. Guardar estado de negociación a disco                │
│ 9. Esperar pausa antes de siguiente ronda               │
└─────────────────────────────────────────────────────────┘
```

### Máquina de estados del agente

```
CONSEGUIR_OBJETIVO  ──(objetivo cumplido)──►  MAXIMIZAR_ORO  ──(sin excedentes)──►  COMPLETADO
```

- **`CONSEGUIR_OBJETIVO`** — Intercambia excedentes por recursos que le faltan.
- **`MAXIMIZAR_ORO`** — Vende los excedentes restantes pidiendo oro.
- **`COMPLETADO`** — Fin de la negociación.

---

## Estrategia de negociación

### IA clásica (regex) — `utilidades_mensajes.py`

Filtros rápidos que resuelven la mayoría de los mensajes **sin coste de LLM**:

| Filtro | Patrones | Función |
|--------|----------|---------|
| Cartas del sistema | `"sistema"`, `"server"`, `"butler"`, `"has recibido…"` | `es_carta_sistema()` |
| Aceptaciones | 8 patrones: `"acepto el trato"`, `"trato hecho"`, `"te he enviado"`, etc. | `es_aceptacion_simple()` |
| Rechazos | 14 patrones: `"no me interesa"`, `"no me conviene"`, `"paso de"`, etc. | `es_rechazo_simple()` |
| Mensajes cortos | < 15 caracteres sin verbos de propuesta | `es_mensaje_corto_sin_propuesta()` |
| Ofertas estructuradas | `"mi N recurso por tu N recurso"`, `"yo te doy … y tú me das …"` | `extraer_oferta_estructurada()` |
| Extracción de `tx_id` | `[tx:abc123]` o `tx:abc123` | `extraer_tx_id()` |

### IA moderna (LLM) — `analysis.py`

Para mensajes que no se resuelven con regex, se usa `pydantic-ai` con Ollama:

- **Modelo:** `qwen3:8b` (con soporte de tools/structured output)
- **1 sola llamada por mensaje** — Devuelve `RespuestaUnificada` con:
  - `es_aceptacion` (bool)
  - `ofrecen` / `piden` (dict recurso → cantidad)
  - `decision` (`"aceptar"` | `"rechazar"` | `"contraofertar"` | `"ignorar"`)
  - `contraoferta_ofrezco` / `contraoferta_pido`
  - `razon` (explicación)
- **Contexto dinámico** — El prompt incluye: modo del agente, recursos actuales, objetivo, necesidades y excedentes.
- **Budget por ronda** — Máximo 12 llamadas LLM/ronda. Las cartas que exceden el presupuesto se difieren.
- **Fallback** — Si el LLM devuelve salida vacía para una carta estructurada, se aplica la decisión rápida local como respaldo.

---

## Pipeline de procesamiento de mensajes

Cada carta del buzón pasa por una cascada de filtros (de más barato a más caro):

```
Carta recibida
  │
  ├─► ¿Carta del sistema?           → ignorar
  ├─► ¿Aceptación textual (regex)?  → ejecutar acuerdo (enviar paquete)
  ├─► ¿Rechazo textual (regex)?     → registrar rechazo + propuesta adaptada
  ├─► ¿Mensaje corto sin propuesta? → ignorar
  ├─► ¿Oferta en formato plantilla? → decisión rápida (sin LLM) o LLM  ─┐
  └─► Lenguaje natural libre        → análisis LLM (pydantic-ai)        │
                                                                         ▼
                                                     ┌──────────────────────────┐
                                                     │   Decisión del agente:   │
                                                     │  • aceptar → enviar pkg  │
                                                     │  • rechazar + contraof.  │
                                                     │  • contraofertar (LLM)   │
                                                     │  • ignorar               │
                                                     └──────────────────────────┘
```

---

## Sistema de propuestas y acuerdos

### Generación de propuestas

- **Rotación round-robin** — Rota por las combinaciones `(excedente, necesidad)` para diversificar ofertas.
- **Propuestas generosas** — Si el agente tiene >15 unidades de un recurso, ofrece hasta 3 unidades por 1.
- **Ofertas de oro** — Si no hay excedentes de recursos, ofrece oro.
- **Propuestas adaptadas** — Tras un rechazo, analiza los recursos mencionados por el otro jugador y genera una nueva propuesta con lo que le interesa.

### Identificación de transacciones

Cada propuesta incluye un **`[tx:id]`** (UUID corto de 10 caracteres) que permite:
- Emparejar aceptaciones con la propuesta original.
- Detectar aceptaciones duplicadas.
- Resolver aceptaciones tardías (caché de expirados).

### Gestión de acuerdos pendientes

| Mecanismo | Descripción |
|-----------|-------------|
| **TTL dinámico** | Los acuerdos expiran tras `min(300s, pausa × 2)` para no bloquear recursos indefinidamente |
| **Caché de expirados** | Acuerdos expirados se retienen 240s adicionales para aceptar respuestas tardías |
| **TX cerrados** | Registro de transacciones completadas (TTL 1200s) para evitar dobles envíos |
| **Validación pre-envío** | Antes de enviar un paquete, verifica que tiene recursos suficientes y que no rompe el objetivo |

### Backoff adaptativo

Cuando una combinación `(destinatario, recurso_ofrezco, recurso_pido)` es rechazada o expira sin respuesta, se aplica **backoff exponencial**:

| Nivel | Espera (rondas) |
|-------|-----------------|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |

Las entradas de backoff se limpian automáticamente tras 20 rondas o cuando un intercambio con esa combinación se cierra con éxito.

---

## Configuración avanzada

Toda la configuración está centralizada en `app/pln_bot/core/config.py` usando un modelo `Settings(BaseModel)` de Pydantic con validación de tipos.

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `max_rondas` | 10 | Rondas máximas de negociación |
| `pausa_entre_rondas` | 30s | Espera entre rondas para recibir respuestas |
| `pausa_entre_acciones` | 1s | Espera entre envíos de cartas |
| `max_propuestas_por_ronda` | 3 | Límite de propuestas nuevas por ronda |
| `max_analisis_llm_por_ronda` | 12 | Presupuesto de llamadas LLM por ronda |
| `forzar_llm_en_ofertas_estructuradas` | `true` | Si `true`, también analiza con LLM las cartas con formato plantilla |
| `rechazo_ttl_rondas` | 2 | Rondas que dura un rechazo en memoria |
| `acuerdo_ttl_segundos` | 300 | TTL base para acuerdos pendientes |
| `acuerdo_gracia_ttl_segundos` | 240 | Ventana de gracia para aceptaciones tardías |
| `tx_cerrado_ttl_segundos` | 1200 | Retención de transacciones cerradas |
| `backoff_escala_rondas` | `[1, 2, 4, 6]` | Escalones del backoff exponencial |

### Parámetros del modelo (Ollama)

| Parámetro | Valor |
|-----------|-------|
| `temperature` | 0.3 |
| `top_p` | 0.7 |
| `top_k` | 20 |
| `repeat_penalty` | 1.2 |
| `num_predict` | 2048 |
| `num_ctx` | 2048 |

---

## Opciones CLI

### `fdi-pln-2602-p1` / `app/main.py`

| Flag | Corto | Default | Descripción |
|------|-------|---------|-------------|
| `--alias` | `-a` | *(menú)* | Nombre del bot. Si se da, ejecuta en modo automático |
| `--modelo` | `-m` | `qwen3:8b` | Modelo de IA a usar |
| `--debug` | `-d` | `false` | Activa modo debug (muestra cada decisión) |
| `--max-rondas` | `-r` | `10` | Rondas máximas |
| `--pausa` | `-p` | `30` | Segundos entre rondas |
| `--api-url` | — | env/default | URL base de la API |

### `app/test_runner.py`

| Flag | Corto | Default | Descripción |
|------|-------|---------|-------------|
| `-n` | — | `3` | Número de bots a lanzar |
| `--prefijo` | — | `Bot` | Prefijo para nombres (`Bot_1`, `Bot_2`, …) |
| `--modelo` | `-m` | `qwen3:8b` | Modelo de IA |
| `--debug` | `-d` | `true` | Activa debug (por defecto activo) |
| `--max-rondas` | `-r` | `10` | Rondas máximas por bot |
| `--pausa` | `-p` | `30` | Pausa entre rondas |
| `--consola` | — | `false` | Salida coloreada en terminal en vez de logs a fichero |

---

## Logging y persistencia

### Logs (`loguru`)

- **Consola** — Solo en modo `--debug`. Formato con timestamp, nivel e icono por tipo (`📤 ENVIO`, `🔍 ANALISIS`, `🧠 DECISION`, etc.).
- **Fichero** — Siempre activo en `app/logs/{alias}.log`. Rotación a 10 MB, retención 7 días.

### Estado de negociación (`app/state/{alias}.json`)

El agente persiste su estado entre ejecuciones:
- Acuerdos pendientes y expirados
- Transacciones cerradas
- Propuestas enviadas y rechazos recibidos
- Tabla de backoff

Esto permite **retomar la negociación** tras un reinicio sin perder contexto.

---

## Tecnologías utilizadas

| Tecnología | Versión | Propósito |
|------------|---------|-----------|
| Python | ≥ 3.12 | Lenguaje principal |
| [uv](https://docs.astral.sh/uv/) | — | Gestor de dependencias y entornos |
| [Ollama](https://ollama.com/) | — | Servidor de modelos LLM locales |
| [pydantic-ai](https://ai.pydantic.dev/) | ≥ 1.57 | Salida estructurada del LLM |
| [Pydantic](https://docs.pydantic.dev/) | ≥ 2.12 | Validación de datos y configuración |
| [click](https://click.palletsprojects.com/) | ≥ 8.3 | Framework CLI |
| [Rich](https://rich.readthedocs.io/) | ≥ 14.3 | Interfaz de terminal enriquecida |
| [loguru](https://loguru.readthedocs.io/) | ≥ 0.7 | Logging estructurado |
| [Hatchling](https://hatch.pypa.io/) | — | Build backend (PEP 517) |
