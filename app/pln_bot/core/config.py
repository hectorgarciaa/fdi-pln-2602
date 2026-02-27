"""
Configuración centralizada del proyecto.

Usa pydantic BaseModel para validar tipos y valores por defecto.
Carga automáticamente un archivo .env en la raíz del proyecto
(si existe) mediante python-dotenv.

Toda la configuración se resuelve en un único punto:
  1. Variables de entorno (máxima prioridad)
  2. Archivo .env (cómodo para desarrollo local)
  3. Valores por defecto definidos aquí (fallback)
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple

# ── Carga de .env (raíz del proyecto) ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env", override=False)


# ── Helpers de resolución ────────────────────────────────────────────────


def _env(var: str, default: str) -> str:
    """Lee una variable de entorno con fallback, limpiando whitespace."""
    return os.getenv(var, default).strip().rstrip("/") or default


def _ensure_http(url: str) -> str:
    """Añade 'http://' si la URL no tiene esquema."""
    if url.startswith(("http://", "https://")):
        return url
    return f"http://{url}"


def modelo_soporta_tools(modelo: str) -> bool:
    """Valida si el modelo soporta el flujo de tools de negociación."""
    return modelo.strip().lower().startswith("qwen")


class OllamaParams(BaseModel):
    """Parámetros de generación para Ollama."""

    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    repeat_penalty: float = 1.2
    num_predict: int = 512
    num_ctx: int = 2048
    num_gpu: int = 1
    num_thread: int = 8
    stop: List[str] = ["---"]


class Settings(BaseModel):
    """Configuración global del proyecto — todos los valores son validados."""

    # ── URLs (resolución: env → .env → default) ─────────────────────────
    api_base_url: str = Field(
        default_factory=lambda: _ensure_http(_env("FDI_PLN__BUTLER_ADDRESS", "127.0.0.1:7719"))
    )
    ollama_url: str = Field(
        default_factory=lambda: _env("FDI_PLN__OLLAMA_URL", "http://127.0.0.1:11434")
    )

    # ── Modelos de IA ────────────────────────────────────────────────────
    modelos_disponibles: Dict[str, Tuple[str, str]] = {
        "1": ("qwen3:8b", "QWEN (tools activas)"),
    }
    modelo_default: str = Field(
        default_factory=lambda: _env("FDI_PLN__MODELO", "qwen3:8b")
    )

    # ── Parámetros de Ollama ─────────────────────────────────────────────
    ollama_params: OllamaParams = OllamaParams()

    # ── Estrategia de análisis IA (pydantic_ai) ──────────────────────────
    max_analisis_llm_por_ronda: int = Field(
        default=12,
        ge=1,
        description="Máximo de análisis pydantic_ai por ronda.",
    )
    forzar_llm_en_ofertas_estructuradas: bool = Field(
        default=True,
        description="Si es True, también analiza con LLM cartas con plantilla.",
    )

    # ── Control de pensamiento (modelo qwen3) ───────────────────────────
    think_timeout: int = Field(default=25, description="Máx. segundos de <think>")
    disable_think: bool = Field(default=False, description="Forzar /no_think")

    # ── Valores por defecto del agente ───────────────────────────────────
    max_rondas: int = Field(default=10, ge=1)
    pausa_entre_rondas: int = Field(default=30, ge=0)
    pausa_entre_acciones: int = Field(default=1, ge=0)
    max_propuestas_por_ronda: int = Field(default=3, ge=1)

    # ── TTLs y backoff ───────────────────────────────────────────────────
    rechazo_ttl_rondas: int = Field(default=2, ge=0)
    acuerdo_ttl_segundos: int = Field(default=300, ge=0)
    acuerdo_gracia_ttl_segundos: int = Field(default=240, ge=0)
    tx_cerrado_ttl_segundos: int = Field(default=1200, ge=0)
    backoff_escala_rondas: List[int] = [1, 2, 4, 6]
    backoff_retencion_rondas: int = Field(default=20, ge=0)

    # ── Orquestador (test_runner) ────────────────────────────────────────
    num_bots_default: int = Field(default=3, ge=1)
    prefijo_bots: str = "Bot"

    # ── Recursos conocidos en el juego ───────────────────────────────────
    recursos_conocidos: List[str] = [
        "oro",
        "madera",
        "piedra",
        "comida",
        "hierro",
        "trigo",
        "carbon",
        "agua",
        "plata",
        "cobre",
        "diamante",
        "lana",
        "tela",
        "cuero",
        "cristal",
        "acero",
        "ladrillos",
        "arroz",
        "queso",
        "pan",
        "leche",
        "carne",
        "pescado",
        "fruta",
        "verdura",
        "sal",
        "azucar",
        "miel",
        "vino",
        "cerveza",
    ]


# ── Instancia global (importar esta) ────────────────────────────────────
settings = Settings()

# ── Alias de compatibilidad (acceso directo sin cambiar los imports) ─────
API_BASE_URL = settings.api_base_url
OLLAMA_URL = settings.ollama_url
MODELOS_DISPONIBLES = settings.modelos_disponibles
MODELO_DEFAULT = settings.modelo_default
OLLAMA_PARAMS = settings.ollama_params.model_dump()
MAX_ANALISIS_LLM_POR_RONDA = settings.max_analisis_llm_por_ronda
FORZAR_LLM_EN_OFERTAS_ESTRUCTURADAS = settings.forzar_llm_en_ofertas_estructuradas
THINK_TIMEOUT = settings.think_timeout
DISABLE_THINK = settings.disable_think
MAX_RONDAS = settings.max_rondas
PAUSA_ENTRE_RONDAS = settings.pausa_entre_rondas
PAUSA_ENTRE_ACCIONES = settings.pausa_entre_acciones
MAX_PROPUESTAS_POR_RONDA = settings.max_propuestas_por_ronda
RECHAZO_TTL_RONDAS = settings.rechazo_ttl_rondas
ACUERDO_TTL_SEGUNDOS = settings.acuerdo_ttl_segundos
ACUERDO_GRACIA_TTL_SEGUNDOS = settings.acuerdo_gracia_ttl_segundos
TX_CERRADO_TTL_SEGUNDOS = settings.tx_cerrado_ttl_segundos
BACKOFF_ESCALA_RONDAS = tuple(settings.backoff_escala_rondas)
BACKOFF_RETENCION_RONDAS = settings.backoff_retencion_rondas
NUM_BOTS_DEFAULT = settings.num_bots_default
PREFIJO_BOTS = settings.prefijo_bots
RECURSOS_CONOCIDOS = settings.recursos_conocidos
