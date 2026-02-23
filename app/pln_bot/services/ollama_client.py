"""
Cliente Ollama para consultas de IA (librería oficial).

Usa loguru para logging y rich para indicadores de progreso en terminal.

Control de pensamiento (modelos qwen3):
  - DISABLE_THINK=True  → añade /no_think al prompt.
  - DISABLE_THINK=False → permite pensar pero corta si supera THINK_TIMEOUT
    y reintenta con /no_think.
  - Siempre limpia los bloques <think> de la respuesta final.
"""

import re
import time

import ollama
from loguru import logger
from rich.console import Console
from rich.status import Status

from ..core.config import OLLAMA_URL, OLLAMA_PARAMS, THINK_TIMEOUT, DISABLE_THINK

# ── Rich console (reutilizada en toda la app) ───────────────────────────
console = Console()

# Modelos que soportan bloques <think>
_MODELOS_CON_THINK = ("qwen3",)


class OllamaClient:
    """Cliente para consultas a Ollama con control de pensamiento."""

    def __init__(self, modelo: str):
        self.modelo = modelo
        self.client = ollama.Client(host=OLLAMA_URL)
        self.soporta_think = any(m in modelo.lower() for m in _MODELOS_CON_THINK)

    # ── Utilidades ───────────────────────────────────────────────────────

    @staticmethod
    def _limpiar_think(texto: str) -> str:
        """Elimina bloques <think>...</think> (completos o cortados)."""
        limpio = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
        limpio = re.sub(r"<think>.*", "", limpio, flags=re.DOTALL)
        return limpio.strip()

    def _preparar_prompt(self, prompt: str) -> str:
        """Añade /no_think si la config lo indica."""
        if DISABLE_THINK and self.soporta_think:
            return f"/no_think\n{prompt}"
        return prompt

    # ── API pública ──────────────────────────────────────────────────────

    def consultar(
        self, prompt: str, timeout: int = 60, mostrar_progreso: bool = True
    ) -> str:
        """
        Envía un prompt a Ollama y devuelve la respuesta limpia.

        Si el modelo se queda "pensando" más de THINK_TIMEOUT segundos,
        corta y reintenta con /no_think.
        """
        prompt = self._preparar_prompt(prompt)

        try:
            if mostrar_progreso:
                return self._consultar_streaming(prompt, timeout)
            else:
                return self._consultar_simple(prompt, timeout)
        except ollama.ResponseError as e:
            logger.error("Error del modelo: {}", e.error)
            return f"Error del modelo: {e.error}"
        except ollama.RequestError as e:
            logger.error("Error de conexión con Ollama: {}", e)
            return f"Error de conexión con Ollama: {e}"
        except Exception as e:
            logger.error("Error inesperado: {}", e)
            return f"Error: {str(e)}"

    # ── Modos de consulta ────────────────────────────────────────────────

    def _consultar_streaming(self, prompt: str, timeout: int) -> str:
        """Consulta con streaming, progreso con rich y control de <think>."""
        respuesta_completa = ""
        dentro_de_think = False
        think_inicio = None
        pensamiento_cortado = False
        inicio_global = time.time()

        stream = self.client.generate(
            model=self.modelo,
            prompt=prompt,
            stream=True,
            options=OLLAMA_PARAMS,
        )

        for chunk in stream:
            texto = chunk.get("response", "")
            respuesta_completa += texto

            # ── Timeout global ──
            if time.time() - inicio_global > timeout:
                console.print("\n[yellow]⏱ Timeout general alcanzado[/]")
                break

            # ── Detección de <think> ──
            if "<think>" in texto.lower() and not dentro_de_think:
                dentro_de_think = True
                think_inicio = time.time()
                console.print("[dim]🧠 (pensando…)[/] ", end="")

            if "</think>" in texto.lower() and dentro_de_think:
                dentro_de_think = False
                think_inicio = None
                console.print("[green]✓[/] ", end="")

            # ── Timeout de pensamiento ──
            if dentro_de_think and think_inicio:
                tiempo_pensando = time.time() - think_inicio
                if tiempo_pensando > THINK_TIMEOUT:
                    pensamiento_cortado = True
                    console.print(f" [red]✂ cortado ({tiempo_pensando:.0f}s)[/]")
                    break

            # Solo mostrar tokens fuera de <think>
            if not dentro_de_think:
                texto_visible = self._limpiar_think(texto)
                if texto_visible:
                    console.print(texto_visible, end="", highlight=False)

            if chunk.get("done"):
                break

        console.print()  # nueva línea

        respuesta_limpia = self._limpiar_think(respuesta_completa)

        if pensamiento_cortado and not respuesta_limpia:
            if self.soporta_think:
                console.print("  [cyan]🔄 Reintentando sin pensamiento…[/]")
                return self._consultar_sin_think(prompt, timeout)
            return "Error: El modelo se quedó pensando sin generar respuesta"

        return respuesta_limpia

    def _consultar_simple(self, prompt: str, timeout: int) -> str:
        """Consulta sin rich pero con streaming interno para controlar <think>."""
        respuesta_completa = ""
        dentro_de_think = False
        think_inicio = None
        pensamiento_cortado = False
        inicio_global = time.time()

        stream = self.client.generate(
            model=self.modelo,
            prompt=prompt,
            stream=True,
            options=OLLAMA_PARAMS,
        )

        for chunk in stream:
            texto = chunk.get("response", "")
            respuesta_completa += texto

            if time.time() - inicio_global > timeout:
                logger.debug("⏱ Timeout general en _consultar_simple ({}s)", timeout)
                break

            if "<think>" in texto.lower() and not dentro_de_think:
                dentro_de_think = True
                think_inicio = time.time()

            if "</think>" in texto.lower() and dentro_de_think:
                dentro_de_think = False
                think_inicio = None

            if dentro_de_think and think_inicio:
                if time.time() - think_inicio > THINK_TIMEOUT:
                    pensamiento_cortado = True
                    logger.debug(
                        "✂ Think cortado en _consultar_simple ({}s)",
                        time.time() - think_inicio,
                    )
                    break

            if chunk.get("done"):
                break

        respuesta_limpia = self._limpiar_think(respuesta_completa)

        if pensamiento_cortado and not respuesta_limpia:
            if self.soporta_think:
                logger.debug("🔄 Reintentando sin think (_consultar_simple)")
                return self._consultar_sin_think(prompt, timeout)
            return "Error: El modelo se quedó pensando sin generar respuesta"

        return respuesta_limpia

    def _consultar_sin_think(self, prompt: str, timeout: int) -> str:
        """Reintento forzando /no_think."""
        if not prompt.strip().startswith("/no_think"):
            prompt = f"/no_think\n{prompt}"
        try:
            response = self.client.generate(
                model=self.modelo,
                prompt=prompt,
                stream=False,
                options=OLLAMA_PARAMS,
            )
            return self._limpiar_think(response.get("response", ""))
        except Exception as e:
            logger.error("Error (reintento sin think): {}", e)
            return f"Error (reintento sin think): {str(e)}"

    # ── Health check ─────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Verifica si Ollama está disponible."""
        try:
            self.client.list()
            return True
        except Exception:
            return False
