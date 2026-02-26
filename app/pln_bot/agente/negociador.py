"""
Agente Negociador Autónomo.

Ejecuta un loop de rondas hasta completar el objetivo del juego:
  1. Actualizar estado  → ¿Qué necesito? ¿Qué me sobra?
  2. Revisar buzón      → Analizar ofertas (IA), aceptar / rechazar
  3. Enviar propuestas  → Contactar jugadores con intercambios
  4. Esperar            → Dar tiempo a respuestas
  5. ¿Objetivo?         → Cambiar a modo MAXIMIZAR ORO o fin

Usa loguru para logging, rich para la interfaz de terminal y
pydantic para validar las respuestas JSON de la IA.
"""

import json
import os
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import MODELO_DEFAULT, modelo_soporta_tools
from .ronda import ejecutar_ronda
from ..services.analysis import AnalisisMensajesService, RespuestaUnificada
from ..services.api_client import APIClient
from ..services.ollama_client import OllamaClient
from ..negociacion.constructor_propuestas import generar_propuesta

# ── Rich console compartida ─────────────────────────────────────────────
console = Console()


# =========================================================================
# ENUMS
# =========================================================================


class ModoAgente(Enum):
    """Estados del agente."""

    CONSEGUIR_OBJETIVO = "conseguir_objetivo"
    MAXIMIZAR_ORO = "maximizar_oro"
    COMPLETADO = "completado"


# =========================================================================
# AGENTE
# =========================================================================


class AgenteNegociador:
    """
    Agente autónomo que negocia para conseguir recursos.

    Uso:
        agente = AgenteNegociador("MiAlias", debug=True)
        agente.ejecutar()
    """

    def __init__(
        self,
        alias: str,
        modelo: str = MODELO_DEFAULT,
        debug: bool = False,
        api_url: str = None,
    ):
        if not modelo_soporta_tools(modelo):
            raise ValueError(
                f"Modelo no soportado para tools: '{modelo}'. Usa un modelo qwen*."
            )

        self.alias = alias
        self.api = APIClient(base_url=api_url, agente=alias)
        self.ia = OllamaClient(modelo)
        self.analisis_mensajes = AnalisisMensajesService(modelo)
        self.debug = debug

        # Estado
        self.modo = ModoAgente.CONSEGUIR_OBJETIVO
        self.info_actual: Optional[Dict] = None
        self.gente: List[str] = []

        # Tracking
        self.contactados_esta_ronda: List[str] = []
        self.acuerdos_pendientes: Dict[str, List[Dict]] = {}
        # Acuerdos que salieron de pendientes por TTL, retenidos por tx
        # para aceptar respuestas tardías sin perder trazabilidad.
        self.acuerdos_expirados_tx: Dict[str, Dict[str, Any]] = {}
        # Índice por remitente para aceptar respuestas tardías sin tx.
        self.acuerdos_expirados_por_remitente: Dict[str, List[Dict[str, Any]]] = {}
        # tx ya ejecutados para evitar dobles envíos por aceptaciones repetidas.
        self.tx_cerrados: Dict[str, float] = {}
        self.intercambios_realizados: List[Dict] = []
        self.cartas_vistas: set = set()

        # Rotación de propuestas
        self.ronda_actual: int = 0
        self.propuesta_index: int = 0

        # Memoria de propuestas y rechazos
        # clave = (destinatario, recurso_ofrezco, recurso_pido), valor = ronda
        self.propuestas_enviadas: Dict[tuple, int] = {}
        # clave = (destinatario, recurso_ofrezco, recurso_pido), valor = ronda
        # Expiran tras RECHAZO_TTL rondas para reintentar con nuevas condiciones
        self.rechazos_recibidos: Dict[tuple, int] = {}
        self.RECHAZO_TTL: int = 2  # rondas antes de reintentar un combo rechazado
        # Ventanas de tiempo para gestión robusta de acuerdos.
        self.ACUERDO_TTL_SEGUNDOS: int = 300
        self.ACUERDO_GRACIA_TTL_SEGUNDOS: int = 240
        self.TX_CERRADO_TTL_SEGUNDOS: int = 1200

        # Snapshot de recursos para detectar paquetes recibidos
        self.recursos_ronda_anterior: Dict[str, int] = {}

        # Configuración
        self.pausa_entre_acciones = 1
        self.pausa_entre_rondas = 30
        self.max_rondas = 10
        self.max_propuestas_por_ronda = 3

        # ── Configurar loguru ────────────────────────────────────────────
        # Limpiar handlers previos (evitar duplicados en multi-bot)
        logger.remove()

        # Consola: solo si debug
        if debug:
            logger.add(
                sys.stderr,
                level="DEBUG",
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>",
            )

        # Fichero: SIEMPRE (app/logs/{alias}.log)
        app_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self._estado_runtime_path = os.path.join(app_dir, "state", f"{alias}.json")
        os.makedirs(os.path.dirname(self._estado_runtime_path), exist_ok=True)
        log_dir = os.path.join(app_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{alias}.log")
        logger.add(
            log_path,
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "{name}:{function}:{line} - {message}",
        )
        logger.info(f"Logs guardados en {log_path}")
        self._cargar_estado_negociacion()

    # =====================================================================
    # LOGGING
    # =====================================================================

    def _log(self, tipo: str, mensaje: str, detalles: Dict = None):
        """Registra una acción con loguru."""
        icono = {
            "ENVIO": "📤",
            "RECEPCION": "📥",
            "ANALISIS": "🔍",
            "DECISION": "🧠",
            "INTERCAMBIO": "🔄",
            "ALERTA": "⚠️",
            "EXITO": "✅",
            "ERROR": "❌",
            "INFO": "ℹ️",
        }.get(tipo, "•")

        extra = f" | {detalles}" if detalles else ""
        log_msg = f"{icono} [{tipo}] {mensaje}{extra}"

        # Mapear tipo → nivel loguru
        nivel = {
            "ERROR": "error",
            "ALERTA": "warning",
            "EXITO": "success",
        }.get(tipo, "debug")

        logger.opt(depth=1).log(nivel.upper(), log_msg)

        # Siempre mostrar en consola si debug está activo
        if self.debug:
            console.print(f"  {log_msg}")

    # =====================================================================
    # CONSULTAS DE ESTADO
    # =====================================================================

    def _actualizar_estado(self) -> Dict:
        """Obtiene y procesa el estado actual."""
        self.info_actual = self.api.get_info()
        self.gente = self.api.get_gente()

        if not self.info_actual:
            return {}

        recursos = self.info_actual.get("Recursos", {})
        objetivo = self.info_actual.get("Objetivo", {})

        necesidades = {}
        for rec, cant_obj in objetivo.items():
            actual = recursos.get(rec, 0)
            if actual < cant_obj:
                necesidades[rec] = cant_obj - actual

        excedentes = {}
        for rec, actual in recursos.items():
            if rec == "oro":
                continue
            obj = objetivo.get(rec, 0)
            if actual > obj:
                excedentes[rec] = actual - obj

        return {
            "recursos": recursos,
            "oro": recursos.get("oro", 0),
            "objetivo": objetivo,
            "necesidades": necesidades,
            "excedentes": excedentes,
            "objetivo_completado": len(necesidades) == 0,
        }

    def _obtener_jugadores_disponibles(self) -> List[str]:
        """Devuelve jugadores que podemos contactar."""
        alias_propios_raw = (
            self.info_actual.get("Alias", []) if self.info_actual else []
        )
        if isinstance(alias_propios_raw, str):
            alias_propios = [alias_propios_raw]
        else:
            alias_propios = alias_propios_raw

        # Asegurar que cada elemento es un string (protección extra)
        gente_str = [str(p) if not isinstance(p, str) else p for p in self.gente]

        disponibles = [
            p for p in gente_str if p != self.alias and p not in alias_propios
        ]

        if not disponibles:
            self._log(
                "INFO",
                "No hay jugadores disponibles",
                {"gente": self.gente, "mis_alias": alias_propios},
            )

        return disponibles

    # =====================================================================
    # PERSISTENCIA DE ESTADO DE NEGOCIACIÓN
    # =====================================================================

    def _cargar_estado_negociacion(self):
        """Carga acuerdos/tx para retomar negociación tras reinicio."""
        if not os.path.exists(self._estado_runtime_path):
            return
        try:
            with open(self._estado_runtime_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return

            acuerdos_pendientes = data.get("acuerdos_pendientes", {})
            acuerdos_expirados_tx = data.get("acuerdos_expirados_tx", {})
            acuerdos_expirados_por_remitente = data.get(
                "acuerdos_expirados_por_remitente", {}
            )
            tx_cerrados = data.get("tx_cerrados", {})
            propuestas_enviadas = data.get("propuestas_enviadas", {})
            rechazos_recibidos = data.get("rechazos_recibidos", {})

            if isinstance(acuerdos_pendientes, dict):
                self.acuerdos_pendientes = acuerdos_pendientes
            if isinstance(acuerdos_expirados_tx, dict):
                self.acuerdos_expirados_tx = acuerdos_expirados_tx
            if isinstance(acuerdos_expirados_por_remitente, dict):
                self.acuerdos_expirados_por_remitente = acuerdos_expirados_por_remitente

            if isinstance(tx_cerrados, dict):
                self.tx_cerrados = {
                    str(tx): float(ts)
                    for tx, ts in tx_cerrados.items()
                    if isinstance(tx, str)
                }

            if isinstance(propuestas_enviadas, dict):
                propuestas: Dict[tuple, int] = {}
                for key, ronda in propuestas_enviadas.items():
                    if not isinstance(key, str):
                        continue
                    partes = key.split("|")
                    if len(partes) != 3:
                        continue
                    try:
                        propuestas[(partes[0], partes[1], partes[2])] = int(ronda)
                    except (TypeError, ValueError):
                        continue
                self.propuestas_enviadas = propuestas

            if isinstance(rechazos_recibidos, dict):
                rechazos: Dict[tuple, int] = {}
                for key, ronda in rechazos_recibidos.items():
                    if not isinstance(key, str):
                        continue
                    partes = key.split("|")
                    if len(partes) != 3:
                        continue
                    try:
                        rechazos[(partes[0], partes[1], partes[2])] = int(ronda)
                    except (TypeError, ValueError):
                        continue
                self.rechazos_recibidos = rechazos

            self._log(
                "INFO",
                "Estado de negociación cargado",
                {
                    "acuerdos_pendientes": sum(
                        len(v) for v in self.acuerdos_pendientes.values()
                    ),
                    "tx_cerrados": len(self.tx_cerrados),
                },
            )
        except Exception as e:
            self._log("ERROR", f"No se pudo cargar estado de negociación: {e}")

    def _guardar_estado_negociacion(self):
        """Guarda acuerdos/tx para continuidad entre ejecuciones."""
        try:
            propuestas_serializadas = {
                "|".join(clave): ronda
                for clave, ronda in self.propuestas_enviadas.items()
                if isinstance(clave, tuple) and len(clave) == 3
            }
            rechazos_serializados = {
                "|".join(clave): ronda
                for clave, ronda in self.rechazos_recibidos.items()
                if isinstance(clave, tuple) and len(clave) == 3
            }

            data = {
                "acuerdos_pendientes": self.acuerdos_pendientes,
                "acuerdos_expirados_tx": self.acuerdos_expirados_tx,
                "acuerdos_expirados_por_remitente": self.acuerdos_expirados_por_remitente,
                "tx_cerrados": self.tx_cerrados,
                "propuestas_enviadas": propuestas_serializadas,
                "rechazos_recibidos": rechazos_serializados,
                "updated_at": time.time(),
            }
            temp_path = f"{self._estado_runtime_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=True, indent=2)
            os.replace(temp_path, self._estado_runtime_path)
        except Exception as e:
            self._log("ERROR", f"No se pudo guardar estado de negociación: {e}")

    # =====================================================================
    # ANÁLISIS DE MENSAJES (IA + pydantic)
    # =====================================================================

    def _analizar_mensaje(
        self,
        remitente: str,
        mensaje: str,
        asunto: str = "",
        necesidades: Optional[Dict[str, int]] = None,
        excedentes: Optional[Dict[str, int]] = None,
        recursos_actuales: Optional[Dict[str, int]] = None,
        objetivo: Optional[Dict[str, int]] = None,
    ) -> RespuestaUnificada:
        """Analiza un mensaje con UNA sola llamada IA (aceptación + extracción).

        Devuelve RespuestaUnificada con parsing + decisión sugerida por LLM.
        """
        try:
            r = self.analisis_mensajes.analizar(
                remitente=remitente,
                mensaje=mensaje,
                asunto=asunto,
                necesidades=necesidades,
                excedentes=excedentes,
                recursos_actuales=recursos_actuales,
                objetivo=objetivo,
                modo_agente=self.modo.value,
            )
            self._log("DEBUG", f"IA unificada: {r.model_dump()}")
            return r
        except Exception as e:
            self._log("ERROR", f"Error pydantic_ai: {e}")
            return RespuestaUnificada(
                decision="ignorar", razon="No se pudo analizar el mensaje"
            )

    # =====================================================================
    # GENERACIÓN DE PROPUESTAS
    # =====================================================================

    def _recursos_comprometidos(self) -> Dict[str, int]:
        """Calcula recursos ya prometidos en acuerdos pendientes.

        Suma todos los recursos_dar de todos los acuerdos pendientes
        para saber cuánto tenemos realmente disponible.
        """
        comprometidos: Dict[str, int] = {}
        for acuerdos in self.acuerdos_pendientes.values():
            for acuerdo in acuerdos:
                for rec, cant in acuerdo.get("recursos_dar", {}).items():
                    comprometidos[rec] = comprometidos.get(rec, 0) + cant
        return comprometidos

    def _excedentes_disponibles(self, excedentes: Dict[str, int]) -> Dict[str, int]:
        """Excedentes reales descontando recursos ya comprometidos."""
        comprometidos = self._recursos_comprometidos()
        disponibles = {}
        for rec, cant in excedentes.items():
            libre = cant - comprometidos.get(rec, 0)
            if libre > 0:
                disponibles[rec] = libre
        return disponibles

    # =====================================================================
    # MEMORIA DE RECHAZOS
    # =====================================================================

    def _rechazo_vigente(self, clave: tuple) -> bool:
        """Comprueba si un rechazo sigue vigente (no ha expirado)."""
        if clave not in self.rechazos_recibidos:
            return False
        ronda_rechazo = self.rechazos_recibidos[clave]
        return (self.ronda_actual - ronda_rechazo) < self.RECHAZO_TTL

    def _generar_texto_propuesta_ia(
        self, destinatario: str, necesidades: Dict, excedentes: Dict, oro: int
    ) -> Optional[Dict]:
        """Usa IA para redactar la propuesta (la lógica es programática)."""
        propuesta = generar_propuesta(self, destinatario, necesidades, excedentes, oro)
        if not propuesta:
            return None

        ofrezco_str = ", ".join(f"{c} {r}" for r, c in propuesta["_ofrezco"].items())
        pido_str = ", ".join(f"{c} {r}" for r, c in propuesta["_pido"].items())

        prompt = (
            f"Genera un mensaje corto, amigable y natural para proponer un intercambio "
            f"de recursos en un juego.\n\n"
            f"DESTINATARIO: {destinatario}\nYO SOY: {self.alias}\n"
            f"OFREZCO: {ofrezco_str}\nPIDO: {pido_str}\n\n"
            f"El mensaje debe dejar MUY CLARO qué ofreces y qué pides, "
            f"con las cantidades exactas. Escríbelo como un humano, sin etiquetas "
            f"ni formatos especiales.\n"
            f"IMPORTANTE: Termina SIEMPRE el mensaje con esta frase exacta:\n"
            f"\"Si aceptas, responde 'acepto el trato'. "
            f"Si no te conviene, responde 'no me conviene'.\"\n"
            f"Escribe SOLO el mensaje, nada más."
        )

        texto = self.ia.consultar(prompt, timeout=30, mostrar_progreso=False)
        if texto and not texto.startswith("Error"):
            tx_id = propuesta.get("_tx_id")
            tx_tag = f" [tx:{tx_id}]" if tx_id else ""
            propuesta["cuerpo"] = texto
            propuesta["asunto"] = (
                f"Intercambio:{tx_tag} mi {ofrezco_str} por tu {pido_str}"
            )
        return propuesta

    # =====================================================================
    # ACCIONES
    # =====================================================================

    def _enviar_carta(self, destinatario: str, asunto: str, cuerpo: str) -> bool:
        """Envía una carta de negociación."""
        exito = self.api.enviar_carta(self.alias, destinatario, asunto, cuerpo)
        self._log(
            "ENVIO",
            f"Carta a {destinatario}",
            {
                "asunto": asunto,
                "cuerpo": cuerpo[:100] + "…" if len(cuerpo) > 100 else cuerpo,
                "exito": exito,
            },
        )
        return exito

    def _enviar_paquete(self, destinatario: str, recursos: Dict[str, int]) -> bool:
        """Envía un paquete de recursos."""
        mis_recursos = self.info_actual.get("Recursos", {}) if self.info_actual else {}
        for rec, cant in recursos.items():
            if mis_recursos.get(rec, 0) < cant:
                self._log(
                    "ERROR",
                    f"❌ No tenemos suficiente {rec} para enviar",
                    {"necesario": cant, "disponible": mis_recursos.get(rec, 0)},
                )
                return False

        exito = self.api.enviar_paquete(destinatario, recursos)
        recursos_str = ", ".join(f"{cant} {rec}" for rec, cant in recursos.items())

        if exito:
            self._log("EXITO", f"📤 ENVIAMOS a {destinatario}: {recursos_str}")
        else:
            self._log(
                "ERROR",
                f"❌ Fallo al enviar paquete a {destinatario}",
                {"recursos": recursos_str},
            )

        if exito:
            self.intercambios_realizados.append(
                {
                    "tipo": "enviado",
                    "destinatario": destinatario,
                    "recursos": recursos,
                    "timestamp": time.time(),
                }
            )
        return exito

    def _procesar_paquetes_recibidos(self):
        """Loggea los paquetes recibidos durante esta ronda.

        Los paquetes aparecen automáticamente en Recursos cuando llegan,
        pero loggeamos el cambio para visibilidad.
        """
        if not self.info_actual:
            return

        recursos_actuales = self.info_actual.get("Recursos", {})

        # Comparar con el snapshot de la ronda anterior
        if self.recursos_ronda_anterior:
            incrementos = {}
            for recurso, cantidad in recursos_actuales.items():
                cantidad_anterior = self.recursos_ronda_anterior.get(recurso, 0)
                incremento = cantidad - cantidad_anterior
                if incremento > 0:
                    incrementos[recurso] = incremento

            if incrementos:
                items_str = ", ".join(
                    f"{cant} {rec}" for rec, cant in incrementos.items()
                )
                self._log("EXITO", f"📥 HEMOS RECIBIDO: {items_str}")

        # Actualizar snapshot para la próxima ronda
        self.recursos_ronda_anterior = recursos_actuales.copy()

    # =====================================================================
    # LOOP PRINCIPAL
    # =====================================================================

    def _ejecutar_ronda(self) -> bool:
        """Ejecuta una ronda completa de negociación."""
        return ejecutar_ronda(self, console)

    def ejecutar(self, max_rondas: int = None):
        """Ejecuta el agente hasta completar el objetivo."""
        max_rondas = max_rondas or self.max_rondas

        console.print(
            Panel.fit(
                f"[bold]🤖 AGENTE NEGOCIADOR AUTÓNOMO[/bold]\n\n"
                f"  Alias:      [cyan]{self.alias}[/]\n"
                f"  Modelo:     [cyan]{self.ia.modelo}[/]\n"
                f"  Debug:      [{'green' if self.debug else 'dim'}]"
                f"{'ACTIVADO' if self.debug else 'desactivado'}[/]\n"
                f"  Max rondas: [cyan]{max_rondas}[/]",
                border_style="bright_blue",
            )
        )

        if not self.api.crear_alias(self.alias):
            console.print(
                f"[yellow]⚠ No se pudo crear el alias '{self.alias}', "
                f"puede que ya exista.[/]"
            )

        for ronda in range(1, max_rondas + 1):
            console.print(f"\n[bold cyan]🔄 RONDA {ronda}/{max_rondas}[/]")

            completado = self._ejecutar_ronda()
            self._guardar_estado_negociacion()
            if completado:
                break

            if ronda < max_rondas:
                console.print(
                    f"\n[dim]⏳ Esperando {self.pausa_entre_rondas}s para respuestas…[/]"
                )
                time.sleep(self.pausa_entre_rondas)

        self._guardar_estado_negociacion()
        self._mostrar_resumen()

    # =====================================================================
    # VISUALIZACIÓN (rich)
    # =====================================================================

    def _mostrar_resumen(self):
        """Muestra resumen de la ejecución con rich."""
        estado = self._actualizar_estado()

        # ── Tabla de resumen ─────────────────────────────────────────────
        table = Table(
            title="📊 Resumen de Ejecución",
            border_style="bright_blue",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("Campo", style="bold")
        table.add_column("Valor")

        table.add_row("💰 Oro final", str(estado.get("oro", 0)))

        obj_ok = estado.get("objetivo_completado", False)
        table.add_row(
            "🎯 Objetivo",
            "[green]✅ COMPLETADO[/]" if obj_ok else "[red]❌ PENDIENTE[/]",
        )

        if estado.get("necesidades"):
            table.add_row("📋 Aún falta", str(estado["necesidades"]))

        table.add_row("🔄 Intercambios", str(len(self.intercambios_realizados)))

        console.print()
        console.print(table)

        # ── Detalle de intercambios ──────────────────────────────────────
        if self.intercambios_realizados:
            console.print("\n[bold]Intercambios realizados:[/]")
            for i in self.intercambios_realizados:
                console.print(f"   → {i['destinatario']}: {i['recursos']}")

    def ver_log(self, ultimos: int = 20):
        """Muestra las últimas entradas del log (compatibilidad)."""
        console.print(f"\n[bold]📜 LOG (últimas {ultimos} entradas):[/]")
        console.rule()
        # En el nuevo sistema, loguru ya se encarga. Mostramos mensaje.
        console.print("[dim]El log completo se encuentra en los handlers de loguru.[/]")
        console.print("[dim]Si debug está activo, todo se muestra en consola.[/]")
