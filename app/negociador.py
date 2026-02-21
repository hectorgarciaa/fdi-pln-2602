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
import re as _re
import sys
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from config import RECURSOS_CONOCIDOS, MODELO_DEFAULT, OLLAMA_URL
from api_client import APIClient
from ollama_client import OllamaClient

# ── Rich console compartida ─────────────────────────────────────────────
console = Console()


# =========================================================================
# MODELOS PYDANTIC — respuestas de la IA
# =========================================================================

class RespuestaEstafa(BaseModel):
    """Respuesta de la IA al analizar posibles estafas."""
    es_estafa: bool = False
    razon: str = ""


class RespuestaAceptacion(BaseModel):
    """Respuesta de la IA al detectar aceptaciones."""
    es_aceptacion: bool = False
    razon: str = ""


class RespuestaAnalisis(BaseModel):
    """Respuesta de la IA al analizar un mensaje de negociación.

    Usa validadores para tolerar respuestas malformadas de la IA
    (p.ej. contraoferta_pedir como lista en vez de dict).
    """
    ofrecen: Dict[str, int] = Field(default_factory=dict)
    piden: Dict[str, int] = Field(default_factory=dict)
    aceptar: bool = False
    razon: str = ""
    contraoferta: bool = False
    contraoferta_dar: Dict[str, int] = Field(default_factory=dict)
    contraoferta_pedir: Dict[str, int] = Field(default_factory=dict)

    from pydantic import field_validator

    @field_validator("ofrecen", "piden", "contraoferta_dar", "contraoferta_pedir", mode="before")
    @classmethod
    def _coerce_to_dict(cls, v: Any) -> Dict[str, int]:
        """Convierte valores no-dict a dict vacío en vez de fallar."""
        if isinstance(v, dict):
            # Asegurar que las claves son str y valores int
            return {str(k): int(val) for k, val in v.items() if val is not None}
        return {}


class RespuestaUnificada(BaseModel):
    """Respuesta única de la IA: aceptación + estafa + extracción en 1 llamada."""
    es_aceptacion: bool = False
    es_estafa: bool = False
    ofrecen: Dict[str, int] = Field(default_factory=dict)
    piden: Dict[str, int] = Field(default_factory=dict)
    razon: str = ""

    from pydantic import field_validator

    @field_validator("ofrecen", "piden", mode="before")
    @classmethod
    def _coerce_to_dict(cls, v: Any) -> Dict[str, int]:
        if isinstance(v, dict):
            return {str(k): int(val) for k, val in v.items() if val is not None}
        return {}


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

    def __init__(self, alias: str, modelo: str = MODELO_DEFAULT, debug: bool = False,
                 api_url: str = None):
        self.alias = alias
        self.api = APIClient(base_url=api_url, agente=alias)
        self.ia = OllamaClient(modelo)
        self.debug = debug

        # ── Agentes pydantic_ai (structured output) ────────────────────
        _ollama_base = OLLAMA_URL.rstrip("/") + "/v1"
        _ai_provider = OllamaProvider(base_url=_ollama_base)
        _ai_model = OpenAIChatModel(modelo, provider=_ai_provider)
        _ai_settings = ModelSettings(
            temperature=0.3,
            top_p=0.7,
            max_tokens=512,
        )

        self._agente = Agent(
            _ai_model,
            output_type=RespuestaUnificada,
            system_prompt=(
                "Eres un analizador de mensajes en un juego de intercambio de recursos.\n"
                "Analiza el mensaje y responde con TODOS estos campos:\n\n"
                "1) es_aceptacion: ¿El mensaje ACEPTA un trato previo?\n"
                "   - SÍ si contiene frases como: 'acepto el trato', 'trato hecho',\n"
                "     'te he enviado', 'cerramos el trato', 'acepto tu propuesta',\n"
                "     'de acuerdo', 'perfecto, te envío'.\n"
                "   - NO si es: rechazo, propuesta nueva, 'si aceptas dime',\n"
                "     'no me conviene', 'por ahora no', 'no me interesa'.\n\n"
                "2) es_estafa: ¿Es un intento de estafa?\n"
                "   - Señales: pedir enviar primero sin garantía, promesas imposibles,\n"
                "     urgencia/presión, cosas gratis, bugs del sistema, confianza ciega.\n"
                "   - Un intercambio legítimo propone dar X a cambio de Y.\n\n"
                "3) ofrecen / piden: EXTRAER recursos del mensaje.\n"
                '   - "ofrecen" = lo que el remitente ofrece DAR (lo que yo recibiría).\n'
                '   - "piden" = lo que el remitente quiere RECIBIR (lo que yo daría).\n'
                "   - Solo recursos y cantidades EXPLÍCITOS. NO inventes.\n"
                "   - Rechazo, saludo o no-propuesta → ofrecen={}, piden={}.\n\n"
                "4) razon: explicación breve de tu análisis.\n\n"
                'Ejemplo: "yo te doy 2 madera y tú me das 3 piedra"\n'
                '→ es_aceptacion=false, es_estafa=false, '
                'ofrecen={"madera": 2}, piden={"piedra": 3}'
            ),
            model_settings=_ai_settings,
            retries=2,
        )

        # Estado
        self.modo = ModoAgente.CONSEGUIR_OBJETIVO
        self.info_actual: Optional[Dict] = None
        self.gente: List[str] = []

        # Seguridad y tracking
        self.lista_negra: List[str] = []
        self.contactados_esta_ronda: List[str] = []
        self.acuerdos_pendientes: Dict[str, List[Dict]] = {}
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
        
        # Snapshot de recursos para detectar paquetes recibidos
        self.recursos_ronda_anterior: Dict[str, int] = {}

        # Configuración
        self.pausa_entre_acciones = 1
        self.pausa_entre_rondas = 30
        self.max_rondas = 10

        # ── Configurar loguru ────────────────────────────────────────────
        # Limpiar handlers previos (evitar duplicados en multi-bot)
        logger.remove()

        # Consola: solo si debug
        if debug:
            logger.add(sys.stderr, level="DEBUG",
                       format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                              "<level>{level: <8}</level> | "
                              "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                              "<level>{message}</level>")

        # Fichero: SIEMPRE (logs/{alias}.log)
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
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

    # =====================================================================
    # LOGGING
    # =====================================================================

    def _log(self, tipo: str, mensaje: str, detalles: Dict = None):
        """Registra una acción con loguru."""
        icono = {
            "ENVIO": "📤", "RECEPCION": "📥", "ANALISIS": "🔍",
            "DECISION": "🧠", "INTERCAMBIO": "🔄", "ALERTA": "⚠️",
            "EXITO": "✅", "ERROR": "❌", "INFO": "ℹ️",
        }.get(tipo, "•")

        extra = f" | {detalles}" if detalles else ""
        log_msg = f"{icono} [{tipo}] {mensaje}{extra}"

        # Mapear tipo → nivel loguru
        nivel = {
            "ERROR": "error", "ALERTA": "warning", "EXITO": "success",
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
        alias_propios_raw = self.info_actual.get("Alias", []) if self.info_actual else []
        if isinstance(alias_propios_raw, str):
            alias_propios = [alias_propios_raw]
        else:
            alias_propios = alias_propios_raw

        # Asegurar que cada elemento es un string (protección extra)
        gente_str = [str(p) if not isinstance(p, str) else p for p in self.gente]

        disponibles = [
            p for p in gente_str
            if p != self.alias
            and p not in alias_propios
            and p not in self.lista_negra
        ]

        if not disponibles:
            self._log("INFO", "No hay jugadores disponibles",
                      {"gente": self.gente, "mis_alias": alias_propios})

        return disponibles

    # =====================================================================
    # ANÁLISIS DE MENSAJES (IA + pydantic)
    # =====================================================================

    def _parsear_json_ia(self, respuesta: str) -> Optional[dict]:
        """Extrae el primer objeto JSON de una respuesta de texto."""
        inicio = respuesta.find("{")
        fin = respuesta.rfind("}") + 1
        if inicio != -1 and fin > inicio:
            try:
                return json.loads(respuesta[inicio:fin])
            except json.JSONDecodeError:
                pass
        return None

    def _decidir_aceptar_programatico(self, ofrecen: Dict[str, int],
                                       piden: Dict[str, int],
                                       necesidades: Dict, excedentes: Dict) -> tuple:
        """Decide si aceptar con lógica determinista. Devuelve (bool, str)."""
        if not ofrecen or not piden:
            return False, "oferta incompleta (falta ofrecen o piden)"

        me_ofrecen_lo_que_necesito = any(r in necesidades for r in ofrecen)
        # Oro siempre es útil (moneda universal), aunque no esté en necesidades
        me_ofrecen_oro = "oro" in ofrecen and ofrecen["oro"] > 0
        piden_solo_excedentes = all(
            r in excedentes and excedentes[r] >= c
            for r, c in piden.items() if c > 0
        )

        if me_ofrecen_lo_que_necesito and piden_solo_excedentes:
            return True, "ofrecen lo que necesito y piden lo que me sobra"
        # Aceptar oro a cambio de excedentes (oro siempre tiene valor)
        if me_ofrecen_oro and piden_solo_excedentes:
            return True, "ofrecen oro a cambio de lo que me sobra"
        if not me_ofrecen_lo_que_necesito and not me_ofrecen_oro:
            return False, "no ofrecen nada de lo que necesito"
        return False, "piden recursos que no me sobran o no tengo suficientes"

    def _analizar_mensaje(self, remitente: str, mensaje: str,
                          necesidades: Dict, excedentes: Dict) -> RespuestaUnificada:
        """Analiza un mensaje con UNA sola llamada IA (aceptación + estafa + extracción).

        Devuelve RespuestaUnificada con todos los campos.
        La decisión de aceptar se toma programáticamente después.
        """
        try:
            result = self._agente.run_sync(
                f'Mensaje de "{remitente}":\n{mensaje}'
            )
            r = result.output
            self._log("DEBUG", f"IA unificada: {r.model_dump()}")

            # Si es estafa, añadir a lista negra
            if r.es_estafa and remitente not in self.lista_negra:
                self.lista_negra.append(remitente)
                self._log("ALERTA", f"IA detecta posible estafa de {remitente}",
                          {"razon": r.razon})

            return r
        except Exception as e:
            self._log("ERROR", f"Error pydantic_ai: {e}")
            return RespuestaUnificada(razon="No se pudo analizar el mensaje")

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

    def _rechazo_vigente(self, clave: tuple) -> bool:
        """Comprueba si un rechazo sigue vigente (no ha expirado)."""
        if clave not in self.rechazos_recibidos:
            return False
        ronda_rechazo = self.rechazos_recibidos[clave]
        return (self.ronda_actual - ronda_rechazo) < self.RECHAZO_TTL

    def _generar_propuesta(self, destinatario: str, necesidades: Dict,
                           excedentes: Dict, oro: int) -> Optional[Dict[str, str]]:
        """Genera una propuesta evitando combinaciones ya enviadas o rechazadas."""
        ofrezco: Dict[str, int] = {}
        pido: Dict[str, int] = {}

        # Usar excedentes reales (descontando recursos comprometidos)
        exc_disp = self._excedentes_disponibles(excedentes)

        if exc_disp and necesidades:
            lista_necesidades = list(necesidades.keys())
            lista_excedentes = list(exc_disp.keys())
            total_combos = len(lista_necesidades) * len(lista_excedentes)

            # Probar todas las combinaciones de (excedente, necesidad)
            # empezando desde propuesta_index para rotar
            for offset in range(total_combos):
                idx = (self.propuesta_index + offset) % total_combos
                idx_ofrezco = idx // len(lista_necesidades)
                idx_pido = idx % len(lista_necesidades)

                recurso_ofrezco = lista_excedentes[idx_ofrezco]
                recurso_pido = lista_necesidades[idx_pido]
                clave = (destinatario, recurso_ofrezco, recurso_pido)

                # Saltar si ya fue rechazada (y no ha expirado) o ya enviada esta ronda
                if self._rechazo_vigente(clave):
                    continue
                if clave in self.propuestas_enviadas \
                        and self.ronda_actual - self.propuestas_enviadas[clave] < 2:
                    continue

                # Si tenemos >15 excedentes de este recurso, ofrecer más (hasta 3)
                # para hacer el trato más atractivo
                cantidad_ofrezco = 1
                mis_recursos_totales = self.info_actual.get("Recursos", {}) if self.info_actual else {}
                cantidad_total_recurso = mis_recursos_totales.get(recurso_ofrezco, 0)
                
                # Verificar que realmente tenemos el recurso disponible
                if cantidad_total_recurso <= 0 or exc_disp[recurso_ofrezco] <= 0:
                    self._log("DEBUG", f"Saltando propuesta: no tenemos {recurso_ofrezco} disponible")
                    continue
                
                if cantidad_total_recurso > 15:
                    # Ofrecer 2-3 unidades por 1 (más generoso cuando sobra mucho)
                    cantidad_ofrezco = min(exc_disp[recurso_ofrezco], 3)
                    cantidad_pido = 1
                    self._log("INFO", f"Oferta generosa a {destinatario}: "
                              f"{cantidad_ofrezco} {recurso_ofrezco} por 1 {recurso_pido} "
                              f"(tenemos {cantidad_total_recurso} {recurso_ofrezco})")
                else:
                    # Trato normal 1:1
                    cantidad_pido = 1
                
                if exc_disp[recurso_ofrezco] < cantidad_ofrezco:
                    continue
                    
                ofrezco = {recurso_ofrezco: cantidad_ofrezco}
                pido = {recurso_pido: cantidad_pido}
                self.propuesta_index = idx + 1
                break
            else:
                # Todas las combinaciones de excedentes agotadas →
                # intentar ofrecer oro como fallback (probar TODAS las necesidades)
                encontrado = False
                if necesidades and oro >= 1:
                    comprometidos = self._recursos_comprometidos()
                    oro_libre = oro - comprometidos.get("oro", 0)
                    for recurso_pido in necesidades:
                        clave = (destinatario, "oro", recurso_pido)
                        if self._rechazo_vigente(clave):
                            continue
                        if oro_libre >= 1:
                            ofrezco = {"oro": 1}
                            pido = {recurso_pido: 1}
                            encontrado = True
                            break
                if not encontrado:
                    comprometidos = self._recursos_comprometidos()
                    self._log("INFO", f"Sin combinaciones nuevas para {destinatario}",
                              {"rechazos_vigentes": len(self.rechazos_recibidos),
                               "comprometidos": comprometidos,
                               "oro_libre": oro - comprometidos.get('oro', 0)})
                    return None

        elif necesidades and oro >= 1:
            # Sin excedentes → solo oro. Probar todas las necesidades.
            comprometidos = self._recursos_comprometidos()
            oro_libre = oro - comprometidos.get("oro", 0)
            encontrado = False
            for recurso_pido in necesidades:
                clave = (destinatario, "oro", recurso_pido)
                if self._rechazo_vigente(clave):
                    continue
                if oro_libre >= 1:
                    ofrezco = {"oro": 1}
                    pido = {recurso_pido: 1}
                    encontrado = True
                    break
            if not encontrado:
                return None
        elif exc_disp:
            recurso_ofrezco = list(exc_disp.keys())[0]
            clave = (destinatario, recurso_ofrezco, "oro")
            if self._rechazo_vigente(clave):
                return None
            ofrezco = {recurso_ofrezco: 1}
            pido = {"oro": 1}
        else:
            return None

        ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
        pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())

        cuerpo = (
            f"Hola {destinatario}, soy {self.alias}. "
            f"Te propongo un intercambio: "
            f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
            f"Si aceptas, responde 'acepto el trato'. "
            f"Si no te conviene, responde 'no me conviene'. "
            f"Saludos, {self.alias}"
        )

        return {
            "asunto": f"Propuesta: mi {ofrezco_str} por tu {pido_str}",
            "cuerpo": cuerpo,
            "_ofrezco": ofrezco,
            "_pido": pido,
        }

    def _generar_contraoferta(self, destinatario: str,
                              ofrecen: Dict[str, int],
                              necesidades: Dict, excedentes: Dict) -> Optional[Dict]:
        """Genera contraoferta cuando la oferta original no nos sirve.

        Verifica recursos disponibles (no comprometidos) antes de prometer.
        Comprueba que no sea una contraoferta ya rechazada.
        """
        # Buscar UN recurso que nos ofrezcan y necesitemos
        pido = {}
        for rec, cant in ofrecen.items():
            if rec in necesidades:
                pido[rec] = 1  # 1:1
                break
        if not pido:
            return None

        # Usar excedentes disponibles (descontando comprometidos)
        exc_disp = self._excedentes_disponibles(excedentes)

        # Ofrecer UN recurso excedente a cambio
        ofrezco = {}
        for rec, cant in exc_disp.items():
            if cant >= 1:
                ofrezco[rec] = 1  # 1:1
                break

        # Si no hay excedentes, intentar pagar con oro
        if not ofrezco:
            recursos = self.info_actual.get("Recursos", {}) if self.info_actual else {}
            oro_total = recursos.get("oro", 0)
            comprometidos = self._recursos_comprometidos()
            oro_libre = oro_total - comprometidos.get("oro", 0)
            if oro_libre >= 1:
                ofrezco = {"oro": 1}
            else:
                return None

        # Comprobar que no sea una contraoferta ya rechazada (vigente)
        for r_o in ofrezco:
            for r_p in pido:
                if self._rechazo_vigente((destinatario, r_o, r_p)):
                    self._log("INFO", f"Contraoferta a {destinatario} ya rechazada: "
                              f"{r_o}→{r_p} — no repetir")
                    return None

        ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
        pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())

        cuerpo = (
            f"Hola {destinatario}, soy {self.alias}. "
            f"Vi tu oferta pero no tengo lo que pides. "
            f"Te hago una contrapropuesta: "
            f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
            f"Si aceptas, responde 'acepto el trato'. "
            f"Si no te conviene, responde 'no me conviene'. "
            f"Saludos, {self.alias}"
        )

        return {
            "asunto": f"Contrapropuesta: mi {ofrezco_str} por tu {pido_str}",
            "cuerpo": cuerpo,
            "_ofrezco": ofrezco,
            "_pido": pido,
        }

    def _generar_texto_propuesta_ia(self, destinatario: str, necesidades: Dict,
                                     excedentes: Dict, oro: int) -> Optional[Dict]:
        """Usa IA para redactar la propuesta (la lógica es programática)."""
        propuesta = self._generar_propuesta(destinatario, necesidades, excedentes, oro)
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
            propuesta["cuerpo"] = texto
            propuesta["asunto"] = f"Intercambio: mi {ofrezco_str} por tu {pido_str}"
        return propuesta

    # =====================================================================
    # ACCIONES
    # =====================================================================

    def _enviar_carta(self, destinatario: str, asunto: str, cuerpo: str) -> bool:
        """Envía una carta de negociación."""
        exito = self.api.enviar_carta(self.alias, destinatario, asunto, cuerpo)
        self._log("ENVIO", f"Carta a {destinatario}", {
            "asunto": asunto,
            "cuerpo": cuerpo[:100] + "…" if len(cuerpo) > 100 else cuerpo,
            "exito": exito,
        })
        return exito

    def _enviar_paquete(self, destinatario: str, recursos: Dict[str, int]) -> bool:
        """Envía un paquete de recursos."""
        mis_recursos = self.info_actual.get("Recursos", {}) if self.info_actual else {}
        for rec, cant in recursos.items():
            if mis_recursos.get(rec, 0) < cant:
                self._log("ERROR", f"❌ No tenemos suficiente {rec} para enviar",
                          {"necesario": cant, "disponible": mis_recursos.get(rec, 0)})
                return False

        exito = self.api.enviar_paquete(destinatario, recursos)
        recursos_str = ", ".join(f"{cant} {rec}" for rec, cant in recursos.items())
        
        if exito:
            self._log("EXITO", f"📤 ENVIAMOS a {destinatario}: {recursos_str}")
        else:
            self._log("ERROR", f"❌ Fallo al enviar paquete a {destinatario}",
                      {"recursos": recursos_str})

        if exito:
            self.intercambios_realizados.append({
                "tipo": "enviado",
                "destinatario": destinatario,
                "recursos": recursos,
                "timestamp": time.time(),
            })
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
                items_str = ", ".join(f"{cant} {rec}" for rec, cant in incrementos.items())
                self._log("EXITO", f"📥 HEMOS RECIBIDO: {items_str}")
        
        # Actualizar snapshot para la próxima ronda
        self.recursos_ronda_anterior = recursos_actuales.copy()

    def _responder_aceptacion(self, remitente: str, mensaje_original: str) -> bool:
        """Responde a una aceptación enviando los recursos acordados.

        Intenta buscar el acuerdo pendiente que mejor coincida con lo que
        el remitente menciona en su mensaje de aceptación.
        Verifica que los recursos sigan disponibles antes de enviar.
        """
        if remitente not in self.acuerdos_pendientes or not self.acuerdos_pendientes[remitente]:
            self._log("INFO", f"Aceptación de {remitente} sin acuerdo pendiente registrado")
            return False

        # Intentar encontrar el acuerdo que coincide con el mensaje
        acuerdo_idx = 0
        msg_lower = mensaje_original.lower()
        acuerdos = self.acuerdos_pendientes[remitente]
        for i, ac in enumerate(acuerdos):
            # Si el mensaje menciona algún recurso del acuerdo, preferirlo
            for rec in ac.get("recursos_dar", {}):
                if rec in msg_lower:
                    acuerdo_idx = i
                    break
            for rec in ac.get("recursos_pedir", {}):
                if rec in msg_lower:
                    acuerdo_idx = i
                    break

        acuerdo = acuerdos.pop(acuerdo_idx)
        if not acuerdos:
            del self.acuerdos_pendientes[remitente]

        recursos_a_enviar = acuerdo.get("recursos_dar", {})
        recursos_a_recibir = acuerdo.get("recursos_pedir", {})
        
        if not recursos_a_enviar:
            self._log("INFO", f"Acuerdo con {remitente} sin recursos a enviar")
            return False

        # Verificar disponibilidad real ANTES de enviar
        self._actualizar_estado()
        mis_recursos = self.info_actual.get("Recursos", {}) if self.info_actual else {}
        for rec, cant in recursos_a_enviar.items():
            disponible = mis_recursos.get(rec, 0)
            if disponible < cant:
                self._log("ALERTA",
                          f"No puedo cumplir acuerdo con {remitente}: "
                          f"necesito {cant} {rec} pero solo tengo {disponible}")
                return False

        envio_str = ", ".join(f"{c} {r}" for r, c in recursos_a_enviar.items())
        recibo_str = ", ".join(f"{c} {r}" for r, c in recursos_a_recibir.items())
        
        self._log("DECISION", 
                  f"🤝 Ejecutando acuerdo con {remitente}: "
                  f"doy {envio_str} por {recibo_str}")
        
        if self._enviar_paquete(remitente, recursos_a_enviar):
            return True
        else:
            self._log("ERROR", f"No se pudo enviar paquete a {remitente}")
            return False

    # =====================================================================
    # LOOP PRINCIPAL
    # =====================================================================

    # ── helpers de filtrado rápido (sin IA) ────────────────────────────
    _RE_RECHAZO = _re.compile(
        r'no me interesa|no acepto|no puedo aceptar|rechaz|no,? gracias'
        r'|no tengo lo que|no necesito|no quiero|paso de'
        r'|no me conviene|por ahora no|no es lo que busco'
        r'|no puedo hacer ese|no me sirve|mejor no',
        _re.IGNORECASE,
    )
    _RE_ACEPTACION = _re.compile(
        r'acepto el trato|trato hecho|te he enviado|cerramos el trato'
        r'|acepto tu propuesta|acepto,? dime|de acuerdo.*env[ií]'
        r'|perfecto.*env[ií]|hecho.*te mando',
        _re.IGNORECASE,
    )
    _RE_PROPUESTA = _re.compile(
        r'\b(ofrezco|te doy|te ofrezco|propongo|te propongo|pido|'
        r'necesito|quiero|busco|doy .* por|a cambio de|\d+\s+\w+.*por)',
        _re.IGNORECASE,
    )
    # Patrón para detectar respuestas de rechazo estructuradas
    # (las que nosotros mismos generamos y que otros bots también envían)
    _RE_RESPUESTA_RECHAZO = _re.compile(
        r'^\s*Gracias por la oferta.*no me conviene'
        r'|^\s*Gracias por la oferta.*Saludos'
        r'|^\s*No me interesa.*Saludos'
        r'|^\s*Por ahora no.*Saludos',
        _re.IGNORECASE | _re.DOTALL,
    )

    def _extraer_recursos_mencionados(self, mensaje: str) -> List[str]:
        """Extrae recursos conocidos mencionados en un mensaje de texto."""
        msg_lower = mensaje.lower()
        encontrados = []
        for recurso in RECURSOS_CONOCIDOS:
            # Buscar el recurso como palabra completa
            if _re.search(r'\b' + _re.escape(recurso) + r'\b', msg_lower):
                encontrados.append(recurso)
        return encontrados

    def _generar_propuesta_adaptada(self, destinatario: str,
                                     recursos_que_quiere: List[str],
                                     necesidades: Dict, excedentes: Dict,
                                     oro: int) -> Optional[Dict]:
        """Genera una propuesta adaptada a lo que el otro jugador pidió.

        Cruza lo que el otro quiere con nuestros excedentes,
        y pide a cambio algo que nosotros necesitamos.
        """
        exc_disp = self._excedentes_disponibles(excedentes)

        # Buscar excedentes que coincidan con lo que el otro quiere
        ofrezco: Dict[str, int] = {}
        for rec in recursos_que_quiere:
            if rec in exc_disp and exc_disp[rec] > 0:
                ofrezco[rec] = 1  # 1:1
                break  # un recurso por propuesta

        if not ofrezco:
            return None

        # Pedir algo que necesitemos
        pido: Dict[str, int] = {}
        for rec, cant in necesidades.items():
            clave = (destinatario, list(ofrezco.keys())[0], rec)
            if not self._rechazo_vigente(clave):
                pido[rec] = 1  # 1:1
                break

        if not pido:
            return None

        ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
        pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())

        cuerpo = (
            f"Hola {destinatario}, soy {self.alias}. "
            f"He visto que necesitas {ofrezco_str}. "
            f"Te propongo un intercambio: "
            f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
            f"Si aceptas, responde 'acepto el trato'. "
            f"Si no te conviene, responde 'no me conviene'. "
            f"Saludos, {self.alias}"
        )

        return {
            "asunto": f"Propuesta: mi {ofrezco_str} por tu {pido_str}",
            "cuerpo": cuerpo,
            "_ofrezco": ofrezco,
            "_pido": pido,
        }

    def _es_carta_sistema(self, remitente: str, mensaje: str) -> bool:
        """Devuelve True para notificaciones del sistema (no son propuestas)."""
        if remitente.lower() in ("sistema", "server", "butler"):
            return True
        if _re.match(r'(?i)^(has recibido|recursos generados|paquete|\s*$)', mensaje.strip()):
            return True
        return False

    def _es_rechazo_simple(self, mensaje: str, asunto: str = "") -> bool:
        """Detecta rechazos textuales sin necesidad de IA."""
        # Respuestas de rechazo estructuradas (las que generamos nosotros)
        if self._RE_RESPUESTA_RECHAZO.search(mensaje):
            return True
        # Si el asunto empieza con 'Re:' y el cuerpo tiene texto de rechazo
        # es una respuesta a nuestra propuesta, no una propuesta nueva
        asunto_limpio = asunto.strip().lower()
        if asunto_limpio.startswith("re:") and self._RE_RECHAZO.search(mensaje):
            return True
        # Si contiene alguna propuesta de intercambio, NO es rechazo simple
        if self._RE_PROPUESTA.search(mensaje):
            return False
        return bool(self._RE_RECHAZO.search(mensaje))

    def _es_mensaje_corto_sin_propuesta(self, mensaje: str) -> bool:
        """Detecta mensajes muy cortos que no contienen propuesta."""
        limpio = mensaje.strip()
        if len(limpio) < 15:  # "ok", "gracias", "hola"
            return not self._RE_PROPUESTA.search(limpio)
        return False

    def _es_aceptacion_simple(self, mensaje: str, asunto: str = "") -> bool:
        """Detecta aceptaciones textuales sin necesidad de IA.

        NO activa para propuestas nuevas (asunto 'Propuesta:' o
        'Contrapropuesta:') ni cuando el cuerpo contiene una propuesta
        de intercambio, ya que esos mensajes incluyen la frase
        'acepto el trato' como instrucción, no como aceptación real.
        """
        # Si el asunto indica propuesta nueva → no es aceptación
        asunto_limpio = asunto.strip().lower()
        if asunto_limpio.startswith(("propuesta:", "contrapropuesta:")):
            return False
        # Si el cuerpo contiene una propuesta de intercambio → no es aceptación
        if self._RE_PROPUESTA.search(mensaje):
            return False
        return bool(self._RE_ACEPTACION.search(mensaje))

    # Regex para extraer recursos del asunto de propuestas y contraofertas
    # Captura "mi X recurso por tu Y recurso" del asunto
    _RE_ASUNTO_PROPUESTA = _re.compile(
        r'(?:mi|Propuesta:?\s*mi|Contrapropuesta:?\s*mi)\s+.*?(\w+)\s+por\s+tu\s+.*?(\w+)',
        _re.IGNORECASE,
    )

    # Regex más detallada: extrae TODOS los recursos del asunto
    # Captura múltiples "N recurso" antes y después del "por tu"
    _RE_ASUNTO_RECURSOS = _re.compile(
        r'mi\s+(.+?)\s+por\s+tu\s+(.+?)\s*$',
        _re.IGNORECASE,
    )
    _RE_RECURSO_INDIVIDUAL = _re.compile(
        r'(\d+)\s+(\w+)',
    )

    def _registrar_rechazo(self, remitente: str, asunto: str):
        """Registra un rechazo del remitente extrayendo los recursos del asunto.

        Funciona tanto con propuestas como contraofertas:
        - 'Re: Propuesta: mi 2 madera por tu 3 queso'
        - 'Re: Contrapropuesta: mi 3 piedra, 1 queso por tu 3 trigo'
        """
        # Intentar extraer todos los recursos del asunto
        m = self._RE_ASUNTO_RECURSOS.search(asunto)
        if m:
            parte_ofrezco = m.group(1)
            parte_pido = m.group(2)
            recs_ofrezco = self._RE_RECURSO_INDIVIDUAL.findall(parte_ofrezco)
            recs_pido = self._RE_RECURSO_INDIVIDUAL.findall(parte_pido)
            for _, r_o in recs_ofrezco:
                for _, r_p in recs_pido:
                    clave = (remitente, r_o.lower(), r_p.lower())
                    self.rechazos_recibidos[clave] = self.ronda_actual
            if recs_ofrezco and recs_pido:
                ofr_str = ", ".join(r for _, r in recs_ofrezco)
                pid_str = ", ".join(r for _, r in recs_pido)
                self._log("INFO", f"Rechazo registrado de {remitente}: "
                          f"{ofr_str}→{pid_str} (no repetir)")
                return

        # Fallback: regex simple
        m = self._RE_ASUNTO_PROPUESTA.search(asunto)
        if m:
            recurso_ofrezco = m.group(1).lower()
            recurso_pido = m.group(2).lower()
            clave = (remitente, recurso_ofrezco, recurso_pido)
            self.rechazos_recibidos[clave] = self.ronda_actual
            self._log("INFO", f"Rechazo registrado de {remitente}: "
                      f"{recurso_ofrezco}→{recurso_pido} (no repetir)")

    def _registrar_rechazo_propio(self, remitente: str,
                                   ofrecen: Dict[str, int],
                                   piden: Dict[str, int]):
        """Registra que NOSOTROS rechazamos una oferta de remitente.

        Lo guardamos al revés: si remitente nos ofrece madera y pide queso,
        no tiene sentido que nosotros le propongamos queso por madera.
        """
        for r_o in ofrecen:
            for r_p in piden:
                # Desde la perspectiva del remitente: él ofrece r_o y pide r_p
                # Si nosotros le proponemos r_p→r_o sería redundante
                clave = (remitente, r_p, r_o)
                self.rechazos_recibidos[clave] = self.ronda_actual

    def _procesar_buzon(self, necesidades: Dict, excedentes: Dict) -> int:
        """Procesa todas las cartas del buzón usando IA para lenguaje natural."""
        buzon = self.info_actual.get("Buzon", {}) if self.info_actual else {}
        intercambios = 0
        cartas_procesadas = []

        for uid, carta in buzon.items():
            carta_id = carta.get("id", uid)
            if carta_id in self.cartas_vistas:
                cartas_procesadas.append(uid)
                continue
            self.cartas_vistas.add(carta_id)

            remitente = carta.get("remi", "Desconocido")
            mensaje = carta.get("cuerpo", "")
            asunto = carta.get("asunto", "")

            self._log("RECEPCION", f"Carta de {remitente}",
                      {"asunto": asunto, "mensaje": mensaje[:150]})

            # ── Filtro 0: cartas del Sistema → ignorar ──
            if self._es_carta_sistema(remitente, mensaje):
                self._log("INFO", f"Carta del sistema de '{remitente}' — ignorada")
                cartas_procesadas.append(uid)
                continue

            # ── Lista negra ──
            if remitente in self.lista_negra:
                self._log("ALERTA", f"Ignorando {remitente} (lista negra)")
                cartas_procesadas.append(uid)
                continue

            # ── Filtro 1: rechazos simples → extraer contexto si lo hay ──
            if self._es_rechazo_simple(mensaje, asunto):
                self._registrar_rechazo(remitente, asunto)

                # Intentar extraer qué recursos quiere el otro jugador
                recursos_mencionados = self._extraer_recursos_mencionados(mensaje)
                # Filtrar: solo los que nosotros tenemos de sobra
                exc_disp = self._excedentes_disponibles(excedentes)
                recursos_que_podemos_dar = [
                    r for r in recursos_mencionados
                    if r in exc_disp and exc_disp[r] > 0
                ]

                if recursos_que_podemos_dar and necesidades:
                    self._log("INFO",
                              f"Rechazo de {remitente} — detectados recursos que quiere: "
                              f"{recursos_que_podemos_dar} (tenemos excedentes)")
                    oro_actual = self.info_actual.get("Recursos", {}).get("oro", 0) if self.info_actual else 0
                    propuesta = self._generar_propuesta_adaptada(
                        remitente, recursos_que_podemos_dar,
                        necesidades, excedentes, oro_actual,
                    )
                    if propuesta:
                        self._log("DECISION",
                                  f"PROPUESTA ADAPTADA a {remitente}: "
                                  f"dar={propuesta['_ofrezco']}, pedir={propuesta['_pido']}")
                        if self._enviar_carta(remitente, propuesta["asunto"], propuesta["cuerpo"]):
                            if remitente not in self.acuerdos_pendientes:
                                self.acuerdos_pendientes[remitente] = []
                            self.acuerdos_pendientes[remitente].append({
                                "recursos_dar": propuesta["_ofrezco"],
                                "recursos_pedir": propuesta["_pido"],
                                "timestamp": time.time(),
                            })
                            for r_o in propuesta["_ofrezco"]:
                                for r_p in propuesta["_pido"]:
                                    self.propuestas_enviadas[(remitente, r_o, r_p)] = self.ronda_actual
                    else:
                        self._log("INFO", f"Rechazo de {remitente} — no se pudo generar propuesta adaptada")
                else:
                    self._log("INFO", f"Rechazo de {remitente} — ignorado (sin contexto aprovechable)")

                cartas_procesadas.append(uid)
                continue

            # ── Filtro 2: mensajes muy cortos sin propuesta ──
            if self._es_mensaje_corto_sin_propuesta(mensaje):
                self._log("INFO", f"Mensaje corto de {remitente} sin propuesta — ignorado")
                cartas_procesadas.append(uid)
                continue

            # ── Filtro 3: aceptaciones textuales → sin IA ──
            if self._es_aceptacion_simple(mensaje, asunto):
                self._log("ANALISIS", f"{remitente} ACEPTA intercambio (detectado por texto, sin IA)")
                if self._responder_aceptacion(remitente, mensaje):
                    intercambios += 1
                cartas_procesadas.append(uid)
                continue

            # ── Análisis unificado (1 sola llamada IA) ──
            r = self._analizar_mensaje(remitente, mensaje, necesidades, excedentes)

            # ── ¿Estafa? → ignorar ──
            if r.es_estafa:
                self._log("ALERTA", f"Estafa detectada de {remitente}: {r.razon}")
                cartas_procesadas.append(uid)
                continue

            # ── ¿Aceptación? → responder ──
            if r.es_aceptacion:
                self._log("ANALISIS", f"{remitente} ACEPTA intercambio")
                if self._responder_aceptacion(remitente, mensaje):
                    intercambios += 1
                cartas_procesadas.append(uid)
                continue

            # ── Decisión programática sobre la propuesta ──
            aceptar, razon = self._decidir_aceptar_programatico(
                r.ofrecen, r.piden, necesidades, excedentes)

            self._log("ANALISIS", f"Carta de {remitente} analizada", {
                "ofrecen": r.ofrecen, "piden": r.piden,
                "aceptar": aceptar, "razon": razon,
            })

            # ── Decisión: aceptar ──
            if aceptar and r.piden:
                # VALIDAR antes de enviar: ¿me piden cosas que realmente me sobran?
                self._actualizar_estado()
                mis_recursos = self.info_actual.get("Recursos", {}) if self.info_actual else {}
                # Recalcular excedentes con estado fresco
                estado_fresco = self._actualizar_estado()
                excedentes_frescos = estado_fresco.get("excedentes", {}) if estado_fresco else {}

                envio_valido = True
                recursos_a_enviar: Dict[str, int] = {}
                for rec, cant in r.piden.items():
                    if cant <= 0:
                        continue
                    disponible = mis_recursos.get(rec, 0)
                    excedente_rec = excedentes_frescos.get(rec, 0)
                    if disponible >= cant and excedente_rec >= cant:
                        recursos_a_enviar[rec] = cant
                    else:
                        self._log("ALERTA",
                                  f"No puedo enviar {cant} {rec} a {remitente}: "
                                  f"disponible={disponible}, excedente={excedente_rec}")
                        envio_valido = False

                if envio_valido and recursos_a_enviar:
                    ofrecen_str = ", ".join(f"{c} {r}" for r, c in r.ofrecen.items())
                    enviado_str = ", ".join(f"{c} {r}" for r, c in recursos_a_enviar.items())
                    self._log("EXITO", f"🤝 ACEPTO oferta de {remitente}: doy {enviado_str} por {ofrecen_str}")
                    if self._enviar_paquete(remitente, recursos_a_enviar):
                        self._enviar_carta(
                            remitente, f"Re: {asunto}",
                            f"Acepto el trato. Te he enviado {enviado_str}. "
                            f"Espero recibir {ofrecen_str} de tu parte. "
                            f"Saludos, {self.alias}",
                        )
                        intercambios += 1
                    else:
                        self._log("ERROR", f"No pude enviar paquete a {remitente}")
                elif not recursos_a_enviar:
                    self._log("DECISION",
                              f"Oferta de {remitente} parecía aceptable pero "
                              f"no hay recursos válidos para enviar — rechazada")
                else:
                    self._log("DECISION",
                              f"Oferta de {remitente} rechazada: "
                              f"no tengo suficientes excedentes para enviar {r.piden}")

            elif r.ofrecen or r.piden:
                # ── ¿Contraoferta? Si ofrecen algo que necesito pero piden
                #    lo que no tengo → intentar contraoferta con mis excedentes
                me_ofrecen_util = any(rec in necesidades for rec in r.ofrecen)
                if me_ofrecen_util and razon == "piden recursos que no me sobran o no tengo suficientes":
                    contra = self._generar_contraoferta(
                        remitente, r.ofrecen, necesidades, excedentes)
                    if contra:
                        self._log("DECISION",
                                  f"CONTRAOFERTA a {remitente}: "
                                  f"dar={contra['_ofrezco']}, pedir={contra['_pido']}")
                        self._enviar_carta(
                            remitente, contra["asunto"], contra["cuerpo"])
                        # Registrar acuerdo pendiente
                        if remitente not in self.acuerdos_pendientes:
                            self.acuerdos_pendientes[remitente] = []
                        self.acuerdos_pendientes[remitente].append({
                            "recursos_dar": contra["_ofrezco"],
                            "recursos_pedir": contra["_pido"],
                            "timestamp": time.time(),
                        })
                        cartas_procesadas.append(uid)
                        continue

                self._log("DECISION", f"RECHAZO oferta de {remitente} ({razon})")
                # Registrar el rechazo que NOSOTROS hacemos para no repetir
                self._registrar_rechazo_propio(remitente, r.ofrecen, r.piden)
                self._enviar_carta(
                    remitente, f"Re: {asunto}",
                    f"Gracias por la oferta, {remitente}, pero por ahora "
                    f"no me conviene ese intercambio. Saludos, {self.alias}",
                )
            else:
                self._log("INFO", f"Mensaje de {remitente} sin propuesta clara: {razon}")

            cartas_procesadas.append(uid)

        for uid in cartas_procesadas:
            self.api.eliminar_carta(uid)

        return intercambios

    def _enviar_propuestas(self, necesidades: Dict, excedentes: Dict, oro: int):
        """Envía propuestas a jugadores no contactados (máx 3/ronda)."""
        jugadores = self._obtener_jugadores_disponibles()
        jugadores = [j for j in jugadores if j not in self.contactados_esta_ronda]

        if not jugadores:
            self._log("INFO", "No hay jugadores a quienes enviar propuestas esta ronda")
            return

        for jugador in jugadores:
            propuesta = self._generar_propuesta(jugador, necesidades, excedentes, oro)
            if propuesta is None:
                self._log("INFO", f"No se generó propuesta para {jugador}")
                continue

            if self._enviar_carta(jugador, propuesta["asunto"], propuesta["cuerpo"]):
                self.contactados_esta_ronda.append(jugador)
                acuerdo = {
                    "recursos_dar": propuesta["_ofrezco"],
                    "recursos_pedir": propuesta["_pido"],
                    "timestamp": time.time(),
                }
                if jugador not in self.acuerdos_pendientes:
                    self.acuerdos_pendientes[jugador] = []
                self.acuerdos_pendientes[jugador].append(acuerdo)

                # Registrar en memoria de propuestas
                for r_o in propuesta["_ofrezco"]:
                    for r_p in propuesta["_pido"]:
                        self.propuestas_enviadas[(jugador, r_o, r_p)] = self.ronda_actual

                self._log("INFO", f"Acuerdo pendiente con {jugador}: "
                          f"dar={propuesta['_ofrezco']}, pedir={propuesta['_pido']}")

            time.sleep(self.pausa_entre_acciones)

    def _ejecutar_ronda(self) -> bool:
        """Ejecuta una ronda completa de negociación."""
        console.rule(f"[bold]📍 RONDA — Modo: {self.modo.value}[/bold]")

        self.ronda_actual += 1
        _inicio_ronda = time.time()

        # Limpiar acuerdos viejos (>90s ≈ 2-3 rondas sin respuesta)
        ahora = time.time()
        _ACUERDO_TTL = 90  # segundos
        for persona in list(self.acuerdos_pendientes.keys()):
            viejos = [a for a in self.acuerdos_pendientes[persona]
                      if ahora - a.get("timestamp", 0) >= _ACUERDO_TTL]
            if viejos:
                self._log("INFO", f"Liberando {len(viejos)} acuerdo(s) pendiente(s) "
                          f"con {persona} (sin respuesta en {_ACUERDO_TTL}s)")
            self.acuerdos_pendientes[persona] = [
                a for a in self.acuerdos_pendientes[persona]
                if ahora - a.get("timestamp", 0) < _ACUERDO_TTL
            ]
            if not self.acuerdos_pendientes[persona]:
                del self.acuerdos_pendientes[persona]

        # Limpiar propuestas_enviadas antiguas para poder reintentar
        claves_viejas = [
            k for k, ronda in self.propuestas_enviadas.items()
            if self.ronda_actual - ronda > 2
        ]
        for k in claves_viejas:
            del self.propuestas_enviadas[k]

        # 1. Actualizar estado
        estado = self._actualizar_estado()
        if not estado:
            self._log("ERROR", "No se pudo conectar a la API")
            return False

        necesidades = estado["necesidades"]
        excedentes = estado["excedentes"]
        oro = estado["oro"]
        objetivo_completado = estado["objetivo_completado"]

        # Log de situación actual más visible
        recursos_actuales = self.info_actual.get("Recursos", {}) if self.info_actual else {}
        if recursos_actuales:
            recursos_str = ", ".join(f"{rec}: {cant}" for rec, cant in sorted(recursos_actuales.items()))
            self._log("INFO", f"📦 INVENTARIO: {recursos_str}")
        
        if necesidades:
            nec_str = ", ".join(f"{cant} {rec}" for rec, cant in necesidades.items())
            self._log("INFO", f"🎯 NECESITAMOS: {nec_str}")
        
        if excedentes:
            exc_str = ", ".join(f"{cant} {rec}" for rec, cant in excedentes.items())
            self._log("INFO", f"💰 EXCEDENTES: {exc_str}")
        
        self._log("INFO", f"🪙 ORO: {oro} | Objetivo: {'✅ COMPLETADO' if objetivo_completado else '⏳ En progreso'}")

        # 2. ¿Cambiar de modo?
        if objetivo_completado and self.modo == ModoAgente.CONSEGUIR_OBJETIVO:
            self._log("EXITO", "¡OBJETIVO COMPLETADO! → modo MAXIMIZAR ORO")
            self.modo = ModoAgente.MAXIMIZAR_ORO

        if self.modo == ModoAgente.MAXIMIZAR_ORO and not excedentes:
            self._log("EXITO", "No hay más excedentes para vender")
            self.modo = ModoAgente.COMPLETADO
            return True

        # 3. Procesar buzón
        self._log("INFO", "Procesando buzón…")
        intercambios = self._procesar_buzon(necesidades, excedentes)

        # 3.5. Procesar paquetes recibidos (los paquetes se reflejan automáticamente en Recursos)
        self._procesar_paquetes_recibidos()

        if intercambios > 0:
            self._log("EXITO", f"✅ {intercambios} intercambio(s) completado(s) esta ronda")
            estado = self._actualizar_estado()
            # Mostrar situación actualizada después de intercambios
            recursos_actuales = self.info_actual.get("Recursos", {}) if self.info_actual else {}
            if recursos_actuales:
                recursos_str = ", ".join(f"{rec}: {cant}" for rec, cant in sorted(recursos_actuales.items()))
                self._log("INFO", f"📦 INVENTARIO ACTUALIZADO: {recursos_str}")
            necesidades = estado["necesidades"]
            excedentes = estado["excedentes"]

        # 4. Enviar propuestas
        if necesidades or (self.modo == ModoAgente.MAXIMIZAR_ORO and excedentes):
            self._log("INFO", "Enviando propuestas…")
            self._enviar_propuestas(necesidades, excedentes, estado["oro"])

        # 5. Reset contactados
        self.contactados_esta_ronda = []

        # 6. Si la ronda fue muy rápida (buzón vacío), esperar para dar
        #    tiempo a que otros bots respondan antes de la siguiente ronda
        _duracion_ronda = time.time() - _inicio_ronda
        _espera_minima = self.pausa_entre_rondas * 0.5  # al menos 50% de la pausa
        if _duracion_ronda < _espera_minima:
            _esperar = _espera_minima - _duracion_ronda
            self._log("INFO", f"Ronda rápida ({_duracion_ronda:.1f}s) — "
                      f"esperando {_esperar:.0f}s para recibir respuestas")
            time.sleep(_esperar)

        return estado["objetivo_completado"] and self.modo == ModoAgente.COMPLETADO

    def ejecutar(self, max_rondas: int = None):
        """Ejecuta el agente hasta completar el objetivo."""
        max_rondas = max_rondas or self.max_rondas

        console.print(Panel.fit(
            f"[bold]🤖 AGENTE NEGOCIADOR AUTÓNOMO[/bold]\n\n"
            f"  Alias:      [cyan]{self.alias}[/]\n"
            f"  Modelo:     [cyan]{self.ia.modelo}[/]\n"
            f"  Debug:      [{'green' if self.debug else 'dim'}]"
            f"{'ACTIVADO' if self.debug else 'desactivado'}[/]\n"
            f"  Max rondas: [cyan]{max_rondas}[/]",
            border_style="bright_blue",
        ))

        if not self.api.crear_alias(self.alias):
            console.print(f"[yellow]⚠ No se pudo crear el alias '{self.alias}', "
                          f"puede que ya exista.[/]")

        for ronda in range(1, max_rondas + 1):
            console.print(f"\n[bold cyan]🔄 RONDA {ronda}/{max_rondas}[/]")

            completado = self._ejecutar_ronda()
            if completado:
                break

            if ronda < max_rondas:
                console.print(
                    f"\n[dim]⏳ Esperando {self.pausa_entre_rondas}s para respuestas…[/]")
                time.sleep(self.pausa_entre_rondas)

        self._mostrar_resumen()

    # =====================================================================
    # VISUALIZACIÓN (rich)
    # =====================================================================

    def _mostrar_resumen(self):
        """Muestra resumen de la ejecución con rich."""
        estado = self._actualizar_estado()

        # ── Tabla de resumen ─────────────────────────────────────────────
        table = Table(title="📊 Resumen de Ejecución", border_style="bright_blue",
                      show_header=False, padding=(0, 2))
        table.add_column("Campo", style="bold")
        table.add_column("Valor")

        table.add_row("💰 Oro final", str(estado.get("oro", 0)))

        obj_ok = estado.get("objetivo_completado", False)
        table.add_row("🎯 Objetivo",
                      "[green]✅ COMPLETADO[/]" if obj_ok else "[red]❌ PENDIENTE[/]")

        if estado.get("necesidades"):
            table.add_row("📋 Aún falta", str(estado["necesidades"]))

        table.add_row("🔄 Intercambios", str(len(self.intercambios_realizados)))
        table.add_row("🛡️ Lista negra", str(len(self.lista_negra)))

        console.print()
        console.print(table)

        # ── Detalle de intercambios ──────────────────────────────────────
        if self.intercambios_realizados:
            console.print("\n[bold]Intercambios realizados:[/]")
            for i in self.intercambios_realizados:
                console.print(f"   → {i['destinatario']}: {i['recursos']}")

        if self.lista_negra:
            console.print("\n[bold]Lista negra:[/]")
            for p in self.lista_negra:
                console.print(f"   ⚠️  {p}")

    def ver_log(self, ultimos: int = 20):
        """Muestra las últimas entradas del log (compatibilidad)."""
        console.print(f"\n[bold]📜 LOG (últimas {ultimos} entradas):[/]")
        console.rule()
        # En el nuevo sistema, loguru ya se encarga. Mostramos mensaje.
        console.print("[dim]El log completo se encuentra en los handlers de loguru.[/]")
        console.print("[dim]Si debug está activo, todo se muestra en consola.[/]")
