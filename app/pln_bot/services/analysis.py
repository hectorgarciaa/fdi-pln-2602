"""
Servicio de análisis de mensajes de negociación usando salida estructurada.
"""

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ..core.config import OLLAMA_URL


class RespuestaUnificada(BaseModel):
    """Respuesta única de la IA: parsing + decisión de negociación."""

    es_aceptacion: bool = False
    ofrecen: Dict[str, int] = Field(default_factory=dict)
    piden: Dict[str, int] = Field(default_factory=dict)
    decision: Literal["aceptar", "rechazar", "contraofertar", "ignorar"] = "ignorar"
    contraoferta_ofrezco: Dict[str, int] = Field(default_factory=dict)
    contraoferta_pido: Dict[str, int] = Field(default_factory=dict)
    razon: str = ""

    from pydantic import field_validator

    @field_validator(
        "ofrecen", "piden", "contraoferta_ofrezco", "contraoferta_pido", mode="before"
    )
    @classmethod
    def _coerce_to_dict(cls, v) -> Dict[str, int]:
        if isinstance(v, dict):
            normalizado: Dict[str, int] = {}
            for k, val in v.items():
                if val is None:
                    continue
                try:
                    cantidad = int(val)
                except (TypeError, ValueError):
                    continue
                if cantidad <= 0:
                    continue
                normalizado[str(k).strip().lower()] = cantidad
            return normalizado
        return {}

    @field_validator("decision", mode="before")
    @classmethod
    def _normalizar_decision(cls, v) -> str:
        if not isinstance(v, str):
            return "ignorar"
        decision = v.strip().lower()
        decisiones_validas = {"aceptar", "rechazar", "contraofertar", "ignorar"}
        alias = {
            "aceptar_oferta": "aceptar",
            "accept": "aceptar",
            "reject": "rechazar",
            "contraoferta": "contraofertar",
            "counteroffer": "contraofertar",
            "ignore": "ignorar",
        }
        return alias.get(
            decision, decision if decision in decisiones_validas else "ignorar"
        )


class AnalisisMensajesService:
    """Encapsula la llamada al modelo para analizar cartas."""

    def __init__(self, modelo: str):
        ollama_base = OLLAMA_URL.rstrip("/") + "/v1"
        ai_provider = OllamaProvider(base_url=ollama_base)
        ai_model = OpenAIChatModel(modelo, provider=ai_provider)
        ai_settings = ModelSettings(
            temperature=0.1,
            top_p=0.8,
            max_tokens=256,
            timeout=25.0,
            parallel_tool_calls=False,
        )
        self._contexto: Dict[str, Dict[str, int] | str] = {
            "necesidades": {},
            "excedentes": {},
            "recursos": {},
            "objetivo": {},
            "modo_agente": "",
        }

        self._agente = Agent(
            ai_model,
            output_type=RespuestaUnificada,
            system_prompt=(
                "Eres un analizador de mensajes en un juego de intercambio de recursos.\n"
                "Debes analizar el mensaje y tomar una decisión de negociación.\n"
                "Si detectas oferta o intercambio, usa tools antes de decidir.\n"
                "Debes invocar tools para validar stock/necesidad/excedente.\n\n"
                "Responde con TODOS estos campos:\n\n"
                "1) es_aceptacion: ¿El mensaje ACEPTA un trato previo?\n"
                "   - SÍ si contiene frases como: 'acepto el trato', 'trato hecho',\n"
                "     'te he enviado', 'cerramos el trato', 'acepto tu propuesta',\n"
                "     'de acuerdo', 'perfecto, te envío'.\n"
                "   - NO si es: rechazo, propuesta nueva, 'si aceptas dime',\n"
                "     'no me conviene', 'por ahora no', 'no me interesa'.\n\n"
                "2) ofrecen / piden: EXTRAER recursos del mensaje.\n"
                '   - "ofrecen" = lo que el remitente ofrece DAR (lo que yo recibiría).\n'
                '   - "piden" = lo que el remitente quiere RECIBIR (lo que yo daría).\n'
                "   - Solo recursos y cantidades EXPLÍCITOS. NO inventes.\n"
                "   - Si hay ambigüedad o falta cantidad explícita, omítelo.\n"
                "   - Normaliza recursos en minúsculas y sin espacios extra.\n"
                "   - Rechazo, saludo o no-propuesta → ofrecen={}, piden={}.\n\n"
                "3) decision: una de {aceptar, rechazar, contraofertar, ignorar}.\n"
                "   - Si es aceptación de trato previo: decision=aceptar.\n"
                "   - Si no hay oferta clara: decision=ignorar.\n"
                "   - Si hay oferta mala pero recuperable: contraofertar.\n\n"
                "4) contraoferta_ofrezco / contraoferta_pido:\n"
                "   - Solo si decision=contraofertar.\n"
                "   - En otros casos: {}.\n"
                "   - No inventes recursos/cantidades fuera del contexto.\n\n"
                "5) razon: explicación breve de tu análisis.\n\n"
                'Ejemplo: "yo te doy 2 madera y tú me das 3 piedra"\n'
                '→ es_aceptacion=false, ofrecen={"madera": 2}, '
                'piden={"piedra": 3}, decision=...'
            ),
            model_settings=ai_settings,
            retries=1,
        )
        self._registrar_tools()

    @staticmethod
    def _normalizar_recursos(recursos: Optional[Dict[str, int]]) -> Dict[str, int]:
        if not isinstance(recursos, dict):
            return {}
        normalizado: Dict[str, int] = {}
        for recurso, cantidad in recursos.items():
            try:
                cant = int(cantidad)
            except (TypeError, ValueError):
                continue
            if cant < 0:
                continue
            normalizado[str(recurso).strip().lower()] = cant
        return normalizado

    def _actualizar_contexto(
        self,
        modo_agente: str,
        necesidades: Optional[Dict[str, int]],
        excedentes: Optional[Dict[str, int]],
        recursos_actuales: Optional[Dict[str, int]],
        objetivo: Optional[Dict[str, int]],
    ) -> None:
        self._contexto = {
            "modo_agente": modo_agente or "",
            "necesidades": self._normalizar_recursos(necesidades),
            "excedentes": self._normalizar_recursos(excedentes),
            "recursos": self._normalizar_recursos(recursos_actuales),
            "objetivo": self._normalizar_recursos(objetivo),
        }

    def _registrar_tools(self) -> None:
        @self._agente.tool_plain
        def consultar_necesidad(recurso: str) -> int:
            """Devuelve cuántas unidades del recurso necesitamos actualmente."""
            recurso_norm = recurso.strip().lower()
            necesidades = self._contexto.get("necesidades", {})
            return int(necesidades.get(recurso_norm, 0))

        @self._agente.tool_plain
        def consultar_excedente(recurso: str) -> int:
            """Devuelve cuántas unidades del recurso nos sobran actualmente."""
            recurso_norm = recurso.strip().lower()
            excedentes = self._contexto.get("excedentes", {})
            return int(excedentes.get(recurso_norm, 0))

        @self._agente.tool_plain
        def consultar_stock(recurso: str) -> int:
            """Devuelve cuántas unidades del recurso tenemos en inventario."""
            recurso_norm = recurso.strip().lower()
            recursos = self._contexto.get("recursos", {})
            return int(recursos.get(recurso_norm, 0))

        @self._agente.tool_plain
        def consultar_objetivo(recurso: str) -> int:
            """Devuelve la cantidad objetivo del recurso para ganar la partida."""
            recurso_norm = recurso.strip().lower()
            objetivo = self._contexto.get("objetivo", {})
            return int(objetivo.get(recurso_norm, 0))

        @self._agente.tool_plain
        def puedo_entregar(recurso: str, cantidad: int) -> bool:
            """Valida si podemos entregar ese recurso sin romper el objetivo."""
            recurso_norm = recurso.strip().lower()
            try:
                cant = int(cantidad)
            except (TypeError, ValueError):
                return False
            if cant <= 0:
                return False

            recursos = self._contexto.get("recursos", {})
            objetivo = self._contexto.get("objetivo", {})
            stock = int(recursos.get(recurso_norm, 0))
            if stock < cant:
                return False

            if recurso_norm == "oro":
                return True

            minimo = int(objetivo.get(recurso_norm, 0))
            return (stock - cant) >= minimo

    @staticmethod
    def _resumen_contexto(recursos: Optional[Dict[str, int]], limite: int = 8) -> str:
        if not recursos:
            return "ninguno"
        items = sorted(recursos.items(), key=lambda kv: (-kv[1], kv[0]))[:limite]
        return ", ".join(f"{rec}:{cant}" for rec, cant in items)

    def analizar(
        self,
        remitente: str,
        mensaje: str,
        asunto: str = "",
        necesidades: Optional[Dict[str, int]] = None,
        excedentes: Optional[Dict[str, int]] = None,
        recursos_actuales: Optional[Dict[str, int]] = None,
        objetivo: Optional[Dict[str, int]] = None,
        modo_agente: str = "",
    ) -> RespuestaUnificada:
        self._actualizar_contexto(
            modo_agente=modo_agente,
            necesidades=necesidades,
            excedentes=excedentes,
            recursos_actuales=recursos_actuales,
            objetivo=objetivo,
        )

        prompt_usuario = (
            "Analiza este mensaje y produce salida estructurada.\n\n"
            f"REMITENTE: {remitente}\n"
            f"ASUNTO: {asunto or '(sin asunto)'}\n"
            f"MODO_AGENTE: {modo_agente or '(desconocido)'}\n"
            f"NECESIDADES_PRINCIPALES: {self._resumen_contexto(necesidades)}\n"
            f"EXCEDENTES_PRINCIPALES: {self._resumen_contexto(excedentes)}\n\n"
            "MENSAJE:\n"
            f"{mensaje}\n\n"
            "Instrucciones críticas:\n"
            "- Si hay oferta, llama tools y valida stock/necesidad/excedente.\n"
            "- Si decides contraofertar, rellena contraoferta_ofrezco y contraoferta_pido.\n"
            "- No inventes cantidades ni recursos."
        )
        result = self._agente.run_sync(
            prompt_usuario,
            usage_limits=UsageLimits(
                request_limit=1,
                tool_calls_limit=8,
                response_tokens_limit=256,
            ),
        )
        return result.output
