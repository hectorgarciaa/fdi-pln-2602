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
        self._modelo = (modelo or "").strip().lower()
        self._usar_no_think = self._modelo.startswith("qwen")
        self._max_chars_mensaje = 700
        ollama_base = OLLAMA_URL.rstrip("/") + "/v1"
        ai_provider = OllamaProvider(base_url=ollama_base)
        ai_model = OpenAIChatModel(modelo, provider=ai_provider)
        ai_settings = ModelSettings(
            temperature=0.0,
            top_p=0.3,
            max_tokens=320,
            timeout=15.0,
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
                "Si detectas oferta o intercambio, analiza primero el texto.\n"
                "Usa tools solo si necesitas validar stock/necesidad/excedente.\n"
                "Haz como máximo 2 llamadas de tools.\n\n"
                "Reglas de brevedad estrictas:\n"
                "- Responde SOLO con datos estructurados, sin texto adicional.\n"
                "- razon debe tener como máximo 15 palabras.\n"
                "- No repitas el mensaje original ni expliques pasos internos.\n"
                "- Si no hay oferta clara, devuelve ofrecen={} y piden={}.\n\n"
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
            retries=0,
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

    def _prefijo_prompt(self) -> str:
        """Desactiva pensamiento largo en modelos qwen para respuestas más estables."""
        if self._usar_no_think:
            return "/no_think\n"
        return ""

    def _recortar_texto(self, texto: str) -> str:
        if not isinstance(texto, str):
            return ""
        limpio = texto.strip()
        if len(limpio) <= self._max_chars_mensaje:
            return limpio
        return f"{limpio[: self._max_chars_mensaje]}..."

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
        mensaje_recortado = self._recortar_texto(mensaje)

        prompt_usuario = (
            f"{self._prefijo_prompt()}"
            "Analiza este mensaje y produce salida estructurada.\n\n"
            f"REMITENTE: {remitente}\n"
            f"ASUNTO: {asunto or '(sin asunto)'}\n"
            f"MODO_AGENTE: {modo_agente or '(desconocido)'}\n"
            f"NECESIDADES_PRINCIPALES: {self._resumen_contexto(necesidades)}\n"
            f"EXCEDENTES_PRINCIPALES: {self._resumen_contexto(excedentes)}\n\n"
            "MENSAJE:\n"
            f"{mensaje_recortado}\n\n"
            "Instrucciones críticas:\n"
            "- Usa tools solo si de verdad aportan valor.\n"
            "- Si decides contraofertar, rellena contraoferta_ofrezco y contraoferta_pido.\n"
            "- No inventes cantidades ni recursos.\n"
            "- Devuelve una razon MUY corta (maximo 15 palabras)."
        )
        limits_principal = UsageLimits(
            request_limit=1,
            tool_calls_limit=3,
            response_tokens_limit=600,
        )
        try:
            result = self._agente.run_sync(
                prompt_usuario, usage_limits=limits_principal
            )
            return result.output
        except Exception as err_principal:
            # Segundo intento más estricto y breve para evitar timeouts/tokens.
            prompt_rescate = (
                f"{self._prefijo_prompt()}"
                "MODO RESCATE: responde ultra-breve y estructurado.\n"
                "Sin tools. Sin explicación extra.\n\n"
                f"REMITENTE: {remitente}\n"
                f"ASUNTO: {asunto or '(sin asunto)'}\n"
                "MENSAJE:\n"
                f"{mensaje_recortado}\n\n"
                "Reglas:\n"
                "- Si no hay oferta explícita: ofrecen={} y piden={} y decision=ignorar.\n"
                "- Si hay oferta explícita: extrae recursos y cantidades exactas.\n"
                "- razon maximo 10 palabras."
            )
            limits_rescate = UsageLimits(
                request_limit=1,
                tool_calls_limit=0,
                response_tokens_limit=180,
            )
            try:
                result = self._agente.run_sync(
                    prompt_rescate,
                    usage_limits=limits_rescate,
                )
                return result.output
            except Exception as err_rescate:
                raise RuntimeError(
                    "Fallo analisis pydantic_ai principal y rescate: "
                    f"{err_principal} | rescate: {err_rescate}"
                ) from err_rescate
