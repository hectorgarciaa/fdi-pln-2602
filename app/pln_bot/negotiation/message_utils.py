"""
Utilidades de parsing y clasificación de mensajes de negociación.
"""

import re as _re
from typing import Dict, List, Optional

from ..core.config import RECURSOS_CONOCIDOS

# Detectores de intención en texto libre
RE_RECHAZO = _re.compile(
    r"no me interesa|no acepto|no puedo aceptar|rechaz|no,? gracias"
    r"|no tengo lo que|no necesito|no quiero|paso de"
    r"|no me conviene|por ahora no|no es lo que busco"
    r"|no puedo hacer ese|no me sirve|mejor no",
    _re.IGNORECASE,
)
RE_ACEPTACION = _re.compile(
    r"acepto el trato|trato hecho|te he enviado|cerramos el trato"
    r"|acepto tu propuesta|acepto,? dime|de acuerdo.*env[ií]"
    r"|perfecto.*env[ií]|hecho.*te mando",
    _re.IGNORECASE,
)
RE_PROPUESTA = _re.compile(
    r"\b(ofrezco|te doy|te ofrezco|propongo|te propongo|pido|"
    r"necesito|quiero|busco|doy .* por|a cambio de|\d+\s+\w+.*por)",
    _re.IGNORECASE,
)
RE_RESPUESTA_RECHAZO = _re.compile(
    r"^\s*Gracias por la oferta.*no me conviene"
    r"|^\s*Gracias por la oferta.*Saludos"
    r"|^\s*No me interesa.*Saludos"
    r"|^\s*Por ahora no.*Saludos",
    _re.IGNORECASE | _re.DOTALL,
)
RE_TX_ID = _re.compile(r"\[tx:([a-z0-9_-]{6,64})\]", _re.IGNORECASE)

# Regex para recursos en asunto de propuestas
RE_ASUNTO_PROPUESTA = _re.compile(
    r"(?:mi|Propuesta:?\s*mi|Contrapropuesta:?\s*mi)\s+.*?(\w+)\s+por\s+tu\s+.*?(\w+)",
    _re.IGNORECASE,
)
RE_ASUNTO_RECURSOS = _re.compile(
    r"mi\s+(.+?)\s+por\s+tu\s+(.+?)\s*$",
    _re.IGNORECASE,
)
RE_RECURSO_INDIVIDUAL = _re.compile(r"(\d+)\s+(\w+)")


def extraer_recursos_mencionados(mensaje: str) -> List[str]:
    """Extrae recursos conocidos mencionados en un mensaje de texto."""
    msg_lower = mensaje.lower()
    encontrados = []
    for recurso in RECURSOS_CONOCIDOS:
        if _re.search(r"\b" + _re.escape(recurso) + r"\b", msg_lower):
            encontrados.append(recurso)
    return encontrados


def extraer_tx_id(*textos: str) -> Optional[str]:
    """Extrae tx_id de asunto/cuerpo si existe el tag [tx:...]."""
    for texto in textos:
        if not texto:
            continue
        m = RE_TX_ID.search(texto)
        if m:
            return m.group(1).lower()
    return None


def es_carta_sistema(remitente: str, mensaje: str) -> bool:
    """Devuelve True para notificaciones del sistema (no son propuestas)."""
    if remitente.lower() in ("sistema", "server", "butler"):
        return True
    if _re.match(
        r"(?i)^(has recibido|recursos generados|paquete|\s*$)", mensaje.strip()
    ):
        return True
    return False


def es_rechazo_simple(mensaje: str, asunto: str = "") -> bool:
    """Detecta rechazos textuales sin necesidad de IA."""
    if RE_RESPUESTA_RECHAZO.search(mensaje):
        return True
    asunto_limpio = asunto.strip().lower()
    if asunto_limpio.startswith("re:") and RE_RECHAZO.search(mensaje):
        return True
    if RE_PROPUESTA.search(mensaje):
        return False
    return bool(RE_RECHAZO.search(mensaje))


def es_mensaje_corto_sin_propuesta(mensaje: str) -> bool:
    """Detecta mensajes muy cortos que no contienen propuesta."""
    limpio = mensaje.strip()
    if len(limpio) < 15:
        return not RE_PROPUESTA.search(limpio)
    return False


def es_aceptacion_simple(mensaje: str, asunto: str = "") -> bool:
    """Detecta aceptaciones textuales sin necesidad de IA."""
    asunto_limpio = asunto.strip().lower()
    if asunto_limpio.startswith(("propuesta:", "contrapropuesta:")):
        return False
    if RE_PROPUESTA.search(mensaje):
        return False
    return bool(RE_ACEPTACION.search(mensaje))


def registrar_rechazo(agente, remitente: str, asunto: str):
    """Registra un rechazo extrayendo recursos del asunto."""
    m = RE_ASUNTO_RECURSOS.search(asunto)
    if m:
        parte_ofrezco = m.group(1)
        parte_pido = m.group(2)
        recs_ofrezco = RE_RECURSO_INDIVIDUAL.findall(parte_ofrezco)
        recs_pido = RE_RECURSO_INDIVIDUAL.findall(parte_pido)
        for _, r_o in recs_ofrezco:
            for _, r_p in recs_pido:
                clave = (remitente, r_o.lower(), r_p.lower())
                agente.rechazos_recibidos[clave] = agente.ronda_actual
        if recs_ofrezco and recs_pido:
            ofr_str = ", ".join(r for _, r in recs_ofrezco)
            pid_str = ", ".join(r for _, r in recs_pido)
            agente._log(
                "INFO",
                f"Rechazo registrado de {remitente}: {ofr_str}→{pid_str} (no repetir)",
            )
            return

    m = RE_ASUNTO_PROPUESTA.search(asunto)
    if m:
        recurso_ofrezco = m.group(1).lower()
        recurso_pido = m.group(2).lower()
        clave = (remitente, recurso_ofrezco, recurso_pido)
        agente.rechazos_recibidos[clave] = agente.ronda_actual
        agente._log(
            "INFO",
            f"Rechazo registrado de {remitente}: {recurso_ofrezco}→{recurso_pido} (no repetir)",
        )


def registrar_rechazo_propio(
    agente, remitente: str, ofrecen: Dict[str, int], piden: Dict[str, int]
):
    """Registra que nosotros rechazamos una oferta para no repetirla invertida."""
    for r_o in ofrecen:
        for r_p in piden:
            clave = (remitente, r_p, r_o)
            agente.rechazos_recibidos[clave] = agente.ronda_actual
