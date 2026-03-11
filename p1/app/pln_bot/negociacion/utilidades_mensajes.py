"""
Utilidades de parsing y clasificación de mensajes de negociación.
"""

import re as _re
from typing import Dict, Iterable, List, Optional

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
RE_TX_ID_TAG = _re.compile(r"\[(?:t?tx):([a-z0-9_-]{6,64})\]", _re.IGNORECASE)
RE_TX_ID_INLINE = _re.compile(
    r"\b(?:t?tx)\s*[:=#-]\s*([a-z0-9_-]{6,64})\b", _re.IGNORECASE
)

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
RE_TOKEN = _re.compile(r"\b([a-záéíóúñ][a-z0-9_áéíóúñ-]{1,})\b", _re.IGNORECASE)
RE_CUERPO_INTERCAMBIO = _re.compile(
    r"yo\s+te\s+doy\s+(.+?)\s+y\s+t[úu]\s+me\s+das\s+(.+?)(?:[\.\n]|$)",
    _re.IGNORECASE | _re.DOTALL,
)


def _parsear_recursos_texto(texto: str) -> Dict[str, int]:
    recursos: Dict[str, int] = {}
    for cant, rec in RE_RECURSO_INDIVIDUAL.findall(texto or ""):
        try:
            cantidad = int(cant)
        except (TypeError, ValueError):
            continue
        if cantidad <= 0:
            continue
        recurso_norm = rec.strip().lower()
        if not recurso_norm:
            continue
        recursos[recurso_norm] = recursos.get(recurso_norm, 0) + cantidad
    return recursos


def extraer_oferta_estructurada(
    asunto: str = "", mensaje: str = ""
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Extrae oferta estructurada de asunto/cuerpo en formato plantilla.

    Devuelve `(ofrecen, piden)` donde:
    - `ofrecen`: recursos que envía el remitente
    - `piden`: recursos que solicita al destinatario
    """
    m_asunto = RE_ASUNTO_RECURSOS.search(asunto or "")
    if m_asunto:
        ofrecen = _parsear_recursos_texto(m_asunto.group(1))
        piden = _parsear_recursos_texto(m_asunto.group(2))
        if ofrecen and piden:
            return ofrecen, piden

    m_cuerpo = RE_CUERPO_INTERCAMBIO.search(mensaje or "")
    if m_cuerpo:
        ofrecen = _parsear_recursos_texto(m_cuerpo.group(1))
        piden = _parsear_recursos_texto(m_cuerpo.group(2))
        if ofrecen and piden:
            return ofrecen, piden

    return {}, {}


def extraer_recursos_mencionados(
    mensaje: str, candidatos: Optional[Iterable[str]] = None
) -> List[str]:
    """Extrae recursos mencionados en un mensaje.

    Si se pasan `candidatos`, busca solo esos nombres (ideal para recursos dinámicos
    conocidos en runtime). Si no, usa heurística ligera basada en patrones del texto.
    """
    msg_lower = mensaje.lower()
    encontrados: List[str] = []

    if candidatos is not None:
        candidatos_norm = [str(c).strip().lower() for c in candidatos if str(c).strip()]
        for recurso in candidatos_norm:
            if _re.search(r"\b" + _re.escape(recurso) + r"\b", msg_lower):
                encontrados.append(recurso)
        if encontrados:
            return list(dict.fromkeys(encontrados))

    # Heurística general cuando no hay lista de candidatos:
    # 1) recursos con cantidad explícita ("2 madera")
    for _cant, recurso in RE_RECURSO_INDIVIDUAL.findall(msg_lower):
        recurso_norm = recurso.strip().lower()
        if recurso_norm:
            encontrados.append(recurso_norm)

    # 2) recursos tras verbos de intención ("necesito X", "quiero X", "busco X")
    for m in _re.finditer(r"\b(?:necesito|quiero|busco|pido)\s+(\w+)", msg_lower):
        recurso_norm = m.group(1).strip().lower()
        if recurso_norm:
            encontrados.append(recurso_norm)

    # 3) fallback suave: tokens útiles (evita puntuación/ruido básico)
    if not encontrados:
        stopwords = {
            "hola",
            "gracias",
            "oferta",
            "trato",
            "acuerdo",
            "conviene",
            "saludos",
            "te",
            "yo",
            "tu",
            "me",
            "das",
            "doy",
            "por",
            "mi",
            "si",
            "no",
            "el",
            "la",
            "de",
            "un",
            "una",
            "que",
        }
        for token in RE_TOKEN.findall(msg_lower):
            if token not in stopwords and len(token) >= 3:
                encontrados.append(token)

    return list(dict.fromkeys(encontrados))


def extraer_tx_id(*textos: str) -> Optional[str]:
    """Extrae tx_id/ttx de asunto/cuerpo en formato tag o inline."""
    for texto in textos:
        if not texto:
            continue
        m = RE_TX_ID_TAG.search(texto)
        if not m:
            m = RE_TX_ID_INLINE.search(texto)
        if m:
            return m.group(1).strip().lower()
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
                agente._registrar_backoff_combo(clave, "rechazo_recibido")
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
        agente._registrar_backoff_combo(clave, "rechazo_recibido")
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
            agente._registrar_backoff_combo(clave, "rechazo_propio")
