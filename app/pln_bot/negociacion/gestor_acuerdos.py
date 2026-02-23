"""
Gestión de acuerdos pendientes, expirados y aceptaciones.
"""

import time
from typing import Any, Dict, Optional

from .utilidades_mensajes import (
    RE_ASUNTO_RECURSOS,
    RE_RECURSO_INDIVIDUAL,
    extraer_recursos_mencionados,
    extraer_tx_id,
)


def registrar_acuerdo_pendiente(
    agente,
    remitente: str,
    recursos_dar: Dict[str, int],
    recursos_pedir: Dict[str, int],
    tx_id: str,
):
    """Guarda un acuerdo pendiente para responder cuando llegue su aceptación."""
    if remitente not in agente.acuerdos_pendientes:
        agente.acuerdos_pendientes[remitente] = []
    agente.acuerdos_pendientes[remitente].append(
        {
            "tx_id": tx_id,
            "recursos_dar": recursos_dar,
            "recursos_pedir": recursos_pedir,
            "timestamp": time.time(),
        }
    )


def mover_a_expirados_por_tx(
    agente, remitente: str, acuerdo: Dict[str, Any], ahora: float
):
    """Mueve un acuerdo pendiente expirado a caché temporal por tx_id."""
    tx_id = acuerdo.get("tx_id")
    if tx_id and tx_id in agente.tx_cerrados:
        return
    exp_data = {
        "remitente": remitente,
        "acuerdo": acuerdo,
        "expira_en": ahora + agente.ACUERDO_GRACIA_TTL_SEGUNDOS,
    }
    if tx_id:
        agente.acuerdos_expirados_tx[tx_id] = exp_data
    agente.acuerdos_expirados_por_remitente.setdefault(remitente, []).append(exp_data)


def limpiar_cache_tx(agente, ahora: float):
    """Limpia tx expirados y tx cerrados antiguos."""
    exp_tx_vencidos = [
        tx
        for tx, data in agente.acuerdos_expirados_tx.items()
        if data.get("expira_en", 0) <= ahora
    ]
    for tx in exp_tx_vencidos:
        del agente.acuerdos_expirados_tx[tx]

    for remitente in list(agente.acuerdos_expirados_por_remitente.keys()):
        vivos = [
            item
            for item in agente.acuerdos_expirados_por_remitente[remitente]
            if item.get("expira_en", 0) > ahora
        ]
        if vivos:
            agente.acuerdos_expirados_por_remitente[remitente] = vivos
        else:
            del agente.acuerdos_expirados_por_remitente[remitente]

    cerrados_vencidos = [
        tx
        for tx, ts in agente.tx_cerrados.items()
        if (ahora - ts) >= agente.TX_CERRADO_TTL_SEGUNDOS
    ]
    for tx in cerrados_vencidos:
        del agente.tx_cerrados[tx]


def responder_aceptacion(
    agente, remitente: str, mensaje_original: str, asunto_original: str = ""
) -> bool:
    """Responde a una aceptación enviando los recursos acordados."""
    tx_id_mensaje = extraer_tx_id(asunto_original, mensaje_original)
    if tx_id_mensaje and tx_id_mensaje in agente.tx_cerrados:
        agente._log(
            "INFO",
            f"Aceptación duplicada de {remitente} para tx={tx_id_mensaje} (ya cerrado)",
        )
        return False

    acuerdos = agente.acuerdos_pendientes.get(remitente, [])
    acuerdo: Optional[Dict[str, Any]] = None
    acuerdo_idx: Optional[int] = None
    origen_acuerdo = "none"
    ahora = time.time()

    if tx_id_mensaje:
        for i, ac in enumerate(acuerdos):
            if ac.get("tx_id") == tx_id_mensaje:
                acuerdo = ac
                acuerdo_idx = i
                origen_acuerdo = "pendiente"
                break

        if acuerdo is None:
            acuerdo_exp = agente.acuerdos_expirados_tx.get(tx_id_mensaje)
            if acuerdo_exp:
                if (
                    acuerdo_exp.get("expira_en", 0) >= ahora
                    and acuerdo_exp.get("remitente") == remitente
                ):
                    acuerdo = acuerdo_exp.get("acuerdo")
                    origen_acuerdo = "expirado"
                    agente._log(
                        "INFO",
                        f"Aceptación tardía de {remitente} para tx={tx_id_mensaje} "
                        f"(recuperado de caché de expirados)",
                    )
                else:
                    agente.acuerdos_expirados_tx.pop(tx_id_mensaje, None)

        if acuerdo is None:
            agente._log(
                "INFO",
                f"Aceptación de {remitente} con tx={tx_id_mensaje} "
                f"sin acuerdo pendiente/expirado coincidente",
            )
            return False
    else:
        m_asunto = RE_ASUNTO_RECURSOS.search(asunto_original or "")
        asunto_mi: Dict[str, int] = {}
        asunto_tu: Dict[str, int] = {}
        if m_asunto:
            for cant, rec in RE_RECURSO_INDIVIDUAL.findall(m_asunto.group(1)):
                asunto_mi[rec.lower()] = asunto_mi.get(rec.lower(), 0) + int(cant)
            for cant, rec in RE_RECURSO_INDIVIDUAL.findall(m_asunto.group(2)):
                asunto_tu[rec.lower()] = asunto_tu.get(rec.lower(), 0) + int(cant)

        if acuerdos:
            if asunto_mi or asunto_tu:
                for i, ac in enumerate(acuerdos):
                    dar_norm = {
                        str(k).lower(): int(v)
                        for k, v in ac.get("recursos_dar", {}).items()
                    }
                    pedir_norm = {
                        str(k).lower(): int(v)
                        for k, v in ac.get("recursos_pedir", {}).items()
                    }
                    if dar_norm == asunto_mi and pedir_norm == asunto_tu:
                        acuerdo = ac
                        acuerdo_idx = i
                        origen_acuerdo = "pendiente"
                        break
            if acuerdo is None and len(acuerdos) == 1:
                acuerdo = acuerdos[0]
                acuerdo_idx = 0
                origen_acuerdo = "pendiente"
            if acuerdo is None and len(acuerdos) > 1:
                recursos_mencionados = set(
                    extraer_recursos_mencionados(mensaje_original)
                )
                candidatos = []
                for i, ac in enumerate(acuerdos):
                    recursos_acuerdo = set(ac.get("recursos_dar", {})) | set(
                        ac.get("recursos_pedir", {})
                    )
                    if recursos_mencionados and recursos_mencionados.issubset(
                        recursos_acuerdo
                    ):
                        candidatos.append(i)
                if len(candidatos) == 1:
                    acuerdo_idx = candidatos[0]
                    acuerdo = acuerdos[acuerdo_idx]
                    origen_acuerdo = "pendiente"
                elif len(candidatos) > 1:
                    acuerdo_idx = min(
                        candidatos, key=lambda i: acuerdos[i].get("timestamp", 0)
                    )
                    acuerdo = acuerdos[acuerdo_idx]
                    origen_acuerdo = "pendiente"
                    agente._log(
                        "INFO",
                        f"Aceptación de {remitente} sin tx: varios candidatos; "
                        f"se aplica FIFO entre coincidencias",
                    )
                else:
                    acuerdo_idx = min(
                        range(len(acuerdos)),
                        key=lambda i: acuerdos[i].get("timestamp", 0),
                    )
                    acuerdo = acuerdos[acuerdo_idx]
                    origen_acuerdo = "pendiente"
                    agente._log(
                        "INFO",
                        f"Aceptación de {remitente} sin tx y sin señales claras; "
                        f"se aplica FIFO",
                    )
        else:
            expirados = [
                item
                for item in agente.acuerdos_expirados_por_remitente.get(remitente, [])
                if item.get("expira_en", 0) >= ahora
            ]
            if expirados:
                elegido = min(
                    expirados, key=lambda item: item["acuerdo"].get("timestamp", 0)
                )
                acuerdo = elegido.get("acuerdo")
                origen_acuerdo = "expirado"
                agente._log(
                    "INFO",
                    f"Aceptación tardía de {remitente} sin tx: "
                    f"se recupera acuerdo expirado por FIFO",
                )
            else:
                agente._log(
                    "INFO",
                    f"Aceptación de {remitente} sin acuerdo pendiente registrado",
                )
                return False

    if acuerdo is None:
        agente._log("ERROR", f"Error interno resolviendo acuerdo de {remitente}")
        return False

    recursos_a_enviar = acuerdo.get("recursos_dar", {})
    recursos_a_recibir = acuerdo.get("recursos_pedir", {})

    if not recursos_a_enviar:
        agente._log("INFO", f"Acuerdo con {remitente} sin recursos a enviar")
        return False

    agente._actualizar_estado()
    mis_recursos = agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
    objetivo = agente.info_actual.get("Objetivo", {}) if agente.info_actual else {}
    for rec, cant in recursos_a_enviar.items():
        disponible = mis_recursos.get(rec, 0)
        if disponible < cant:
            agente._log(
                "ALERTA",
                f"No puedo cumplir acuerdo con {remitente}: "
                f"necesito {cant} {rec} pero solo tengo {disponible}",
            )
            return False
        if rec != "oro":
            minimo_objetivo = objetivo.get(rec, 0)
            restante = disponible - cant
            if restante < minimo_objetivo:
                agente._log(
                    "ALERTA",
                    f"No puedo cumplir acuerdo con {remitente}: "
                    f"enviar {cant} {rec} rompería objetivo "
                    f"({rec} quedaría en {restante}/{minimo_objetivo})",
                )
                return False

    envio_str = ", ".join(f"{c} {r}" for r, c in recursos_a_enviar.items())
    recibo_str = ", ".join(f"{c} {r}" for r, c in recursos_a_recibir.items())
    tx_info = acuerdo.get("tx_id")
    tx_tag = f" [tx:{tx_info}]" if tx_info else ""

    agente._log(
        "DECISION",
        f"🤝 Ejecutando acuerdo{tx_tag} con {remitente}: "
        f"doy {envio_str} por {recibo_str}",
    )

    if agente._enviar_paquete(remitente, recursos_a_enviar):
        if origen_acuerdo == "pendiente" and acuerdo_idx is not None:
            acuerdos_actuales = agente.acuerdos_pendientes.get(remitente, [])
            if 0 <= acuerdo_idx < len(acuerdos_actuales):
                acuerdos_actuales.pop(acuerdo_idx)
            if not acuerdos_actuales:
                agente.acuerdos_pendientes.pop(remitente, None)
        if tx_info:
            agente.tx_cerrados[tx_info] = time.time()
            agente.acuerdos_expirados_tx.pop(tx_info, None)
            if remitente in agente.acuerdos_expirados_por_remitente:
                agente.acuerdos_expirados_por_remitente[remitente] = [
                    item
                    for item in agente.acuerdos_expirados_por_remitente[remitente]
                    if item.get("acuerdo", {}).get("tx_id") != tx_info
                ]
                if not agente.acuerdos_expirados_por_remitente[remitente]:
                    del agente.acuerdos_expirados_por_remitente[remitente]
        return True

    agente._log("ERROR", f"No se pudo enviar paquete a {remitente}")
    return False
