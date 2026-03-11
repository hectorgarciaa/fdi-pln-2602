"""
Construcción de propuestas y contraofertas.
"""

import uuid
from typing import Dict, List, Optional


def nuevo_tx_id() -> str:
    """Genera un identificador corto para emparejar propuestas y aceptaciones."""
    return uuid.uuid4().hex[:10]


def _en_modo_maximizar_oro(agente) -> bool:
    """Indica si el agente está en modo de venta de excedentes por oro."""
    modo = getattr(agente, "modo", None)
    modo_valor = getattr(modo, "value", str(modo)).strip().lower()
    return modo_valor == "maximizar_oro"


def generar_propuesta(
    agente, destinatario: str, necesidades: Dict, excedentes: Dict, oro: int
) -> Optional[Dict[str, str]]:
    """Genera una propuesta evitando combinaciones ya enviadas o rechazadas."""
    ofrezco: Dict[str, int] = {}
    pido: Dict[str, int] = {}

    # Usar excedentes reales (descontando recursos comprometidos)
    exc_disp = agente._excedentes_disponibles(excedentes)

    if exc_disp and necesidades:
        lista_necesidades = list(necesidades.keys())
        lista_excedentes = list(exc_disp.keys())
        total_combos = len(lista_necesidades) * len(lista_excedentes)
        bloqueadas_rechazo = 0
        bloqueadas_reciente = 0
        bloqueadas_backoff = 0

        for offset in range(total_combos):
            idx = (agente.propuesta_index + offset) % total_combos
            idx_ofrezco = idx // len(lista_necesidades)
            idx_pido = idx % len(lista_necesidades)

            recurso_ofrezco = lista_excedentes[idx_ofrezco]
            recurso_pido = lista_necesidades[idx_pido]
            clave = (destinatario, recurso_ofrezco, recurso_pido)

            if agente._rechazo_vigente(clave):
                bloqueadas_rechazo += 1
                continue
            en_backoff, _ = agente._combo_en_backoff(clave)
            if en_backoff:
                bloqueadas_backoff += 1
                continue
            if (
                clave in agente.propuestas_enviadas
                and agente.ronda_actual - agente.propuestas_enviadas[clave] < 2
            ):
                bloqueadas_reciente += 1
                continue

            cantidad_ofrezco = 1
            mis_recursos_totales = (
                agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
            )
            cantidad_total_recurso = mis_recursos_totales.get(recurso_ofrezco, 0)

            if cantidad_total_recurso <= 0 or exc_disp[recurso_ofrezco] <= 0:
                agente._log(
                    "DEBUG",
                    f"Saltando propuesta: no tenemos {recurso_ofrezco} disponible",
                )
                continue

            if cantidad_total_recurso > 15:
                cantidad_ofrezco = min(exc_disp[recurso_ofrezco], 3)
                cantidad_pido = 1
                agente._log(
                    "INFO",
                    f"Oferta generosa a {destinatario}: "
                    f"{cantidad_ofrezco} {recurso_ofrezco} por 1 {recurso_pido} "
                    f"(tenemos {cantidad_total_recurso} {recurso_ofrezco})",
                )
            else:
                cantidad_pido = 1

            if exc_disp[recurso_ofrezco] < cantidad_ofrezco:
                continue

            ofrezco = {recurso_ofrezco: cantidad_ofrezco}
            pido = {recurso_pido: cantidad_pido}
            agente.propuesta_index = idx + 1
            break
        else:
            encontrado = False
            if necesidades and oro >= 1:
                comprometidos = agente._recursos_comprometidos()
                oro_libre = oro - comprometidos.get("oro", 0)
                for recurso_pido in necesidades:
                    clave = (destinatario, "oro", recurso_pido)
                    if agente._rechazo_vigente(clave):
                        continue
                    en_backoff, _ = agente._combo_en_backoff(clave)
                    if en_backoff:
                        continue
                    if oro_libre >= 1:
                        ofrezco = {"oro": 1}
                        pido = {recurso_pido: 1}
                        encontrado = True
                        break
            if not encontrado:
                comprometidos = agente._recursos_comprometidos()
                agente._log(
                    "INFO",
                    f"Sin combinaciones nuevas para {destinatario}",
                    {
                        "rechazos_vigentes": len(agente.rechazos_recibidos),
                        "comprometidos": comprometidos,
                        "oro_libre": oro - comprometidos.get("oro", 0),
                        "bloq_rechazo": bloqueadas_rechazo,
                        "bloq_reciente": bloqueadas_reciente,
                        "bloq_backoff": bloqueadas_backoff,
                    },
                )
                return None

    elif necesidades and oro >= 1:
        comprometidos = agente._recursos_comprometidos()
        oro_libre = oro - comprometidos.get("oro", 0)
        encontrado = False
        for recurso_pido in necesidades:
            clave = (destinatario, "oro", recurso_pido)
            if agente._rechazo_vigente(clave):
                continue
            en_backoff, _ = agente._combo_en_backoff(clave)
            if en_backoff:
                continue
            if oro_libre >= 1:
                ofrezco = {"oro": 1}
                pido = {recurso_pido: 1}
                encontrado = True
                break
        if not encontrado:
            return None
    elif exc_disp:
        # Elegimos el excedente más alto para vender antes lo que más sobra.
        recurso_ofrezco = max(exc_disp.items(), key=lambda kv: (kv[1], kv[0]))[0]
        cantidad_ofrezco = 1
        en_max_oro = _en_modo_maximizar_oro(agente)
        cantidad_oro_pedida = max(3, cantidad_ofrezco * 3) if en_max_oro else 1
        clave = (destinatario, recurso_ofrezco, "oro")
        if agente._rechazo_vigente(clave):
            return None
        en_backoff, _ = agente._combo_en_backoff(clave)
        if en_backoff:
            return None
        ofrezco = {recurso_ofrezco: cantidad_ofrezco}
        pido = {"oro": cantidad_oro_pedida}
        if en_max_oro:
            agente._log(
                "INFO",
                f"Modo MAXIMIZAR_ORO: pidiendo {cantidad_oro_pedida} oro "
                f"por {cantidad_ofrezco} {recurso_ofrezco}",
            )
    else:
        return None

    ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
    pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())
    tx_id = nuevo_tx_id()

    cuerpo = (
        f"Hola {destinatario}, soy {agente.alias}. "
        f"[tx:{tx_id}] "
        f"Te propongo un intercambio: "
        f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
        f"Para este trato, usa siempre este id: [tx:{tx_id}]. "
        f"Si aceptas, responde 'acepto el trato [tx:{tx_id}]'. "
        f"Si no te conviene, responde 'no me conviene [tx:{tx_id}]'. "
        f"Saludos, {agente.alias}"
    )

    return {
        "asunto": f"Propuesta: [tx:{tx_id}] mi {ofrezco_str} por tu {pido_str}",
        "cuerpo": cuerpo,
        "_ofrezco": ofrezco,
        "_pido": pido,
        "_tx_id": tx_id,
    }


def generar_contraoferta(
    agente,
    destinatario: str,
    ofrecen: Dict[str, int],
    necesidades: Dict,
    excedentes: Dict,
) -> Optional[Dict]:
    """Genera contraoferta cuando la oferta original no nos sirve."""
    pido = {}
    for rec, cant in ofrecen.items():
        if rec in necesidades:
            pido[rec] = 1
            break
    if not pido:
        return None

    exc_disp = agente._excedentes_disponibles(excedentes)

    ofrezco = {}
    for rec, cant in exc_disp.items():
        if cant >= 1:
            ofrezco[rec] = 1
            break

    if not ofrezco:
        recursos = agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
        oro_total = recursos.get("oro", 0)
        comprometidos = agente._recursos_comprometidos()
        oro_libre = oro_total - comprometidos.get("oro", 0)
        if oro_libre >= 1:
            ofrezco = {"oro": 1}
        else:
            return None

    for r_o in ofrezco:
        for r_p in pido:
            if agente._rechazo_vigente((destinatario, r_o, r_p)):
                agente._log(
                    "INFO",
                    f"Contraoferta a {destinatario} ya rechazada: "
                    f"{r_o}→{r_p} — no repetir",
                )
                return None

    ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
    pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())
    tx_id = nuevo_tx_id()

    cuerpo = (
        f"Hola {destinatario}, soy {agente.alias}. "
        f"[tx:{tx_id}] "
        f"Vi tu oferta pero no tengo lo que pides. "
        f"Te hago una contrapropuesta: "
        f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
        f"Para este trato, usa siempre este id: [tx:{tx_id}]. "
        f"Si aceptas, responde 'acepto el trato [tx:{tx_id}]'. "
        f"Si no te conviene, responde 'no me conviene [tx:{tx_id}]'. "
        f"Saludos, {agente.alias}"
    )

    return {
        "asunto": f"Contrapropuesta: [tx:{tx_id}] mi {ofrezco_str} por tu {pido_str}",
        "cuerpo": cuerpo,
        "_ofrezco": ofrezco,
        "_pido": pido,
        "_tx_id": tx_id,
    }


def generar_propuesta_adaptada(
    agente,
    destinatario: str,
    recursos_que_quiere: List[str],
    necesidades: Dict,
    excedentes: Dict,
    oro: int,
) -> Optional[Dict]:
    """Genera una propuesta adaptada a lo que el otro jugador pidió."""
    exc_disp = agente._excedentes_disponibles(excedentes)

    ofrezco: Dict[str, int] = {}
    for rec in recursos_que_quiere:
        if rec in exc_disp and exc_disp[rec] > 0:
            ofrezco[rec] = 1
            break

    if not ofrezco:
        return None

    pido: Dict[str, int] = {}
    for rec, cant in necesidades.items():
        clave = (destinatario, list(ofrezco.keys())[0], rec)
        if not agente._rechazo_vigente(clave):
            pido[rec] = 1
            break

    if not pido:
        return None

    ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
    pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())
    tx_id = nuevo_tx_id()

    cuerpo = (
        f"Hola {destinatario}, soy {agente.alias}. "
        f"[tx:{tx_id}] "
        f"He visto que necesitas {ofrezco_str}. "
        f"Te propongo un intercambio: "
        f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
        f"Para este trato, usa siempre este id: [tx:{tx_id}]. "
        f"Si aceptas, responde 'acepto el trato [tx:{tx_id}]'. "
        f"Si no te conviene, responde 'no me conviene [tx:{tx_id}]'. "
        f"Saludos, {agente.alias}"
    )

    return {
        "asunto": f"Propuesta: [tx:{tx_id}] mi {ofrezco_str} por tu {pido_str}",
        "cuerpo": cuerpo,
        "_ofrezco": ofrezco,
        "_pido": pido,
        "_tx_id": tx_id,
    }
