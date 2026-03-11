"""
Envío de propuestas iniciales de intercambio.
"""

import time
from typing import Dict

from .gestor_acuerdos import registrar_acuerdo_pendiente
from .constructor_propuestas import generar_propuesta


def enviar_propuestas(agente, necesidades: Dict, excedentes: Dict, oro: int):
    """Envía propuestas a jugadores no contactados (máx configurable/ronda)."""
    jugadores = agente._obtener_jugadores_disponibles()
    jugadores = [j for j in jugadores if j not in agente.contactados_esta_ronda]
    max_envios = max(1, int(getattr(agente, "max_propuestas_por_ronda", 3)))
    envios_realizados = 0

    if not jugadores:
        agente._log("INFO", "No hay jugadores a quienes enviar propuestas esta ronda")
        return

    for jugador in jugadores:
        if agente.acuerdos_pendientes.get(jugador):
            agente._log(
                "INFO",
                f"Saltando {jugador}: ya hay acuerdo(s) pendiente(s) con ese jugador",
            )
            continue

        propuesta = generar_propuesta(agente, jugador, necesidades, excedentes, oro)
        if propuesta is None:
            agente._log("INFO", f"No se generó propuesta para {jugador}")
            continue

        if agente._enviar_carta(jugador, propuesta["asunto"], propuesta["cuerpo"]):
            agente.contactados_esta_ronda.append(jugador)
            registrar_acuerdo_pendiente(
                agente,
                jugador,
                propuesta["_ofrezco"],
                propuesta["_pido"],
                propuesta["_tx_id"],
            )

            # Registrar en memoria de propuestas
            for r_o in propuesta["_ofrezco"]:
                for r_p in propuesta["_pido"]:
                    agente.propuestas_enviadas[(jugador, r_o, r_p)] = (
                        agente.ronda_actual
                    )

            agente._log(
                "INFO",
                f"Acuerdo pendiente con {jugador}: "
                f"dar={propuesta['_ofrezco']}, pedir={propuesta['_pido']} "
                f"[tx:{propuesta.get('_tx_id')}]",
            )
            envios_realizados += 1
            if envios_realizados >= max_envios:
                agente._log(
                    "INFO",
                    f"Límite de propuestas por ronda alcanzado ({max_envios})",
                )
                break

        time.sleep(agente.pausa_entre_acciones)
