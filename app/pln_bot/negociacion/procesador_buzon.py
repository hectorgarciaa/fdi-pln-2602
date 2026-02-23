"""
Procesamiento del buzón de cartas para el agente negociador.
"""

from typing import Dict

from .gestor_acuerdos import registrar_acuerdo_pendiente, responder_aceptacion
from .utilidades_mensajes import (
    es_aceptacion_simple,
    es_carta_sistema,
    es_mensaje_corto_sin_propuesta,
    es_rechazo_simple,
    extraer_recursos_mencionados,
    extraer_tx_id,
    registrar_rechazo,
    registrar_rechazo_propio,
)
from .politica_negociacion import decidir_aceptar_programatico
from .constructor_propuestas import generar_contraoferta, generar_propuesta_adaptada


def procesar_buzon(agente, necesidades: Dict, excedentes: Dict) -> int:
    """Procesa todas las cartas del buzón usando IA para lenguaje natural."""
    buzon = agente.info_actual.get("Buzon", {}) if agente.info_actual else {}
    intercambios = 0
    cartas_procesadas = []

    for uid, carta in buzon.items():
        carta_id = carta.get("id", uid)
        if carta_id in agente.cartas_vistas:
            cartas_procesadas.append(uid)
            continue
        agente.cartas_vistas.add(carta_id)

        remitente = carta.get("remi", "Desconocido")
        mensaje = carta.get("cuerpo", "")
        asunto = carta.get("asunto", "")

        agente._log(
            "RECEPCION",
            f"Carta de {remitente}",
            {"asunto": asunto, "mensaje": mensaje[:150]},
        )

        # ── Filtro 0: cartas del Sistema → ignorar ──
        if es_carta_sistema(remitente, mensaje):
            agente._log("INFO", f"Carta del sistema de '{remitente}' — ignorada")
            cartas_procesadas.append(uid)
            continue

        # ── Filtro 1: rechazos simples → extraer contexto si lo hay ──
        if es_rechazo_simple(mensaje, asunto):
            registrar_rechazo(agente, remitente, asunto)

            # Intentar extraer qué recursos quiere el otro jugador
            recursos_mencionados = extraer_recursos_mencionados(mensaje)
            # Filtrar: solo los que nosotros tenemos de sobra
            exc_disp = agente._excedentes_disponibles(excedentes)
            recursos_que_podemos_dar = [
                r for r in recursos_mencionados if r in exc_disp and exc_disp[r] > 0
            ]

            if recursos_que_podemos_dar and necesidades:
                agente._log(
                    "INFO",
                    f"Rechazo de {remitente} — detectados recursos que quiere: "
                    f"{recursos_que_podemos_dar} (tenemos excedentes)",
                )
                oro_actual = (
                    agente.info_actual.get("Recursos", {}).get("oro", 0)
                    if agente.info_actual
                    else 0
                )
                propuesta = generar_propuesta_adaptada(
                    agente,
                    remitente,
                    recursos_que_podemos_dar,
                    necesidades,
                    excedentes,
                    oro_actual,
                )
                if propuesta:
                    agente._log(
                        "DECISION",
                        f"PROPUESTA ADAPTADA a {remitente}: "
                        f"dar={propuesta['_ofrezco']}, pedir={propuesta['_pido']} "
                        f"[tx:{propuesta.get('_tx_id')}]",
                    )
                    if agente._enviar_carta(
                        remitente, propuesta["asunto"], propuesta["cuerpo"]
                    ):
                        registrar_acuerdo_pendiente(
                            agente,
                            remitente,
                            propuesta["_ofrezco"],
                            propuesta["_pido"],
                            propuesta["_tx_id"],
                        )
                        for r_o in propuesta["_ofrezco"]:
                            for r_p in propuesta["_pido"]:
                                agente.propuestas_enviadas[(remitente, r_o, r_p)] = (
                                    agente.ronda_actual
                                )
                else:
                    agente._log(
                        "INFO",
                        f"Rechazo de {remitente} — no se pudo generar propuesta adaptada",
                    )
            else:
                agente._log(
                    "INFO",
                    f"Rechazo de {remitente} — ignorado (sin contexto aprovechable)",
                )

            cartas_procesadas.append(uid)
            continue

        # ── Filtro 2: mensajes muy cortos sin propuesta ──
        if es_mensaje_corto_sin_propuesta(mensaje):
            agente._log(
                "INFO", f"Mensaje corto de {remitente} sin propuesta — ignorado"
            )
            cartas_procesadas.append(uid)
            continue

        # ── Filtro 3: aceptaciones textuales → sin IA ──
        if es_aceptacion_simple(mensaje, asunto):
            agente._log(
                "ANALISIS",
                f"{remitente} ACEPTA intercambio (detectado por texto, sin IA)",
            )
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            cartas_procesadas.append(uid)
            continue

        # ── Análisis unificado (1 sola llamada IA) ──
        r = agente._analizar_mensaje(remitente, mensaje)

        # ── ¿Aceptación? → responder ──
        if r.es_aceptacion:
            agente._log("ANALISIS", f"{remitente} ACEPTA intercambio")
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            cartas_procesadas.append(uid)
            continue

        # ── Decisión programática sobre la propuesta ──
        aceptar, razon = decidir_aceptar_programatico(
            r.ofrecen, r.piden, necesidades, excedentes
        )

        agente._log(
            "ANALISIS",
            f"Carta de {remitente} analizada",
            {
                "ofrecen": r.ofrecen,
                "piden": r.piden,
                "aceptar": aceptar,
                "razon": razon,
            },
        )

        # ── Decisión: aceptar ──
        if aceptar and r.piden:
            # VALIDAR antes de enviar: ¿me piden cosas que realmente me sobran?
            estado_fresco = agente._actualizar_estado()
            mis_recursos = (
                agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
            )
            excedentes_frescos = (
                estado_fresco.get("excedentes", {}) if estado_fresco else {}
            )

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
                    agente._log(
                        "ALERTA",
                        f"No puedo enviar {cant} {rec} a {remitente}: "
                        f"disponible={disponible}, excedente={excedente_rec}",
                    )
                    envio_valido = False

            if envio_valido and recursos_a_enviar:
                ofrecen_str = ", ".join(f"{c} {r}" for r, c in r.ofrecen.items())
                enviado_str = ", ".join(
                    f"{c} {r}" for r, c in recursos_a_enviar.items()
                )
                tx_id_mensaje = extraer_tx_id(asunto, mensaje)
                tx_tag = f" [tx:{tx_id_mensaje}]" if tx_id_mensaje else ""
                agente._log(
                    "EXITO",
                    f"🤝 ACEPTO oferta de {remitente}: doy {enviado_str} por {ofrecen_str}",
                )
                if agente._enviar_paquete(remitente, recursos_a_enviar):
                    agente._enviar_carta(
                        remitente,
                        f"Re: {asunto}",
                        f"Acepto el trato{tx_tag}. Te he enviado {enviado_str}. "
                        f"Espero recibir {ofrecen_str} de tu parte. "
                        f"Saludos, {agente.alias}",
                    )
                    intercambios += 1
                else:
                    agente._log("ERROR", f"No pude enviar paquete a {remitente}")
            elif not recursos_a_enviar:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} parecía aceptable pero "
                    f"no hay recursos válidos para enviar — rechazada",
                )
            else:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} rechazada: "
                    f"no tengo suficientes excedentes para enviar {r.piden}",
                )

        elif r.ofrecen or r.piden:
            # ── ¿Contraoferta? Si ofrecen algo que necesito pero piden
            #    lo que no tengo → intentar contraoferta con mis excedentes
            me_ofrecen_util = any(rec in necesidades for rec in r.ofrecen)
            if (
                me_ofrecen_util
                and razon == "piden recursos que no me sobran o no tengo suficientes"
            ):
                contra = generar_contraoferta(
                    agente, remitente, r.ofrecen, necesidades, excedentes
                )
                if contra:
                    agente._log(
                        "DECISION",
                        f"CONTRAOFERTA a {remitente}: "
                        f"dar={contra['_ofrezco']}, pedir={contra['_pido']} "
                        f"[tx:{contra.get('_tx_id')}]",
                    )
                    agente._enviar_carta(remitente, contra["asunto"], contra["cuerpo"])
                    # Registrar acuerdo pendiente
                    registrar_acuerdo_pendiente(
                        agente,
                        remitente,
                        contra["_ofrezco"],
                        contra["_pido"],
                        contra["_tx_id"],
                    )
                    cartas_procesadas.append(uid)
                    continue

            agente._log("DECISION", f"RECHAZO oferta de {remitente} ({razon})")
            # Registrar el rechazo que NOSOTROS hacemos para no repetir
            registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
            agente._enviar_carta(
                remitente,
                f"Re: {asunto}",
                f"Gracias por la oferta, {remitente}, pero por ahora "
                f"no me conviene ese intercambio. Saludos, {agente.alias}",
            )
        else:
            agente._log("INFO", f"Mensaje de {remitente} sin propuesta clara: {razon}")

        cartas_procesadas.append(uid)

    for uid in cartas_procesadas:
        agente.api.eliminar_carta(uid)

    return intercambios
