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
from .constructor_propuestas import (
    generar_propuesta_adaptada,
    nuevo_tx_id,
)


def _normalizar_recursos(recursos: Dict[str, int]) -> Dict[str, int]:
    normalizados: Dict[str, int] = {}
    if not isinstance(recursos, dict):
        return normalizados
    for rec, cant in recursos.items():
        try:
            cantidad = int(cant)
        except (TypeError, ValueError):
            continue
        if cantidad <= 0:
            continue
        normalizados[str(rec).strip().lower()] = cantidad
    return normalizados


def _construir_contraoferta_ia(
    agente, destinatario: str, ofrezco: Dict[str, int], pido: Dict[str, int]
) -> dict | None:
    ofrezco = _normalizar_recursos(ofrezco)
    pido = _normalizar_recursos(pido)
    if not ofrezco or not pido:
        return None

    ofrezco_str = ", ".join(f"{c} {r}" for r, c in ofrezco.items())
    pido_str = ", ".join(f"{c} {r}" for r, c in pido.items())
    tx_id = nuevo_tx_id()

    cuerpo = (
        f"Hola {destinatario}, soy {agente.alias}. "
        f"[tx:{tx_id}] "
        f"Te hago una contrapropuesta: "
        f"yo te doy {ofrezco_str} y tú me das {pido_str}. "
        f"Si aceptas, responde 'acepto el trato'. "
        f"Si no te conviene, responde 'no me conviene'. "
        f"Saludos, {agente.alias}"
    )

    return {
        "asunto": f"Contrapropuesta: [tx:{tx_id}] mi {ofrezco_str} por tu {pido_str}",
        "cuerpo": cuerpo,
        "_ofrezco": ofrezco,
        "_pido": pido,
        "_tx_id": tx_id,
    }


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

        tx_id_mensaje = extraer_tx_id(asunto, mensaje)
        aceptacion_textual = es_aceptacion_simple(mensaje, asunto)

        # Evita procesar en bucle al mismo remitente en una ronda, salvo aceptaciones.
        if (
            remitente in agente.remitentes_gestionados_esta_ronda
            and not aceptacion_textual
            and not tx_id_mensaje
        ):
            agente._log(
                "INFO",
                f"Remitente {remitente} ya gestionado esta ronda — carta ignorada",
            )
            cartas_procesadas.append(uid)
            continue

        # Silencia temporalmente remitentes con ráfagas de cartas.
        silenciado = agente._registrar_carta_recibida(remitente)
        if silenciado and not aceptacion_textual and not tx_id_mensaje:
            agente._log(
                "INFO",
                f"Carta de {remitente} ignorada por cooldown anti-spam",
            )
            cartas_procesadas.append(uid)
            continue

        # ── Filtro 1: aceptaciones textuales → sin IA ──
        if aceptacion_textual:
            agente._log(
                "ANALISIS",
                f"{remitente} ACEPTA intercambio (detectado por texto, sin IA)",
            )
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            cartas_procesadas.append(uid)
            continue

        # ── Filtro 1: rechazos simples → extraer contexto si lo hay ──
        if es_rechazo_simple(mensaje, asunto):
            registrar_rechazo(agente, remitente, asunto)

            # Intentar extraer qué recursos quiere el otro jugador
            exc_disp = agente._excedentes_disponibles(excedentes)
            recursos_mencionados = extraer_recursos_mencionados(
                mensaje, candidatos=exc_disp.keys()
            )
            # Filtrar: solo los que nosotros tenemos de sobra
            recursos_que_podemos_dar = [
                r for r in recursos_mencionados if r in exc_disp and exc_disp[r] > 0
            ]

            if recursos_que_podemos_dar and necesidades:
                agente._log(
                    "INFO",
                    f"Rechazo de {remitente} — detectados recursos que quiere: "
                    f"{recursos_que_podemos_dar} (tenemos excedentes)",
                )

                en_cooldown, restante = agente._en_cooldown_rechazo_adaptado(remitente)
                if en_cooldown:
                    agente._log(
                        "INFO",
                        f"Rechazo de {remitente} detectado, pero la respuesta "
                        f"adaptada está en cooldown ({restante}s)",
                    )
                    cartas_procesadas.append(uid)
                    continue

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
                        agente._registrar_rechazo_adaptado(remitente)
                        agente.remitentes_gestionados_esta_ronda.add(remitente)
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

        # ── Análisis unificado (1 sola llamada IA) ──
        recursos_actuales = (
            agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
        )
        objetivo_actual = (
            agente.info_actual.get("Objetivo", {}) if agente.info_actual else {}
        )
        r = agente._analizar_mensaje(
            remitente=remitente,
            mensaje=mensaje,
            asunto=asunto,
            necesidades=necesidades,
            excedentes=excedentes,
            recursos_actuales=recursos_actuales,
            objetivo=objetivo_actual,
        )

        # ── ¿Aceptación? → responder ──
        if r.es_aceptacion:
            agente._log("ANALISIS", f"{remitente} ACEPTA intercambio")
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            cartas_procesadas.append(uid)
            continue

        # ── Decisión guiada por LLM ──
        razon = r.razon or "sin explicación"
        hay_oferta = bool(r.ofrecen or r.piden)
        decision = r.decision if hay_oferta else "ignorar"

        if hay_oferta and agente._oferta_duplicada_reciente(
            remitente, r.ofrecen, r.piden
        ):
            agente._log(
                "INFO",
                f"Oferta duplicada reciente de {remitente} — ignorada",
                {"ofrecen": r.ofrecen, "piden": r.piden},
            )
            cartas_procesadas.append(uid)
            continue

        agente._log(
            "ANALISIS",
            f"Carta de {remitente} analizada",
            {
                "ofrecen": r.ofrecen,
                "piden": r.piden,
                "decision": decision,
                "razon": razon,
                "contraoferta_ofrezco": r.contraoferta_ofrezco,
                "contraoferta_pido": r.contraoferta_pido,
            },
        )

        # ── Decisión: aceptar ──
        if decision == "aceptar" and r.piden and r.ofrecen:
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
                    agente.remitentes_gestionados_esta_ronda.add(remitente)
                else:
                    agente._log("ERROR", f"No pude enviar paquete a {remitente}")
                    agente.remitentes_gestionados_esta_ronda.add(remitente)
            elif not recursos_a_enviar:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} parecía aceptable pero "
                    f"no hay recursos válidos para enviar — rechazada",
                )
                registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                agente.remitentes_gestionados_esta_ronda.add(remitente)
            else:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} rechazada: "
                    f"no tengo suficientes excedentes para enviar {r.piden}",
                )
                registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                agente.remitentes_gestionados_esta_ronda.add(remitente)

        elif decision == "contraofertar" and hay_oferta:
            contra = _construir_contraoferta_ia(
                agente, remitente, r.contraoferta_ofrezco, r.contraoferta_pido
            )
            if contra is None:
                agente._log(
                    "DECISION",
                    f"Contraoferta de IA inválida para {remitente} ({razon})",
                )
                registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                agente.remitentes_gestionados_esta_ronda.add(remitente)
            else:
                exc_disp = agente._excedentes_disponibles(excedentes)
                envio_valido = all(
                    exc_disp.get(rec, 0) >= cant
                    for rec, cant in contra["_ofrezco"].items()
                )
                if not envio_valido:
                    agente._log(
                        "DECISION",
                        f"Contraoferta de IA rechazada por excedente insuficiente con {remitente}",
                        {"ofrezco": contra["_ofrezco"], "excedentes": exc_disp},
                    )
                    registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                    agente.remitentes_gestionados_esta_ronda.add(remitente)
                else:
                    agente._log(
                        "DECISION",
                        f"CONTRAOFERTA (LLM) a {remitente}: "
                        f"dar={contra['_ofrezco']}, pedir={contra['_pido']} "
                        f"[tx:{contra.get('_tx_id')}]",
                    )
                    if agente._enviar_carta(
                        remitente, contra["asunto"], contra["cuerpo"]
                    ):
                        registrar_acuerdo_pendiente(
                            agente,
                            remitente,
                            contra["_ofrezco"],
                            contra["_pido"],
                            contra["_tx_id"],
                        )
                        for r_o in contra["_ofrezco"]:
                            for r_p in contra["_pido"]:
                                agente.propuestas_enviadas[(remitente, r_o, r_p)] = (
                                    agente.ronda_actual
                                )
                        agente.remitentes_gestionados_esta_ronda.add(remitente)
                    else:
                        agente._log(
                            "ERROR",
                            f"No se pudo enviar contraoferta a {remitente}",
                        )
                        agente.remitentes_gestionados_esta_ronda.add(remitente)

        elif hay_oferta:
            agente._log("DECISION", f"RECHAZO oferta de {remitente} ({razon})")
            registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
            agente._log(
                "INFO",
                f"Oferta de {remitente} no interesante: no se envía respuesta directa",
            )
            agente.remitentes_gestionados_esta_ronda.add(remitente)
        else:
            agente._log("INFO", f"Mensaje de {remitente} sin propuesta clara: {razon}")

        cartas_procesadas.append(uid)

    for uid in cartas_procesadas:
        agente.api.eliminar_carta(uid)

    return intercambios
