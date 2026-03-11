"""
Procesamiento del buzón de cartas para el agente negociador.
"""

from typing import Dict

from ..services.analysis import RespuestaUnificada
from .gestor_acuerdos import registrar_acuerdo_pendiente, responder_aceptacion
from .utilidades_mensajes import (
    es_aceptacion_simple,
    es_carta_sistema,
    es_mensaje_corto_sin_propuesta,
    es_rechazo_simple,
    extraer_oferta_estructurada,
    extraer_recursos_mencionados,
    extraer_tx_id,
    registrar_rechazo,
    registrar_rechazo_propio,
)
from .constructor_propuestas import (
    generar_contraoferta,
    generar_propuesta,
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


def _enviar_rechazo_no_silencioso(
    agente, remitente: str, asunto_original: str, razon: str
) -> None:
    """Envía rechazo explícito para no dejar ofertas sin respuesta."""
    tx_id = extraer_tx_id(asunto_original)
    tx_tag = f" [tx:{tx_id}]" if tx_id else ""
    asunto = f"Re: {asunto_original}" if asunto_original else "Rechazo de propuesta"
    cuerpo = f"No me conviene{tx_tag}. Motivo: {razon}. Saludos, {agente.alias}"
    agente._enviar_carta(remitente, asunto, cuerpo)


def _registrar_propuesta_en_memoria(agente, remitente: str, propuesta: Dict) -> None:
    """Guarda tracking de propuesta enviada desde cualquier flujo."""
    registrar_acuerdo_pendiente(
        agente,
        remitente,
        propuesta["_ofrezco"],
        propuesta["_pido"],
        propuesta["_tx_id"],
    )
    for r_o in propuesta["_ofrezco"]:
        for r_p in propuesta["_pido"]:
            agente.propuestas_enviadas[(remitente, r_o, r_p)] = agente.ronda_actual


def _responder_contraoferta_o_rechazo(
    agente,
    remitente: str,
    asunto: str,
    razon: str,
    ofrecen: Dict[str, int],
    piden: Dict[str, int],
    necesidades: Dict[str, int],
    excedentes: Dict[str, int],
) -> None:
    """Intenta contraofertar antes de enviar rechazo definitivo."""
    contra = generar_contraoferta(agente, remitente, ofrecen, necesidades, excedentes)
    if not contra:
        oro_actual = (
            agente.info_actual.get("Recursos", {}).get("oro", 0)
            if agente.info_actual
            else 0
        )
        contra = generar_propuesta(
            agente, remitente, necesidades, excedentes, oro_actual
        )

    if contra:
        agente._log(
            "DECISION",
            f"RECHAZO + CONTRAOFERTA a {remitente}: "
            f"dar={contra['_ofrezco']}, pedir={contra['_pido']} "
            f"[tx:{contra.get('_tx_id')}]",
            {"motivo_rechazo_original": razon},
        )
        if agente._enviar_carta(remitente, contra["asunto"], contra["cuerpo"]):
            _registrar_propuesta_en_memoria(agente, remitente, contra)
            return
        agente._log("ERROR", f"No se pudo enviar contraoferta a {remitente}")

    _enviar_rechazo_no_silencioso(agente, remitente, asunto, razon)


def _decision_rapida_oferta(
    agente,
    ofrecen: Dict[str, int],
    piden: Dict[str, int],
    necesidades: Dict[str, int],
    excedentes: Dict[str, int],
) -> RespuestaUnificada:
    """Decisión rápida sin LLM para ofertas en formato plantilla."""
    if not ofrecen or not piden:
        return RespuestaUnificada(
            decision="ignorar",
            razon="oferta estructurada incompleta",
            ofrecen=ofrecen,
            piden=piden,
        )

    exc_disp = agente._excedentes_disponibles(excedentes)
    puede_entregar = all(exc_disp.get(rec, 0) >= cant for rec, cant in piden.items())

    valor_recibo = sum(
        min(int(cant), int(necesidades.get(rec, 0))) for rec, cant in ofrecen.items()
    )
    coste_entrego = sum(
        min(int(cant), int(necesidades.get(rec, 0))) for rec, cant in piden.items()
    )

    if puede_entregar and valor_recibo > 0 and valor_recibo >= coste_entrego:
        return RespuestaUnificada(
            ofrecen=ofrecen,
            piden=piden,
            decision="aceptar",
            razon="analisis_rapido: cubre necesidades y no empeora objetivo",
        )

    return RespuestaUnificada(
        ofrecen=ofrecen,
        piden=piden,
        decision="rechazar",
        razon="analisis_rapido: oferta no favorable o no entregable con excedentes",
    )


def procesar_buzon(agente, necesidades: Dict, excedentes: Dict) -> int:
    """Procesa todas las cartas del buzón usando IA para lenguaje natural."""
    buzon = agente.info_actual.get("Buzon", {}) if agente.info_actual else {}
    # Protección adicional: si Buzon es None, convertir a dict vacío
    if buzon is None:
        buzon = {}
    intercambios = 0
    analisis_llm_realizados = 0
    max_analisis_llm = max(1, int(getattr(agente, "max_analisis_llm_por_ronda", 3)))

    def _cerrar_carta(uid_local: str, carta_id_local: str, motivo: str) -> None:
        agente.cartas_vistas.add(carta_id_local)
        if not agente.api.eliminar_carta(uid_local):
            agente._log(
                "ALERTA",
                f"No se pudo eliminar carta {uid_local} tras {motivo}",
            )

    for uid, carta in buzon.items():
        carta_id = carta.get("id", uid)
        if carta_id in agente.cartas_vistas:
            if not agente.api.eliminar_carta(uid):
                agente._log(
                    "ALERTA",
                    f"No se pudo limpiar carta ya vista {uid} del buzón",
                )
            continue

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
            _cerrar_carta(uid, carta_id, "ignorar carta de sistema")
            continue

        aceptacion_textual = es_aceptacion_simple(mensaje, asunto)

        # ── Filtro 1: aceptaciones textuales → sin IA ──
        if aceptacion_textual:
            agente._log(
                "ANALISIS",
                f"{remitente} ACEPTA intercambio (detectado por texto, sin IA)",
            )
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            _cerrar_carta(uid, carta_id, "procesar aceptación textual")
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

            _cerrar_carta(uid, carta_id, "procesar rechazo simple")
            continue

        # ── Filtro 2: mensajes muy cortos sin propuesta ──
        if es_mensaje_corto_sin_propuesta(mensaje):
            agente._log(
                "INFO", f"Mensaje corto de {remitente} sin propuesta — ignorado"
            )
            _cerrar_carta(uid, carta_id, "ignorar mensaje corto")
            continue

        # ── Oferta estructurada: por defecto también pasa por pydantic_ai ──
        ofrecen_struct, piden_struct = extraer_oferta_estructurada(asunto, mensaje)
        carta_estructurada = bool(ofrecen_struct and piden_struct)
        forzar_llm_estructurada = bool(
            getattr(agente, "forzar_llm_en_ofertas_estructuradas", True)
        )
        if carta_estructurada and not forzar_llm_estructurada:
            r = _decision_rapida_oferta(
                agente,
                ofrecen=ofrecen_struct,
                piden=piden_struct,
                necesidades=necesidades,
                excedentes=excedentes,
            )
            agente._log(
                "ANALISIS",
                f"Carta estructurada de {remitente} analizada sin LLM",
                {
                    "ofrecen": r.ofrecen,
                    "piden": r.piden,
                    "decision": r.decision,
                    "razon": r.razon,
                },
            )
        else:
            # ── Análisis unificado (1 sola llamada IA) ──
            if analisis_llm_realizados >= max_analisis_llm:
                agente._log(
                    "INFO",
                    f"Presupuesto LLM agotado ({max_analisis_llm}/ronda). "
                    f"Carta de {remitente} diferida.",
                )
                continue

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
                modo_analisis="estructurado" if carta_estructurada else "normal",
            )
            analisis_llm_realizados += 1
            if carta_estructurada:
                agente._log(
                    "ANALISIS",
                    f"Carta estructurada de {remitente} analizada con pydantic_ai",
                )

        # Si una carta estructurada vuelve vacía desde LLM, aplicar fallback local
        # para no perder ofertas claras por timeout/token-limit del modelo.
        if carta_estructurada and not r.es_aceptacion and not (r.ofrecen or r.piden):
            r_fallback = _decision_rapida_oferta(
                agente,
                ofrecen=ofrecen_struct,
                piden=piden_struct,
                necesidades=necesidades,
                excedentes=excedentes,
            )
            agente._log(
                "ANALISIS",
                f"Fallback local para carta estructurada de {remitente} tras salida vacía de LLM",
                {
                    "decision": r_fallback.decision,
                    "razon": r_fallback.razon,
                    "ofrecen": r_fallback.ofrecen,
                    "piden": r_fallback.piden,
                },
            )
            r = r_fallback

        # ── ¿Aceptación? → responder ──
        if r.es_aceptacion:
            agente._log("ANALISIS", f"{remitente} ACEPTA intercambio")
            if responder_aceptacion(agente, remitente, mensaje, asunto):
                intercambios += 1
            _cerrar_carta(uid, carta_id, "procesar aceptación LLM")
            continue

        # ── Decisión guiada por LLM ──
        razon = r.razon or "sin explicación"
        hay_oferta = bool(r.ofrecen or r.piden)
        decision = r.decision if hay_oferta else "ignorar"

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
                else:
                    agente._log("ERROR", f"No pude enviar paquete a {remitente}")
            elif not recursos_a_enviar:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} parecía aceptable pero "
                    f"no hay recursos válidos para enviar — rechazada",
                )
                registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                _responder_contraoferta_o_rechazo(
                    agente,
                    remitente,
                    asunto,
                    "no tengo recursos válidos para cumplir lo pedido",
                    r.ofrecen,
                    r.piden,
                    necesidades,
                    excedentes,
                )
            else:
                agente._log(
                    "DECISION",
                    f"Oferta de {remitente} rechazada: "
                    f"no tengo suficientes excedentes para enviar {r.piden}",
                )
                registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
                _responder_contraoferta_o_rechazo(
                    agente,
                    remitente,
                    asunto,
                    f"no tengo suficientes excedentes para enviar {r.piden}",
                    r.ofrecen,
                    r.piden,
                    necesidades,
                    excedentes,
                )

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
                _responder_contraoferta_o_rechazo(
                    agente,
                    remitente,
                    asunto,
                    f"no puedo contraofertar ({razon})",
                    r.ofrecen,
                    r.piden,
                    necesidades,
                    excedentes,
                )
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
                    _responder_contraoferta_o_rechazo(
                        agente,
                        remitente,
                        asunto,
                        "no tengo excedente suficiente para contraofertar",
                        r.ofrecen,
                        r.piden,
                        necesidades,
                        excedentes,
                    )
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
                    else:
                        agente._log(
                            "ERROR",
                            f"No se pudo enviar contraoferta a {remitente}",
                        )

        elif hay_oferta:
            agente._log("DECISION", f"RECHAZO oferta de {remitente} ({razon})")
            registrar_rechazo_propio(agente, remitente, r.ofrecen, r.piden)
            _responder_contraoferta_o_rechazo(
                agente,
                remitente,
                asunto,
                razon,
                r.ofrecen,
                r.piden,
                necesidades,
                excedentes,
            )
            agente._log(
                "INFO",
                f"Oferta de {remitente} rechazada con intento de contraoferta",
            )
        else:
            agente._log("INFO", f"Mensaje de {remitente} sin propuesta clara: {razon}")

        _cerrar_carta(uid, carta_id, "procesar carta general")

    return intercambios
