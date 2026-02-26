"""
Ejecución de una ronda completa del agente negociador.
"""

import time

from ..negociacion.gestor_acuerdos import limpiar_cache_tx, mover_a_expirados_por_tx
from ..negociacion.procesador_buzon import procesar_buzon
from ..negociacion.enviador_propuestas import enviar_propuestas


def ejecutar_ronda(agente, console) -> bool:
    """Ejecuta una ronda completa de negociación."""
    console.rule(f"[bold]📍 RONDA — Modo: {agente.modo.value}[/bold]")

    agente.ronda_actual += 1
    inicio_ronda = time.time()

    # Evita bloquear recursos durante demasiadas rondas por acuerdos sin respuesta.
    ttl_base = max(1, int(getattr(agente, "ACUERDO_TTL_SEGUNDOS", 300)))
    pausa_rondas = max(1, int(getattr(agente, "pausa_entre_rondas", 30)))
    ttl_dinamico = max(20, pausa_rondas * 2)
    ttl_activo = min(ttl_base, ttl_dinamico)

    ahora = time.time()
    for persona in list(agente.acuerdos_pendientes.keys()):
        acuerdos_activos = []
        acuerdos_expirados = []
        for acuerdo in agente.acuerdos_pendientes[persona]:
            if ahora - acuerdo.get("timestamp", 0) < ttl_activo:
                acuerdos_activos.append(acuerdo)
            else:
                acuerdos_expirados.append(acuerdo)
                mover_a_expirados_por_tx(agente, persona, acuerdo, ahora)

        if acuerdos_expirados:
            agente._log(
                "INFO",
                f"Moviendo {len(acuerdos_expirados)} acuerdo(s) de {persona} "
                f"a caché de expirados (TTL activo={ttl_activo}s, "
                f"gracia={agente.ACUERDO_GRACIA_TTL_SEGUNDOS}s)",
            )
        if acuerdos_activos:
            agente.acuerdos_pendientes[persona] = acuerdos_activos
        else:
            del agente.acuerdos_pendientes[persona]

    limpiar_cache_tx(agente, ahora)

    claves_viejas = [
        k
        for k, ronda in agente.propuestas_enviadas.items()
        if agente.ronda_actual - ronda > 2
    ]
    for k in claves_viejas:
        del agente.propuestas_enviadas[k]

    estado = agente._actualizar_estado()
    if not estado:
        agente._log("ERROR", "No se pudo conectar a la API")
        return False

    necesidades = estado["necesidades"]
    excedentes = estado["excedentes"]
    oro = estado["oro"]
    objetivo_completado = estado["objetivo_completado"]

    recursos_actuales = (
        agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
    )
    if recursos_actuales:
        recursos_str = ", ".join(
            f"{rec}: {cant}" for rec, cant in sorted(recursos_actuales.items())
        )
        agente._log("INFO", f"📦 INVENTARIO: {recursos_str}")

    if necesidades:
        nec_str = ", ".join(f"{cant} {rec}" for rec, cant in necesidades.items())
        agente._log("INFO", f"🎯 NECESITAMOS: {nec_str}")

    if excedentes:
        exc_str = ", ".join(f"{cant} {rec}" for rec, cant in excedentes.items())
        agente._log("INFO", f"💰 EXCEDENTES: {exc_str}")

    agente._log(
        "INFO",
        f"🪙 ORO: {oro} | Objetivo: {'✅ COMPLETADO' if objetivo_completado else '⏳ En progreso'}",
    )

    modo_enum = type(agente.modo)
    if objetivo_completado and agente.modo == modo_enum.CONSEGUIR_OBJETIVO:
        agente._log("EXITO", "¡OBJETIVO COMPLETADO! → modo MAXIMIZAR ORO")
        agente.modo = modo_enum.MAXIMIZAR_ORO

    if agente.modo == modo_enum.MAXIMIZAR_ORO and not excedentes:
        agente._log("EXITO", "No hay más excedentes para vender")
        agente.modo = modo_enum.COMPLETADO
        return True

    agente._log("INFO", "Procesando buzón…")
    intercambios = procesar_buzon(agente, necesidades, excedentes)

    agente._procesar_paquetes_recibidos()

    if intercambios > 0:
        agente._log(
            "EXITO", f"✅ {intercambios} intercambio(s) completado(s) esta ronda"
        )
        estado = agente._actualizar_estado()
        recursos_actuales = (
            agente.info_actual.get("Recursos", {}) if agente.info_actual else {}
        )
        if recursos_actuales:
            recursos_str = ", ".join(
                f"{rec}: {cant}" for rec, cant in sorted(recursos_actuales.items())
            )
            agente._log("INFO", f"📦 INVENTARIO ACTUALIZADO: {recursos_str}")
        necesidades = estado["necesidades"]
        excedentes = estado["excedentes"]

    if necesidades or (agente.modo == modo_enum.MAXIMIZAR_ORO and excedentes):
        agente._log("INFO", "Enviando propuestas…")
        enviar_propuestas(agente, necesidades, excedentes, estado["oro"])

    agente.contactados_esta_ronda = []

    duracion_ronda = time.time() - inicio_ronda
    espera_minima = agente.pausa_entre_rondas * 0.5
    if duracion_ronda < espera_minima:
        esperar = espera_minima - duracion_ronda
        agente._log(
            "INFO",
            f"Ronda rápida ({duracion_ronda:.1f}s) — esperando {esperar:.0f}s para recibir respuestas",
        )
        time.sleep(esperar)

    return estado["objetivo_completado"] and agente.modo == modo_enum.COMPLETADO
