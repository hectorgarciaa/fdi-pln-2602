"""
Interfaz de usuario — punto de entrada principal.

Usa **click** para la CLI y **rich** para la interfaz de terminal.

Uso interactivo (menú):
    python main.py

Uso automático (lanzar bot directo):
    python main.py --alias Bot_1 --modelo qwen3:8b --debug --max-rondas 10
"""

import json

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table

from app.pln_bot.agente.negociador import AgenteNegociador
from app.pln_bot.services.api_client import APIClient
from app.pln_bot.core.config import (
    MODELOS_DISPONIBLES,
    MODELO_DEFAULT,
    MAX_RONDAS,
    PAUSA_ENTRE_RONDAS,
    modelo_soporta_tools,
)

console = Console()


# =========================================================================
# CLI  (click)
# =========================================================================


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--alias",
    "-a",
    default=None,
    help="Nombre del bot.  Si se da, se ejecuta en modo automático (sin menú).",
)
@click.option(
    "--modelo",
    "-m",
    default=MODELO_DEFAULT,
    show_default=True,
    help="Modelo de IA a usar.",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    default=False,
    help="Activar modo debug (muestra cada decisión del agente).",
)
@click.option(
    "--max-rondas",
    "-r",
    default=MAX_RONDAS,
    show_default=True,
    help="Rondas máximas de negociación.",
)
@click.option(
    "--pausa",
    "-p",
    default=PAUSA_ENTRE_RONDAS,
    show_default=True,
    help="Segundos de espera entre rondas.",
)
@click.option("--api-url", default=None, help="URL base de la API del juego.")
def main(alias, modelo, debug, max_rondas, pausa, api_url):
    """🤖 Agente Negociador Autónomo para fdi-pln-butler."""
    if not modelo_soporta_tools(modelo):
        console.print(
            f"[red]❌ Modelo '{modelo}' no soportado.[/] "
            "Este proyecto usa tools y requiere un modelo qwen*."
        )
        return

    # ── Modo automático (CLI) ────────────────────────────────────────────
    if alias:
        console.print(f"[bold cyan]🤖 Iniciando bot '{alias}' en modo automático…[/]")
        _ejecutar_agente(
            alias=alias,
            modelo=modelo,
            debug=debug,
            max_rondas=max_rondas,
            pausa=pausa,
            interactivo=False,
            api_url=api_url,
        )
        return

    # ── Modo interactivo (menú) ──────────────────────────────────────────
    console.print(
        Panel.fit(
            "[bold]🎮 SISTEMA DE NEGOCIACIÓN AUTÓNOMO[/bold]\n\n"
            "El agente negociará automáticamente para:\n"
            "  1️⃣  Conseguir los recursos objetivo\n"
            "  2️⃣  Maximizar el oro vendiendo excedentes\n\n"
            "Activando [bold]DEBUG[/] verás todo lo que hace el agente:\n"
            "  📤 Cartas enviadas   📥 Cartas recibidas\n"
            "  🔍 Análisis          🧠 Decisiones\n"
            "  🔄 Intercambios",
            border_style="bright_blue",
        )
    )

    while True:
        console.print("\n[bold]1.[/] 🤖 INICIAR AGENTE AUTÓNOMO")
        console.print("[bold]2.[/] 📡 Operaciones API (manual)")
        console.print("[bold]0.[/] Salir")

        opcion = Prompt.ask("\nOpción", choices=["0", "1", "2"], default="1")

        if opcion == "1":
            alias = Prompt.ask("\nTu alias para negociar")
            if alias:
                _menu_agente(alias)
        elif opcion == "2":
            _menu_api()
        elif opcion == "0":
            console.print("\n[dim]¡Hasta luego![/]")
            break


# =========================================================================
# MENÚ DEL AGENTE (interactivo)
# =========================================================================


def _menu_agente(alias: str):
    """Configura y lanza el bot en modo interactivo."""
    console.rule("[bold]🤖 Configuración del Agente[/bold]")

    # ── Modelo ───────────────────────────────────────────────────────────
    table = Table(title="Modelos disponibles", show_header=True, border_style="cyan")
    table.add_column("#", style="bold")
    table.add_column("Modelo")
    table.add_column("Descripción")

    for key, (modelo, desc) in MODELOS_DISPONIBLES.items():
        marca = " ← default" if modelo == MODELO_DEFAULT else ""
        table.add_row(key, modelo, f"{desc}{marca}")

    console.print(table)

    opcion_modelo = Prompt.ask(
        "Selecciona modelo", choices=list(MODELOS_DISPONIBLES), default="1"
    )
    modelo = MODELOS_DISPONIBLES[opcion_modelo][0]

    # ── Opciones ─────────────────────────────────────────────────────────
    debug = Confirm.ask("¿Activar modo DEBUG?", default=True)
    max_rondas = IntPrompt.ask("Máximo de rondas", default=MAX_RONDAS)
    pausa = IntPrompt.ask("Segundos entre rondas", default=PAUSA_ENTRE_RONDAS)

    # ── Confirmar ────────────────────────────────────────────────────────
    resumen = Table(
        title="📋 Resumen de Configuración",
        show_header=False,
        border_style="bright_blue",
        padding=(0, 2),
    )
    resumen.add_column("Campo", style="bold")
    resumen.add_column("Valor")
    resumen.add_row("Alias", alias)
    resumen.add_row("Modelo", modelo)
    resumen.add_row("Debug", "✅ ACTIVADO" if debug else "desactivado")
    resumen.add_row("Max rondas", str(max_rondas))
    resumen.add_row("Pausa", f"{pausa}s")

    console.print(resumen)

    if not Confirm.ask("\n¿Iniciar agente?", default=True):
        return

    _ejecutar_agente(
        alias,
        modelo,
        debug,
        max_rondas,
        pausa,
        interactivo=True,
    )


# =========================================================================
# EJECUTAR AGENTE
# =========================================================================


def _ejecutar_agente(
    alias: str,
    modelo: str,
    debug: bool,
    max_rondas: int,
    pausa: int,
    interactivo: bool = False,
    api_url: str = None,
):
    """Crea y ejecuta el agente negociador."""
    try:
        agente = AgenteNegociador(alias, modelo, debug, api_url=api_url)
    except ValueError as e:
        console.print(f"[red]❌ {e}[/]")
        return
    agente.pausa_entre_rondas = pausa

    try:
        agente.ejecutar(max_rondas)
    except KeyboardInterrupt:
        console.print("\n\n[yellow]⏹️ Agente detenido por el usuario[/]")
        agente._mostrar_resumen()

    if not interactivo:
        return

    # ── Post-ejecución ───────────────────────────────────────────────────
    while True:
        console.rule("[bold]📜 Opciones post-ejecución[/bold]")
        console.print("[bold]1.[/] Ver log (loguru)")
        console.print("[bold]2.[/] Continuar ejecución")
        console.print("[bold]0.[/] Salir")

        opcion = Prompt.ask("Opción", choices=["0", "1", "2"], default="0")

        if opcion == "1":
            agente.ver_log()
        elif opcion == "2":
            rondas = IntPrompt.ask("Rondas adicionales", default=5)
            try:
                agente.ejecutar(rondas)
            except KeyboardInterrupt:
                console.print("\n[yellow]⏹️ Detenido[/]")
                agente._mostrar_resumen()
        elif opcion == "0":
            break


# =========================================================================
# MENÚ API MANUAL
# =========================================================================


def _menu_api():
    """Menú para operaciones manuales de la API."""
    api = APIClient()

    while True:
        console.rule("[bold]📡 Operaciones API (manual)[/bold]")
        console.print("[bold]1.[/] Ver mi información")
        console.print("[bold]2.[/] Ver jugadores")
        console.print("[bold]3.[/] Crear alias")
        console.print("[bold]4.[/] Eliminar alias")
        console.print("[bold]5.[/] Enviar carta")
        console.print("[bold]6.[/] Enviar paquete")
        console.print("[bold]7.[/] Eliminar carta")
        console.print("[bold]0.[/] Volver")

        opcion = Prompt.ask("Opción", choices=[str(i) for i in range(8)], default="0")

        if opcion == "1":
            info = api.get_info()
            if info:
                console.print_json(json.dumps(info, ensure_ascii=False))

        elif opcion == "2":
            gente = api.get_gente()
            console.print("\n[bold]👥 Jugadores:[/]")
            for p in gente:
                console.print(f"  - {p}")

        elif opcion == "3":
            nombre = Prompt.ask("Nombre del alias")
            if nombre:
                api.crear_alias(nombre)

        elif opcion == "4":
            nombre = Prompt.ask("Alias a eliminar")
            if nombre:
                api.eliminar_alias(nombre)

        elif opcion == "5":
            remi = Prompt.ask("Remitente (tu alias)")
            dest = Prompt.ask("Destinatario")
            asunto = Prompt.ask("Asunto")
            cuerpo = Prompt.ask("Cuerpo")
            if all([remi, dest, asunto, cuerpo]):
                ok = api.enviar_carta(remi, dest, asunto, cuerpo)
                console.print(
                    "[green]✓ Carta enviada[/]" if ok else "[red]✗ Error al enviar[/]"
                )

        elif opcion == "6":
            dest = Prompt.ask("Destinatario")
            recursos = {}
            console.print("[dim]Recursos (Enter vacío para terminar):[/]")
            while True:
                r = Prompt.ask("  Recurso", default="")
                if not r:
                    break
                c = IntPrompt.ask(f"  Cantidad de {r}", default=1)
                recursos[r] = c
            if recursos:
                ok = api.enviar_paquete(dest, recursos)
                console.print(
                    f"[green]✓ Paquete enviado: {recursos}[/]"
                    if ok
                    else "[red]✗ Error al enviar[/]"
                )

        elif opcion == "7":
            uid = Prompt.ask("UID de la carta")
            if uid:
                api.eliminar_carta(uid)
                console.print(f"[green]✓ Carta {uid} eliminada[/]")

        elif opcion == "0":
            break


if __name__ == "__main__":
    main()
