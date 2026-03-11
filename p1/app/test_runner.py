#!/usr/bin/env python3
"""
Orquestador de bots — lanza N bots negociadores en paralelo.

Cada bot es un subproceso independiente de main.py con alias único.
Usa **click** para la CLI y **rich** para la salida coloreada.

Uso:
    python test_runner.py                           # 3 bots por defecto
    python test_runner.py -n 5                      # 5 bots
    python test_runner.py -n 2 -m qwen3:8b -d       # 2 bots, debug
    python test_runner.py -n 4 --prefijo Agent      # Agent_1 … Agent_4
    python test_runner.py -n 3 --consola            # salida coloreada en terminal

Cierre limpio con Ctrl+C.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from app.pln_bot.core.config import (
    MODELO_DEFAULT,
    MAX_RONDAS,
    PAUSA_ENTRE_RONDAS,
    NUM_BOTS_DEFAULT,
    PREFIJO_BOTS,
    modelo_soporta_tools,
)

# ── Configuración ────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
LOGS_DIR = APP_DIR / "logs"
MAIN_SCRIPT = APP_DIR / "main.py"

# Estilos rich para cada bot (se repiten si hay más de 8)
ESTILOS = [
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "red",
    "bright_white",
    "dark_orange",
]

console = Console()


# ── Utilidades ───────────────────────────────────────────────────────────


def construir_comando(
    alias: str,
    modelo: str,
    max_rondas: int,
    pausa: int,
    debug: bool,
) -> list[str]:
    """Construye la lista de argumentos para lanzar un bot."""
    cmd = [
        sys.executable,
        str(MAIN_SCRIPT),
        "--alias",
        alias,
        "--modelo",
        modelo,
        "--max-rondas",
        str(max_rondas),
        "--pausa",
        str(pausa),
    ]
    if debug:
        cmd.append("--debug")
    return cmd


def lanzar_modo_logs(
    aliases: list[str],
    modelo: str,
    max_rondas: int,
    pausa: int,
    debug: bool,
):
    """Lanza bots redirigiendo salida a archivos de log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    procesos: list[tuple[str, subprocess.Popen, object]] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"  # Forzar UTF-8 en Windows

    for alias in aliases:
        log_path = LOGS_DIR / f"{alias}_{timestamp}.log"
        log_file = open(log_path, "w", encoding="utf-8")

        cmd = construir_comando(alias, modelo, max_rondas, pausa, debug)
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(MAIN_SCRIPT.parent),
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        procesos.append((alias, proc, log_file))
        console.print(f"  [green]✅ {alias}[/]  (PID {proc.pid})  →  {log_path}")

    return procesos


def lanzar_modo_consola(
    aliases: list[str],
    modelo: str,
    max_rondas: int,
    pausa: int,
    debug: bool,
):
    """Lanza bots con salida coloreada en terminal (usa rich)."""
    procesos: list[tuple[str, subprocess.Popen, None]] = []
    hilos: list[threading.Thread] = []

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"  # Forzar UTF-8 en Windows

    def lector(alias: str, stream, estilo: str):
        prefijo = f"[{estilo}][{alias}][/{estilo}] "
        try:
            for linea in iter(stream.readline, ""):
                if linea:
                    console.print(f"{prefijo}{linea.rstrip()}")
        except ValueError:
            pass
        finally:
            stream.close()

    for i, alias in enumerate(aliases):
        estilo = ESTILOS[i % len(ESTILOS)]
        cmd = construir_comando(alias, modelo, max_rondas, pausa, debug)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=str(MAIN_SCRIPT.parent),
            env=env,
            preexec_fn=os.setsid if sys.platform != "win32" else None,
        )
        procesos.append((alias, proc, None))
        console.print(f"  [{estilo}]✅ {alias}[/{estilo}]  (PID {proc.pid})")

        t = threading.Thread(
            target=lector, args=(alias, proc.stdout, estilo), daemon=True
        )
        t.start()
        hilos.append(t)

    return procesos


def matar_procesos(procesos: list):
    """Envía SIGTERM/SIGKILL a todos los procesos hijos."""
    for alias, proc, log_file in procesos:
        if proc.poll() is None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except (ProcessLookupError, OSError):
                pass

    time.sleep(1)

    for alias, proc, log_file in procesos:
        if proc.poll() is None:
            try:
                if sys.platform != "win32":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass
        if log_file and not log_file.closed:
            log_file.close()


def esperar_procesos(procesos: list):
    """Espera a que todos los procesos terminen normalmente."""
    for alias, proc, log_file in procesos:
        proc.wait()
        if log_file and not log_file.closed:
            log_file.close()


def _tabla_resumen(procesos: list) -> Table:
    """Genera la tabla de resumen final con rich."""
    table = Table(title="📊 Resumen Final", border_style="bright_blue")
    table.add_column("Bot", style="bold")
    table.add_column("PID", justify="right")
    table.add_column("Estado")

    for alias, proc, _ in procesos:
        rc = proc.returncode
        if rc == 0:
            estado = "[green]✅ OK[/]"
        elif rc is None:
            estado = "[red]🛑 forzado[/]"
        else:
            estado = f"[yellow]⚠️  código {rc}[/]"
        table.add_row(alias, str(proc.pid), estado)

    return table


# =========================================================================
# CLI  (click)
# =========================================================================


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-n",
    "num_bots",
    default=NUM_BOTS_DEFAULT,
    show_default=True,
    help="Número de bots a lanzar.",
)
@click.option(
    "--prefijo",
    default=PREFIJO_BOTS,
    show_default=True,
    help="Prefijo para los nombres de los bots.",
)
@click.option(
    "-m", "--modelo", default=MODELO_DEFAULT, show_default=True, help="Modelo de IA."
)
@click.option(
    "-d",
    "--debug/--no-debug",
    default=True,
    show_default=True,
    help="Activar/desactivar modo debug.",
)
@click.option(
    "-r",
    "--max-rondas",
    default=MAX_RONDAS,
    show_default=True,
    help="Rondas máximas por bot.",
)
@click.option(
    "-p",
    "--pausa",
    default=PAUSA_ENTRE_RONDAS,
    show_default=True,
    help="Pausa entre rondas (segundos).",
)
@click.option(
    "--consola",
    is_flag=True,
    default=False,
    help="Mostrar salida coloreada en terminal en vez de logs a archivos.",
)
def main(num_bots, prefijo, modelo, debug, max_rondas, pausa, consola):
    """🚀 Orquestador de bots negociadores — lanza N bots en paralelo."""
    if not modelo_soporta_tools(modelo):
        console.print(
            f"[red]❌ Modelo '{modelo}' no soportado.[/] "
            "Este proyecto usa tools y requiere un modelo qwen*."
        )
        sys.exit(2)

    aliases = [f"{prefijo}_{i}" for i in range(1, num_bots + 1)]

    console.print(
        Panel.fit(
            f"[bold]🚀 ORQUESTADOR DE BOTS NEGOCIADORES[/bold]\n\n"
            f"  Bots:       [cyan]{num_bots}[/]  ({', '.join(aliases)})\n"
            f"  Modelo:     [cyan]{modelo}[/]\n"
            f"  Debug:      [{'green' if debug else 'dim'}]{'SÍ' if debug else 'NO'}[/]\n"
            f"  Max rondas: [cyan]{max_rondas}[/]\n"
            f"  Pausa:      [cyan]{pausa}s[/]\n"
            f"  Salida:     [cyan]{'consola' if consola else f'archivos en {LOGS_DIR}/'}[/]",
            border_style="bright_blue",
        )
    )

    console.print("\n[bold]🔧 Lanzando bots…[/]\n")

    if consola:
        procesos = lanzar_modo_consola(aliases, modelo, max_rondas, pausa, debug)
    else:
        procesos = lanzar_modo_logs(aliases, modelo, max_rondas, pausa, debug)

    console.rule()
    console.print("[green]✅ Todos los bots lanzados. Pulsa Ctrl+C para detenerlos.[/]")
    console.rule()

    # ── Señales para cierre limpio ───────────────────────────────────────
    detenido = False

    def handler_sigint(signum, frame):
        nonlocal detenido
        if detenido:
            return
        detenido = True
        console.print("\n\n[yellow]⏹️  Ctrl+C recibido — deteniendo todos los bots…[/]")
        matar_procesos(procesos)
        console.print(_tabla_resumen(procesos))
        sys.exit(0)

    signal.signal(signal.SIGINT, handler_sigint)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handler_sigint)

    # ── Esperar ──────────────────────────────────────────────────────────
    try:
        esperar_procesos(procesos)
    except KeyboardInterrupt:
        handler_sigint(None, None)

    console.print("\n[bold green]🏁 Todos los bots han terminado.[/]")
    if not consola:
        console.print(f"\n[dim]📂 Logs disponibles en: {LOGS_DIR}/[/]")
    console.print(_tabla_resumen(procesos))


if __name__ == "__main__":
    main()
