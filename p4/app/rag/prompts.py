"""Prompts del modo RAG."""

from __future__ import annotations

from p4.app.rag.context_builder import RagContext


def build_system_prompt() -> str:
    """Prompt de sistema con reglas estrictas de grounding y citación."""

    return (
        "Eres un asistente de recuperación de información sobre Don Quijote de la Mancha. "
        "Responde únicamente con base en el contexto proporcionado. "
        "No inventes hechos no presentes en las fuentes. "
        "Si la información es insuficiente, dilo explícitamente. "
        "Cita las fuentes utilizadas mediante los identificadores [F1], [F2], etc. "
        "Cuando sea posible, apoya cada afirmación relevante con al menos una referencia. "
        "Si hay conflicto entre fuentes, indícalo. "
        "No menciones acceso a información externa al contexto. "
        "Salvo petición explícita del usuario, responde en español. "
        "Devuelve solo JSON válido, sin markdown ni texto extra."
    )


def build_user_prompt(query: str, context: RagContext) -> str:
    """Prompt de usuario estructurado con consulta, contexto y formato de salida."""

    available_sources = ", ".join(source.source_id for source in context.sources) or "-"
    return (
        f"Consulta del usuario:\n{query}\n\n"
        f"Fuentes disponibles:\n{available_sources}\n\n"
        "Contexto recuperado:\n"
        f"{context.prompt_context}\n\n"
        "Formato de salida requerido:\n"
        "{"
        '"answer": "respuesta breve pero útil, con citas [F1] cuando proceda", '
        '"used_sources": ["F1", "F2"], '
        '"insufficient_evidence": false'
        "}\n\n"
        "Reglas adicionales:\n"
        "- answer debe ser autosuficiente y no inventar nada fuera del contexto.\n"
        "- used_sources solo puede contener identificadores presentes en las fuentes disponibles.\n"
        "- Si no puedes responder con suficiente evidencia, indícalo en answer y marca insufficient_evidence=true.\n"
        "- No devuelvas claves extra.\n"
        "- No envuelvas el JSON en bloques de código."
    )
