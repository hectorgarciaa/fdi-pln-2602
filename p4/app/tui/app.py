"""Interfaz principal de terminal con Textual."""

from __future__ import annotations

import asyncio

from rich.markdown import Markdown
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Select, Static

from p4.app.errors import QuijoteIRError
from p4.app.models import RagResponse, RagSource, SearchResult
from p4.app.rag import format_rag_answer_markdown
from p4.app.services import QuijoteSearchService


class SearchInput(Input):
    """Input robusto para terminales que envían `space` sin carácter imprimible."""

    async def _on_key(self, event: events.Key) -> None:  # pragma: no cover - UI glue
        if event.key == "space":
            self._restart_blink()
            event.stop()
            selection = self.selection
            if selection.is_empty:
                self.insert_text_at_cursor(" ")
            else:
                self.replace(" ", *selection)
            event.prevent_default()
            return

        await super()._on_key(event)


class QuijoteSearchTUI(App[None]):
    """Aplicación TUI para consultar el corpus en modo clásico o semántico."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: horizontal;
        height: 1fr;
    }

    #sidebar {
        width: 34;
        min-width: 30;
        padding: 1 2;
        border: round $primary;
        background: $surface;
    }

    #main {
        width: 1fr;
        padding: 1;
        layout: vertical;
    }

    #results {
        height: 1fr;
        border: round $accent;
    }

    #answer {
        height: 13;
        border: round $success;
        padding: 1 2;
        overflow-y: auto;
        background: $panel-lighten-1;
    }

    #detail {
        height: 13;
        border: round $secondary;
        padding: 1 2;
        overflow-y: auto;
        background: $panel;
    }

    #status {
        height: auto;
        min-height: 4;
        border: round $warning;
        padding: 1 2;
    }

    .section-title {
        text-style: bold;
        color: $accent;
        padding-top: 1;
    }

    Button {
        width: 1fr;
        margin-top: 1;
    }

    Input, Select {
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+r", "submit_search", "Buscar"),
        Binding("/", "focus_query", "Ir a búsqueda"),
        Binding("c", "set_classical_mode", "Clásica"),
        Binding("s", "set_semantic_mode", "Semántica"),
        Binding("g", "set_rag_mode", "RAG"),
        Binding("q", "quit", "Salir"),
    ]

    def __init__(self, service: QuijoteSearchService) -> None:
        super().__init__()
        self.service = service
        self.current_results: dict[str, SearchResult | RagSource] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("Quijote IR", classes="section-title")
                yield Static("Modo de búsqueda")
                yield Select(
                    options=[
                        ("Clásica (TF-IDF + lemas)", "classical"),
                        ("Semántica (vectores spaCy)", "semantic"),
                        ("RAG (híbrido + Ollama)", "rag"),
                    ],
                    value="classical",
                    id="mode-select",
                )
                yield SearchInput(
                    placeholder="Escribe tu consulta y pulsa Enter", id="query-input"
                )
                yield Button("Buscar", variant="primary", id="search-button")
                yield Static("Recursos", classes="section-title")
                yield Button("Reconstruir chunks", id="build-chunks")
                yield Button("Reconstruir índice clásico", id="build-classical")
                yield Button("Reconstruir embeddings", id="build-semantic")
                yield Static("", id="artifact-status")
            with Vertical(id="main"):
                yield Static(
                    "La respuesta generada aparecerá aquí cuando uses el modo RAG.",
                    id="answer",
                )
                yield DataTable(id="results")
                yield Static(
                    "Selecciona un resultado para ver el detalle.", id="detail"
                )
                yield Static("Listo.", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one("#results", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Ref", "Score", "Parte", "Capítulo", "Chunk")
        await self.refresh_artifact_status()
        self.action_focus_query()

    async def refresh_artifact_status(self) -> None:
        status = await asyncio.to_thread(self.service.describe_artifacts)
        lines = []
        for name, info in status.items():
            lines.append(f"[{'OK' if info['exists'] else '--'}] {name}")
        self.query_one("#artifact-status", Static).update("\n".join(lines))

    async def action_submit_search(self) -> None:
        await self.run_search()

    def action_focus_query(self) -> None:
        self.query_one("#query-input", Input).focus()

    def action_set_classical_mode(self) -> None:
        self.query_one("#mode-select", Select).value = "classical"
        self.action_focus_query()

    def action_set_semantic_mode(self) -> None:
        self.query_one("#mode-select", Select).value = "semantic"
        self.action_focus_query()

    def action_set_rag_mode(self) -> None:
        self.query_one("#mode-select", Select).value = "rag"
        self.action_focus_query()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "search-button":
            await self.run_search()
        elif event.button.id == "build-chunks":
            await self.run_build("chunks")
        elif event.button.id == "build-classical":
            await self.run_build("classical")
        elif event.button.id == "build-semantic":
            await self.run_build("semantic")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "query-input":
            await self.run_search()

    async def on_data_table_row_highlighted(
        self, event: DataTable.RowHighlighted
    ) -> None:
        row_key = str(event.row_key.value)
        result = self.current_results.get(row_key)
        if result is not None:
            self.update_detail(result)

    async def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        row_key = str(event.row_key.value)
        result = self.current_results.get(row_key)
        if result is not None:
            self.update_detail(result)

    async def run_search(self) -> None:
        query = self.query_one("#query-input", Input).value.strip()
        mode = str(self.query_one("#mode-select", Select).value)
        if not query:
            self.set_status("Introduce una consulta antes de buscar.")
            return

        if mode == "rag":
            await self.run_rag(query)
            return

        self.set_status(f"Buscando en modo {mode}...")
        try:
            results = await asyncio.to_thread(
                self.service.search, mode, query, self.service.settings.top_k
            )
        except QuijoteIRError as exc:
            self.clear_results()
            self.set_status(str(exc))
            self.query_one("#detail", Static).update(str(exc))
            return

        self.reset_answer_panel()
        self.populate_results(results)
        if not results:
            self.set_status("No se encontraron resultados.")
            self.query_one("#detail", Static).update(
                "No hubo coincidencias para esta consulta."
            )
            return

        self.set_status(f"{len(results)} resultados cargados.")
        self.update_detail(results[0])

    async def run_rag(self, query: str) -> None:
        self.set_status("Recuperando contexto y generando respuesta RAG...")
        try:
            response = await asyncio.to_thread(self.service.answer_rag, query)
        except QuijoteIRError as exc:
            self.clear_results()
            self.set_status(str(exc))
            self.query_one("#answer", Static).update(str(exc))
            self.query_one("#detail", Static).update(str(exc))
            return

        self.update_answer(response)
        self.populate_results(response.sources)
        if not response.sources:
            self.query_one("#detail", Static).update(
                "No se recuperaron fuentes suficientes para esta consulta."
            )
            self.set_status("Respuesta RAG generada sin fuentes recuperadas.")
            return

        self.update_detail(response.sources[0])
        self.set_status(
            f"Respuesta RAG generada con {len(response.references)} referencias citadas."
        )

    async def run_build(self, target: str) -> None:
        actions = {
            "chunks": self.service.build_chunks,
            "classical": self.service.build_classical_index,
            "semantic": self.service.build_semantic_index,
        }
        self.set_status(f"Reconstruyendo {target}...")
        try:
            await asyncio.to_thread(actions[target])
        except QuijoteIRError as exc:
            self.set_status(str(exc))
            return

        await self.refresh_artifact_status()
        self.set_status(f"Recurso '{target}' reconstruido correctamente.")

    def populate_results(self, results: list[SearchResult | RagSource]) -> None:
        table = self.query_one("#results", DataTable)
        table.clear(columns=False)
        self.current_results = {result.chunk_id: result for result in results}
        for result in results:
            row_label = (
                result.source_id
                if hasattr(result, "source_id") and result.source_id
                else str(result.rank)
            )
            table.add_row(
                row_label,
                f"{result.score:.6f}",
                result.part,
                result.title,
                result.chunk_id,
                key=result.chunk_id,
            )
        if results:
            table.move_cursor(row=0, column=0)

    def clear_results(self) -> None:
        self.current_results = {}
        self.query_one("#results", DataTable).clear(columns=False)

    def update_detail(self, result: SearchResult | RagSource) -> None:
        explanation = result.explanation
        lines = [
            f"## {result.title}",
            f"**Parte:** {result.part}",
        ]
        if isinstance(result, RagSource) and result.source_id:
            lines.append(f"**Fuente:** `[{result.source_id}]`")
        lines.extend(
            [
                f"**Chunk:** `{result.chunk_id}`",
                f"**Score:** `{result.score:.6f}`",
                f"**Rango de párrafos:** {result.metadata.get('paragraph_span', '-')}",
                f"**Tamaño del chunk:** {result.metadata.get('chunk_word_count', '-')} palabras",
                f"**Modo:** {explanation.get('mode', '-')}",
            ]
        )

        if isinstance(result, RagSource):
            retrieval_modes = ", ".join(explanation.get("retrieval_modes", [])) or "-"
            lines.extend(
                [
                    f"**Recuperadores:** {retrieval_modes}",
                    f"**Rango clásica:** {explanation.get('classical_rank', '-')}",
                    f"**Rango semántica:** {explanation.get('semantic_rank', '-')}",
                ]
            )
        else:
            matched_surface = (
                ", ".join(explanation.get("matched_surface_terms", [])) or "-"
            )
            matched_lemma = ", ".join(explanation.get("matched_lemma_terms", [])) or "-"
            lines.extend(
                [
                    f"**Términos exactos:** {matched_surface}",
                    f"**Lemas coincidentes:** {matched_lemma}",
                ]
            )

        lines.extend(["", result.fragment])
        markdown = Markdown("\n".join(lines))
        self.query_one("#detail", Static).update(markdown)

    def update_answer(self, response: RagResponse) -> None:
        self.query_one("#answer", Static).update(
            Markdown(format_rag_answer_markdown(response))
        )

    def reset_answer_panel(self) -> None:
        self.query_one("#answer", Static).update(
            "La respuesta generada aparecerá aquí cuando uses el modo RAG."
        )

    def set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)
