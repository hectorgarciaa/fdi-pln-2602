"""
Sistema de Information Retrieval para Don Quijote de la Mancha
==============================================================
Backend Flask + Motor TF-IDF/BM25 con N-gramas (1,2).

Dependencias:
    pip install beautifulsoup4 scikit-learn spacy rank-bm25 flask flask-cors
    python -m spacy download es_core_news_sm

Uso:
    python quijote_ir.py          # Lanza el servidor web en http://localhost:5000
    python quijote_ir.py --cli    # Modo interactivo por consola
"""

import sys
from rank_bm25 import BM25Okapi



def main_cli(buscador: BuscadorQuijote):
    """Modo interactivo por consola."""
    print("\n" + "="*70)
    print("🔎 MODO INTERACTIVO — Escribe tu consulta (o 'salir')")
    print("   Prefija con 'bm25:' para usar BM25")
    print("="*70 + "\n")

    while True:
        try:
            entrada = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Hasta luego!")
            break

        if not entrada or entrada.lower() in ("salir", "exit", "quit"):
            print("👋 ¡Hasta luego!")
            break

        if entrada.lower().startswith("bm25:"):
            metodo, query = "bm25", entrada[5:].strip()
        else:
            metodo, query = "tfidf", entrada

        resultados = buscador.buscar(query, top_n=5, metodo=metodo)
        print(f"\n{'='*70}")
        print(f"🔍 \"{query}\"  |  {metodo.upper()}  |  {len(resultados)} resultados")
        print(f"{'='*70}")
        for i, r in enumerate(resultados, 1):
            print(f"\n  📖 #{i}  [Parte {r['parte']}] {r['titulo']}")
            print(f"     Score: {r['score']}")
            print(f"     {r['fragmento'][:200]}")
        print()


def inicializar_motor():
    """Carga HTML, extrae capítulos, construye índices."""
    print(f"📂 Cargando: {HTML_PATH}")
    if not HTML_PATH.exists():
        print(f"❌ No se encontró: {HTML_PATH}")
        sys.exit(1)

    soup = cargar_html(HTML_PATH)
    capitulos = extraer_capitulos(soup)
    print(f"📑 {len(capitulos)} capítulos extraídos.\n")

    nlp = cargar_spacy()
    return BuscadorQuijote(capitulos, nlp)


if __name__ == "__main__":
    buscador = inicializar_motor()

    if "--cli" in sys.argv:
        main_cli(buscador)
    else:
        app = crear_app(buscador)
        print("🌐 Servidor web en http://localhost:5000")
        print("   Abre esa URL en tu navegador para usar la interfaz.\n")
        app.run(debug=False, port=5000)
