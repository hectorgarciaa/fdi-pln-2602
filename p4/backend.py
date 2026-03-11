# ============================================================================
# 5. FLASK WEB SERVER
# ============================================================================

def crear_app(buscador: BuscadorQuijote):
    """Crea la aplicación Flask con la API de búsqueda."""
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS

    app = Flask(__name__, static_folder=os.path.dirname(__file__))
    CORS(app)

    @app.route("/")
    def index():
        return send_from_directory(os.path.dirname(__file__), "index.html")

    @app.route("/api/search", methods=["GET"])
    def api_search():
        query = request.args.get("query", "").strip()
        metodo = request.args.get("metodo", "tfidf").strip().lower()
        top_n = int(request.args.get("top_n", "5"))

        if not query:
            return jsonify({"resultados": [], "query": "", "metodo": metodo})

        resultados = buscador.buscar(query, top_n=top_n, metodo=metodo)
        return jsonify({
            "resultados": resultados,
            "query": query,
            "metodo": metodo,
            "total_capitulos": len(buscador.capitulos),
        })

    @app.route("/api/capitulos", methods=["GET"])
    def api_capitulos():
        """Devuelve la lista de capítulos (sin texto completo)."""
        caps = [
            {"parte": c["parte"], "titulo": c["titulo"], "num_palabras": len(c["texto"].split())}
            for c in buscador.capitulos
        ]
        return jsonify({"capitulos": caps, "total": len(caps)})

    return app

