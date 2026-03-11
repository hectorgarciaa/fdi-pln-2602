from conf import HTML_PATH, FRONTMATTER_IDS

# ============================================================================
# 4. MOTOR DE BÚSQUEDA
# ============================================================================

class BuscadorQuijote:
    """Motor de búsqueda TF-IDF + BM25 con N-gramas (1,2)."""

    def __init__(self, capitulos: list[dict], nlp):
        self.capitulos = capitulos
        self.nlp = nlp

        print("⏳ Preprocesando capítulos...")
        self.textos_procesados = []
        for i, cap in enumerate(capitulos):
            print(f"  [{i+1}/{len(capitulos)}] Parte {cap['parte']} - {cap['titulo'][:55]}...")
            self.textos_procesados.append(preprocesar_texto(cap["texto"], nlp))

        # TF-IDF
        print("\n📊 Construyendo índice TF-IDF (N-gramas 1,2)...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_df=0.85, min_df=1, sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.textos_procesados)

        # BM25
        print("📊 Construyendo índice BM25 (N-gramas 1,2)...")
        self.corpus_bm25 = []
        for texto in self.textos_procesados:
            tokens = texto.split()
            bigrams = [f"{tokens[j]}_{tokens[j+1]}" for j in range(len(tokens) - 1)]
            self.corpus_bm25.append(tokens + bigrams)
        self.bm25 = BM25Okapi(self.corpus_bm25)

        print("✅ Índices listos.\n")

    def buscar(self, query: str, top_n: int = 5, metodo: str = "tfidf") -> list[dict]:
        query_proc = preprocesar_texto(query, self.nlp)
        if not query_proc.strip():
            return []
        if metodo == "bm25":
            return self._buscar_bm25(query_proc, top_n)
        return self._buscar_tfidf(query_proc, top_n)

    def _buscar_tfidf(self, qp: str, top_n: int) -> list[dict]:
        qvec = self.vectorizer.transform([qp])
        sims = cosine_similarity(qvec, self.tfidf_matrix).flatten()
        idxs = sims.argsort()[::-1][:top_n]
        return [self._resultado(i, sims[i], qp) for i in idxs if sims[i] > 0]

    def _buscar_bm25(self, qp: str, top_n: int) -> list[dict]:
        tokens = qp.split()
        bigrams = [f"{tokens[j]}_{tokens[j+1]}" for j in range(len(tokens) - 1)]
        scores = self.bm25.get_scores(tokens + bigrams)
        idxs = scores.argsort()[::-1][:top_n]
        return [self._resultado(i, scores[i], qp) for i in idxs if scores[i] > 0]

    def _resultado(self, idx: int, score: float, qp: str) -> dict:
        cap = self.capitulos[idx]
        return {
            "parte": cap["parte"],
            "titulo": cap["titulo"],
            "score": round(float(score), 4),
            "fragmento": self._extraer_fragmento(cap["texto"], qp),
            "num_palabras": len(cap["texto"].split()),
        }

    @staticmethod
    def _extraer_fragmento(texto: str, query: str, longitud: int = 350) -> str:
        texto_lower = texto.lower()
        mejor_pos = len(texto)
        for t in query.split():
            pos = texto_lower.find(t)
            if 0 <= pos < mejor_pos:
                mejor_pos = pos

        if mejor_pos == len(texto):
            return texto[:longitud].strip() + "..."

        inicio = max(0, mejor_pos - longitud // 4)
        fin = min(len(texto), inicio + longitud)
        frag = texto[inicio:fin].strip()
        if inicio > 0:
            frag = "..." + frag
        if fin < len(texto):
            frag = frag + "..."
        return frag