from pathlib import Path

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================

HTML_PATH = Path(__file__).parent / "quijote" / "quijote.html"
NLP_MODEL = "es_core_news_sm"

# IDs de front-matter a ignorar
FRONTMATTER_IDS = {
    "1_tasa", "1_erratas", "1_rey", "1_duque", "1_prologo", "1_libro",
    "2_tasa", "2_erratas", "2_aprobaciones", "2_conde", "2_prologo",
}

# Stopwords en español (con variantes cervantinas)
STOPWORDS_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde",
    "en", "entre", "hacia", "hasta", "para", "por", "según", "sin",
    "so", "sobre", "tras", "mediante", "durante",
    "y", "e", "o", "u", "ni", "que", "si", "pero", "mas", "sino",
    "aunque", "porque", "pues", "como", "cuando", "donde", "mientras",
    "yo", "tú", "él", "ella", "nosotros", "nosotras", "vosotros",
    "vosotras", "ellos", "ellas", "me", "te", "se", "nos", "os", "le",
    "les", "mí", "ti", "sí", "conmigo", "contigo", "consigo",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    "su", "sus", "mi", "mis", "tu", "tus",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra",
    "vuestros", "vuestras", "suyo", "suya", "suyos", "suyas",
    "quien", "quienes", "cual", "cuales", "cuyo", "cuya", "cuyos", "cuyas",
    "no", "sí", "también", "tampoco", "ya", "aún", "aun", "todavía",
    "muy", "más", "menos", "bien", "mal", "aquí", "ahí", "allí",
    "así", "tan", "tanto", "solo", "sólo", "otro", "otra", "otros", "otras",
    "ser", "estar", "haber", "tener", "ir", "hacer", "poder", "decir",
    "es", "era", "fue", "son", "está", "están", "ha", "han", "hay",
    "he", "has", "había", "tiene", "tienen", "tenía", "va",
    "sido", "siendo",
    "dél", "della", "dello", "dellos", "dellas", "deste", "desta",
    "destos", "destas", "desa", "desas", "desos", "deso",
    "vuesa", "vuesas", "vos",
    "al", "del",
}