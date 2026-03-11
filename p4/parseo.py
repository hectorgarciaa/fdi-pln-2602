from conf import FRONTMATTER_IDS
from pathlib import Path
from bs4 import BeautifulSoup

# ============================================================================
# 2. PARSING DEL HTML
# ============================================================================

def cargar_html(ruta: Path) -> BeautifulSoup:
    """Lee el archivo HTML y devuelve un objeto BeautifulSoup."""
    with open(ruta, "r", encoding="utf-8") as f:
        return BeautifulSoup(f.read(), "html.parser")


def extraer_capitulos(soup: BeautifulSoup) -> list[dict]:
    """
    Segmenta el libro en capítulos usando las etiquetas <h3> con <a name="...">.
    Ignora front-matter y licencia de Gutenberg.
    """
    secciones = []
    for h3 in soup.find_all("h3"):
        ancla = h3.find("a", attrs={"name": True})
        if ancla:
            secciones.append({
                "id": ancla["name"],
                "titulo": h3.get_text(strip=True),
                "elemento": h3,
            })

    capitulos = []
    for sec in secciones:
        if sec["id"] in FRONTMATTER_IDS:
            continue

        textos = []
        sibling = sec["elemento"].find_next_sibling()
        while sibling:
            if sibling.name == "h3" and sibling.find("a", attrs={"name": True}):
                break
            if sibling.name == "h2" and sibling.find("a", attrs={"name": True}):
                break
            texto = sibling.get_text(" ", strip=True)
            if texto:
                textos.append(texto)
            sibling = sibling.find_next_sibling()

        texto_cap = " ".join(textos)
        if not texto_cap.strip():
            continue

        parte = "I" if sec["id"].startswith("1_") else "II"
        capitulos.append({
            "id": sec["id"],
            "parte": parte,
            "titulo": sec["titulo"],
            "texto": texto_cap,
        })

    return capitulos