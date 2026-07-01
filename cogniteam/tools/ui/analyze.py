from typing import Dict

from bs4 import BeautifulSoup


def analyze_html_js(code: str) -> Dict[str, str]:
    print(f"\n-- [analyze_html_js] Analizando código (len: {len(code)})")
    if isinstance(code, dict) and "result" in code:
        code = code["result"]
    if not code or not code.strip():
        return {"result": "Código vacío."}

    issues = []
    warnings = []
    soup = None
    try:
        soup = BeautifulSoup(code, "html.parser")
    except Exception as e:
        issues.append(f"Error parseando HTML: {e}")

    if soup:
        if not soup.find("html"):
            issues.append("Falta <html>.")
        if not soup.find("head"):
            issues.append("Falta <head>.")
        elif not soup.head.find("title") or not soup.head.find("title").string or not soup.head.find("title").string.strip():
            warnings.append("Falta <title> o está vacío.")
        if not soup.find("body"):
            issues.append("Falta <body>.")
        for i, img in enumerate(soup.find_all("img")):
            if not img.has_attr("alt"):
                warnings.append(f"Img {i+1} (src: '{img.get('src', 'N/A')[:30]}') sin alt.")

    if "<!doctype html>" not in code.lower()[:50]:
        warnings.append("<!DOCTYPE html> no encontrado al inicio.")
    if code.lower().count("<script") > code.lower().count("</script>"):
        issues.append("Desbalance de <script>.")
    if code.lower().count("<style") > code.lower().count("</style>"):
        issues.append("Desbalance de <style>.")

    parts = ["Informe de Análisis HTML/JS:"]
    if issues:
        parts.append("\nProblemas:")
        parts.extend(f"- {issue}" for issue in issues)
    else:
        parts.append("\nSin problemas estructurales.")
    if warnings:
        parts.append("\nAdvertencias:")
        parts.extend(f"- {w}" for w in warnings)
    else:
        parts.append("\nSin advertencias.")
    return {"result": "\n".join(parts)}
