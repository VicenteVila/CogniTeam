from typing import Dict

from bs4 import BeautifulSoup


def fix_ui_code(code_str: str) -> Dict[str, str]:
    print(f"\n-- [fix_ui_code] Formateando código (len: {len(code_str)})")
    if isinstance(code_str, dict) and "result" in code_str:
        code_str = code_str["result"]
    if not code_str or not code_str.strip():
        return {"result": "/* Código vacío. */"}

    try:
        soup = BeautifulSoup(code_str, "html.parser")
        formatted = soup.prettify()
        return {"result": f"<!-- Formateado por fix_ui_code -->\n{formatted}"}
    except Exception as e:
        print(f"ERROR en fix_ui_code: {e}")
        return {"result": f"<!-- Error al formatear: {e} -->\n{code_str}"}
