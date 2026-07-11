import re
from typing import Dict, List, Optional


def detect_html_conflicts(html: str) -> List[str]:
    """Detecta implementaciones duplicadas en el HTML combinado.

    Busca:
    1. Múltiples <style> blocks que definan reglas para los mismos selectores
    2. Múltiples <script> blocks con declaraciones globales repetidas
    3. IDs duplicados en elementos HTML
    """
    warnings: List[str] = []

    # ── 1. Múltiples <style> blocks con selectores solapados ──
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, re.DOTALL | re.IGNORECASE)
    if len(style_blocks) > 1:
        selectors_per_block = []
        for block in style_blocks:
            selectors = set(re.findall(r"^\s*([.#]?\w[\w-]*(?:\s*,\s*[.#]?\w[\w-]*)*)\s*\{", block, re.MULTILINE))
            flat = set()
            for s in selectors:
                for part in s.split(","):
                    flat.add(part.strip())
            selectors_per_block.append(flat)
        for i in range(len(selectors_per_block)):
            for j in range(i + 1, len(selectors_per_block)):
                overlap = selectors_per_block[i] & selectors_per_block[j]
                if overlap:
                    warnings.append(
                        f"  [Conflicto CSS] Los <style> #{i+1} y #{j+1} definen los mismos selectores: {', '.join(sorted(overlap)[:5])}"
                    )

    # ── 2. Múltiples <script> blocks con declaraciones globales repetidas ──
    script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    if len(script_blocks) > 1:
        decls_per_block = []
        for block in script_blocks:
            # Extraer const/let/var y function declarations a nivel global
            consts = set(re.findall(r"\b(?:const|let|var)\s+(\w+)", block))
            funcs = set(re.findall(r"(?:^|;|\})\s*function\s+\*?(\w+)\s*\(", block))
            game_funcs = set(re.findall(r"(?:^|;|\})\s*(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>)", block))
            decls_per_block.append(consts | funcs | game_funcs)
        for i in range(len(decls_per_block)):
            for j in range(i + 1, len(decls_per_block)):
                overlap = decls_per_block[i] & decls_per_block[j]
                if overlap:
                    warnings.append(
                        f"  [Conflicto JS] Los <script> #{i+1} y #{j+1} declaran las mismas variables/funciones: {', '.join(sorted(overlap)[:8])}"
                    )
                    break  # Una advertencia por par es suficiente

        # Detectar game loops redundantes
        game_indicators = {"gameloop", "game_loop", "initgame", "updategame", "startgame"}
        blocks_with_game = sum(
            1 for b in script_blocks
            if any(indicator in b.lower() for indicator in game_indicators)
        )
        if blocks_with_game > 1:
            warnings.append(
                f"  [Conflicto JS] {blocks_with_game} bloques <script> contienen funciones de game-loop. "
                "Probablemente hay implementaciones duplicadas."
            )

    # ── 3. IDs duplicados en el HTML ──
    ids = re.findall(r'\bid=["\']([^"\']+)["\']', html)
    seen = set()
    for id_ in ids:
        if id_ in seen:
            warnings.append(f"  [HTML Duplicado] ID '{id_}' aparece múltiples veces.")
        seen.add(id_)

    return warnings


def combine_ui_to_html(
    html: str,
    css: Optional[str] = None,
    js: Optional[str] = None,
    filepath: Optional[str] = None,
) -> Dict[str, str]:
    """Combina HTML, CSS y JS separados en un unico archivo HTML con CSS/JS inline."""
    print(f"\n-- [combine_ui_to_html] Ensamblando HTML ({len(html)}c) + CSS ({len(css or '')}c) + JS ({len(js or '')}c)")

    if not html or not html.strip():
        return {"result": "Error: HTML vacio."}

    has_html_tag = re.search(r"<html", html, re.IGNORECASE)
    has_head_tag = re.search(r"<head", html, re.IGNORECASE)
    has_body_tag = re.search(r"<body", html, re.IGNORECASE)

    if has_html_tag and has_head_tag and has_body_tag:
        if css:
            style_tag = f"<style>\n{css}\n</style>"
            head_end = re.search(r"</head>", html, re.IGNORECASE)
            if head_end:
                html = html[:head_end.start()] + style_tag + "\n" + html[head_end.start():]
            else:
                html += f"\n{style_tag}"

        if js:
            script_tag = f"<script>\n{js}\n</script>"
            body_end = re.search(r"</body>", html, re.IGNORECASE)
            if body_end:
                html = html[:body_end.start()] + script_tag + "\n" + html[body_end.start():]
            else:
                html += f"\n{script_tag}"
    else:
        css_block = f"<style>\n{css}\n</style>\n" if css else ""
        js_block = f"\n<script>\n{js}\n</script>" if js else ""
        html = f"<!DOCTYPE html>\n<html lang=\"es\">\n<head>\n<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n{css_block}</head>\n<body>\n{html}{js_block}\n</body>\n</html>"

    html = html.strip()

    # ── Validación post-combinación ──
    conflicts = detect_html_conflicts(html)
    combined_ok = not conflicts
    for w in conflicts:
        print(f"  ⚠ {w}")
    if not combined_ok:
        print(f"  ⚠ El HTML combinado tiene conflictos de implementación. Revisa el resultado.")
    else:
        print(f"  ✅ HTML combinado sin conflictos detectados.")

    if filepath:
        from cogniteam.tools.filesystem.operations import write_file_sandboxed
        write_result = write_file_sandboxed(filepath, html)
        print(f"  Archivo escrito: {filepath}")
        return {"result": html, "filepath": filepath, "write_result": write_result, "conflicts": conflicts}

    return {"result": html, "conflicts": conflicts}
