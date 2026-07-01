import re
from typing import Dict, Optional


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
        html_lower = html.lower()
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
    if filepath:
        from cogniteam.tools.filesystem.operations import write_file_sandboxed
        write_result = write_file_sandboxed(filepath, html)
        print(f"  Archivo escrito: {filepath}")
        return {"result": html, "filepath": filepath, "write_result": write_result}

    return {"result": html}
