import re
from typing import Dict

from cogniteam.tools.utils.llm import llm_complete
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff()
def generate_ui_code(description: str) -> Dict[str, str]:
    print(f"\n-- [generate_ui_code (HTML)] Para: '{description[:100]}...'")
    if not description:
        return {"result": "Error: descripción vacía."}

    react_instr = ""
    if any(k in description.lower() for k in ["react", "jsx", "usestate", "useeffect"]):
        react_instr = """
        Instrucciones Específicas para React/JSX:
        - Incluye React y ReactDOM vía CDN (<script> en <head>).
        - Usa <script type="text/babel"> en <body> para el código JSX.
        - Renderiza el componente en un <div> con id="root".
        - Usa React 18 (ReactDOM.createRoot).
        - Componente funcional y autocontenido.
        """

    prompt = (
        f"Eres un desarrollador frontend experto. Genera un ÚNICO HTML COMPLETO.\n"
        f"CSS inline en <style> en <head>. JS al final del <body> en <script>.\n"
        f"{react_instr}\n"
        f"Output ÚNICA Y EXCLUSIVAMENTE el código HTML.\n"
        f"Debe comenzar con `<!DOCTYPE html>` y terminar con `</html>`.\n"
        f"Sin markdown ni explicaciones.\n\n"
        f"Descripción:\n---\n{description}\n---\n\nHTML COMPLETO:"
    )

    try:
        raw = llm_complete(prompt, task="code", max_tokens=4096, timeout_seconds=120)
        if raw:
            code = raw.strip()
            if "```html" in code:
                m = re.search(r"```html\s*([\s\S]*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            elif code.startswith("```"):
                code = re.sub(r"^```[\w\s]*\n?", "", code)
                code = re.sub(r"\n?```$", "", code).strip()
            doctype = re.search(r"<!DOCTYPE html.*?>", code, re.IGNORECASE | re.DOTALL)
            if doctype:
                code = code[doctype.start():]
            elif not code.lower().startswith("<!doctype"):
                code = "<!DOCTYPE html>\n" + code
            return {"result": code}
        return {"result": "Error: Respuesta vacía de LLM."}
    except Exception as e:
        print(f"ERROR en generate_ui_code: {e}")
        return {"result": f"Error generando HTML: {e}"}
