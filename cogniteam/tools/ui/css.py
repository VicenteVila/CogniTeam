import re
from typing import Dict

from cogniteam.tools.utils.llm import llm_complete
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff()
def generate_css_code(description: str) -> Dict[str, str]:
    print(f"\n-- [generate_css_code] Para: '{description[:100]}...'")
    if not description:
        return {"result": "Error: descripción vacía."}

    prompt = (
        f"Eres un desarrollador frontend experto en CSS.\n"
        f"Genera código CSS PURO. Sin etiquetas <style>, sin HTML, sin markdown.\n\n"
        f"Descripción:\n---\n{description}\n---\n\nCSS PURO:"
    )

    try:
        raw = llm_complete(prompt, task="code", max_tokens=2000, timeout_seconds=120)
        if raw:
            code = raw.strip()
            if "```css" in code:
                m = re.search(r"```css\s*([\s\S]*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            elif code.startswith("```"):
                code = re.sub(r"^```[\w\s]*\n?", "", code)
                code = re.sub(r"\n?```$", "", code).strip()
            code = re.sub(r"</?style[^>]*>", "", code, flags=re.I).strip()
            return {"result": code}
        return {"result": "Error: respuesta vacía de LLM."}
    except Exception as e:
        print(f"ERROR en generate_css_code: {e}")
        return {"result": f"Error generando CSS: {e}"}
