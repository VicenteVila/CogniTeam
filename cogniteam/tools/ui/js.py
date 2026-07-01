import re
from typing import Dict

from cogniteam.tools.utils.llm import llm_complete
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff()
def generate_js_code(description: str) -> Dict[str, str]:
    print(f"\n-- [generate_js_code] Para: '{description[:100]}...'")
    if not description:
        return {"result": "Error: descripción vacía."}

    prompt = (
        f"Eres un desarrollador JavaScript experto.\n"
        f"Genera código JavaScript PURO y funcional.\n"
        f"Sin etiquetas <script>, sin HTML, sin markdown.\n"
        f"Verifica existencia de elementos DOM antes de manipularlos.\n\n"
        f"Descripción:\n---\n{description}\n---\n\nJS PURO:"
    )

    try:
        raw = llm_complete(prompt, task="code", max_tokens=3500, timeout_seconds=120)
        if raw:
            code = raw.strip()
            if "```javascript" in code:
                m = re.search(r"```javascript\s*([\s\S]*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            elif "```js" in code:
                m = re.search(r"```js\s*([\s\S]*?)\s*```", code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
            elif code.startswith("```"):
                code = re.sub(r"^```[\w\s]*\n?", "", code)
                code = re.sub(r"\n?```$", "", code).strip()
            code = re.sub(r"</?script[^>]*>", "", code, flags=re.I).strip()
            return {"result": code}
        return {"result": "Error: respuesta vacía de LLM."}
    except Exception as e:
        print(f"ERROR en generate_js_code: {e}")
        return {"result": f"Error generando JS: {e}"}
