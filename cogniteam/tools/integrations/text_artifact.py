import json
import re
import traceback
from typing import Dict, Optional

from cogniteam.tools.utils.llm import llm_complete
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff()
def generate_textual_artifact(
    description_of_artifact: str,
    context_information: Optional[str] = None,
) -> Dict[str, str]:
    print(f"\n-- [generate_textual_artifact] '{description_of_artifact[:100]}...'")

    actual_context: Optional[str] = None
    if context_information:
        if isinstance(context_information, dict):
            actual_context = str(
                context_information.get("result")
                or context_information.get("data")
                or json.dumps(context_information, indent=2, ensure_ascii=False)
            )
        else:
            actual_context = str(context_information)

    prompt_parts = [
        "Genera un ARTEFACTO TEXTUAL detallado basado en la descripción y contexto.",
        "Output ÚNICA Y EXCLUSIVAMENTE el texto del artefacto.",
        "NO generes código ejecutable a menos que se pida explícitamente.",
        "NO añadas frases introductorias ni conclusiones.",
        f"\nDESCRIPCIÓN:\n---\n{description_of_artifact}\n---",
    ]
    if actual_context and actual_context.strip():
        prompt_parts.append(
            f"\nCONTEXTO:\n---\n{actual_context[:10000]}\n---"
        )
    prompt_parts.append("\n\nARTEFACTO GENERADO:")

    try:
        raw = llm_complete("".join(prompt_parts), task="reasoning", max_tokens=3000, timeout_seconds=120)
        if raw:
            text = raw.strip()
            # Auto-extraer bloque de código si la salida contiene uno
            code_block = re.search(r"```(?:\w+)?\s*\n([\s\S]*?)```", text, re.DOTALL)
            if code_block:
                text = code_block.group(1).strip()
            return {"result": text}
        return {"result": "Error: respuesta vacía del LLM."}
    except Exception as e:
        print(f"ERROR generate_textual_artifact: {e}")
        traceback.print_exc()
        return {"result": f"Error: {e}"}
