from typing import Dict, Optional

from cogniteam.tools.utils.llm import llm_complete
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff()
def extract_info_from_text(
    text_content: str,
    question_or_instruction: str,
    ollama_model_name: Optional[str] = None,
) -> Dict[str, str]:
    print(
        f"\n-- [extract_info_from_text] Query: '{question_or_instruction[:100]}...' "
        f"sobre texto (len: {len(text_content) if text_content else 'N/A'})"
    )
    if not text_content or not question_or_instruction:
        return {"result": "Error: contenido y pregunta requeridos."}

    actual_text = (
        str(text_content["result"])
        if isinstance(text_content, dict) and "result" in text_content
        else str(text_content)
    )

    prompt = (
        f"Eres un asistente IA experto en extracción de información.\n"
        f"Analiza el 'CONTENIDO TEXTUAL' y responde a la 'PREGUNTA O INSTRUCCIÓN'.\n"
        f"BASA TU RESPUESTA ÚNICAMENTE en el contenido textual. NO inventes.\n"
        f"Si la info no está, INDÍCALO. Sé conciso.\n\n"
        f"CONTENIDO TEXTUAL:\n---\n{actual_text[:25000]}\n---\n\n"
        f"PREGUNTA O INSTRUCCIÓN: {question_or_instruction}\n\nRespuesta:"
    )

    try:
        raw = llm_complete(prompt, task="extract", max_tokens=2000, timeout_seconds=120)
        if raw:
            return {"result": raw.strip()}
        return {"result": "Error: LLM respuesta vacía."}
    except Exception as e:
        print(f"ERROR en extract_info: {e}")
        return {"result": f"Error en extracción: {e}"}
