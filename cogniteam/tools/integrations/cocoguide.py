import base64
import io
import json
import os
import traceback
from typing import Dict, List, Optional, Union

from PIL import Image

import litellm
from google import genai
from google.genai.types import HarmBlockThreshold, HarmCategory

from cogniteam.config.settings import settings
from cogniteam.tools.utils.llm import get_genai_model_name, get_litellm_model_name
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff
def generar_guia_cocoguide(
    instruccion_usuario_para_guia: str,
    ruta_archivo_datos_json: Optional[str] = None,
    fragmento_codigo_sugerido: Optional[str] = None,
    script_content_direct: Optional[str] = None,
    screenshot_base64_direct: Optional[str] = None,
    screenshot_format_direct: Optional[str] = "PNG",
) -> Dict[str, str]:
    print(
        f"\n-- [generar_guia_cocoguide] Instrucción: "
        f"'{instruccion_usuario_para_guia[:70]}...'"
    )

    script_text_final: Optional[str] = None
    screenshot_base64_final: Optional[str] = None
    screenshot_format_final: str = "PNG"
    source_script: str = "Script Desconocido"

    if ruta_archivo_datos_json:
        path = ruta_archivo_datos_json
        if not os.path.isfile(path) and not os.path.isabs(path):
            path = os.path.join(settings.project_root, ruta_archivo_datos_json)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                script_text_final = data.get("script_text")
                screenshot_base64_final = data.get("screenshot_base64")
                screenshot_format_final = data.get("screenshot_format", "PNG").lower()
                source_script = data.get(
                    "source_script_path",
                    f"Script desde: {os.path.basename(path)}",
                )
            except Exception as e:
                print(f"  Error leyendo '{path}': {e}")

    if script_content_direct:
        script_text_final = script_content_direct
        source_script = "Script directo"
    if screenshot_base64_direct:
        screenshot_base64_final = screenshot_base64_direct
        screenshot_format_final = screenshot_format_direct.lower()

    if not script_text_final:
        return {"result": "Error: no se pudo obtener el script."}

    contents: List[Union[str, Image.Image]] = [
        f"Eres CoCoGuide, asistente experto en Python. "
        f"Usuario trabaja en: '{source_script}'.\n"
        f"Script:\n```python\n{script_text_final}\n```\n"
    ]
    if fragmento_codigo_sugerido:
        contents.append(
            f"Fragmento a integrar:\n```python\n{fragmento_codigo_sugerido}\n```\n"
        )
    contents.append(f"Instrucción: \"{instruccion_usuario_para_guia}\"\n")

    if screenshot_base64_final:
        contents.append("Imagen adjunta: captura de editor.\n")
        try:
            img_bytes = base64.b64decode(screenshot_base64_final)
            img_pil = Image.open(io.BytesIO(img_bytes))
            contents.append(img_pil)
        except Exception as e:
            print(f"  No se pudo añadir imagen: {e}")

    model = settings.model_name
    if screenshot_base64_final and not any(
        kw in model.lower() for kw in ["gemini-1.5", "vision", "flash", "pro"]
    ):
        model = "gemini-1.5-pro-latest"

    try:
        if model.startswith("ollama/"):
            prompt = "".join(p for p in contents if isinstance(p, str))
            response = litellm.completion(
                model=get_litellm_model_name(model),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2048,
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return {"result": response.choices[0].message.content.strip()}
        else:
            model_instance = genai.GenerativeModel(get_genai_model_name(model))
            safety = [
                {"category": c, "threshold": HarmBlockThreshold.BLOCK_NONE}
                for c in HarmCategory
                if c != HarmCategory.HARM_CATEGORY_UNSPECIFIED
            ]
            response = model_instance.generate_content(
                contents=contents,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
                safety_settings=safety,
            )
            if hasattr(response, "text") and response.text:
                return {"result": response.text.strip()}
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback
                and response.prompt_feedback.block_reason
            ):
                reason = response.prompt_feedback.block_reason
                return {
                    "result": f"Prompt bloqueado. Razón: {reason.name if hasattr(reason, 'name') else reason}"
                }

        return {"result": "Error: respuesta vacía del LLM."}
    except Exception as e:
        print(f"ERROR CoCoGuide: {e}")
        traceback.print_exc()
        return {"result": f"Error CoCoGuide: {e}"}
