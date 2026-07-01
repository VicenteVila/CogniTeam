from cogniteam.tools.integrations.api import call_api_real
from cogniteam.tools.integrations.cocoguide import generar_guia_cocoguide
from cogniteam.tools.integrations.pdf import create_pdf_from_text
from cogniteam.tools.integrations.text_artifact import generate_textual_artifact
from cogniteam.tools.integrations.tts import speak_text

__all__ = [
    "call_api_real",
    "create_pdf_from_text",
    "generate_textual_artifact",
    "generar_guia_cocoguide",
    "speak_text",
]
