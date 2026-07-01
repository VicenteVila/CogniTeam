import os
import traceback
from typing import Dict, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

from cogniteam.config.settings import settings
from cogniteam.utils.security import is_path_safe
from xml.sax.saxutils import escape


def create_pdf_from_text(
    relative_filepath: str,
    title_str: str,
    text_content: str,
) -> Dict[str, str]:
    print(f"\n-- [create_pdf_from_text] '{relative_filepath}'")

    actual_text = text_content
    if isinstance(text_content, dict):
        actual_text = str(text_content.get("result") or text_content.get("data", ""))

    filepath_with_ext = (
        relative_filepath
        if relative_filepath.lower().endswith(".pdf")
        else relative_filepath + ".pdf"
    )
    safe_path = is_path_safe(filepath_with_ext)
    if not safe_path:
        return {"result": f"Ruta '{filepath_with_ext}' inválida."}

    title = (title_str or "Documento Sin Título").strip()
    content = (actual_text or "Contenido no proporcionado.").strip()

    try:
        doc = SimpleDocTemplate(
            safe_path,
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18,
        )
        styles = getSampleStyleSheet()

        elements = []
        title_style = styles["h1"]
        title_style.alignment = TA_CENTER
        elements.append(Paragraph(escape(title), title_style))
        elements.append(Spacer(1, 0.25 * inch))

        body_style = styles["Normal"]
        body_style.alignment = TA_JUSTIFY
        escaped = escape(content).replace("\n", "<br/>\n")
        elements.append(Paragraph(escaped, body_style))

        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        doc.build(elements)
        return {"result": f"PDF creado: '{filepath_with_ext}'."}
    except Exception as e:
        print(f"ERROR create_pdf_from_text: {e}")
        traceback.print_exc()
        return {"result": f"Error creando PDF: {e}"}
