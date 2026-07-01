from cogniteam.config.constants import *
from cogniteam.config.settings import settings


def ask_confirmation(prompt_text: str) -> bool:
    if settings.auto_confirm:
        print(f"  (auto-confirmado: {prompt_text[:60]}...)")
        return True
    try:
        response = input(
            f"\n{COLOR_MAGENTA}??? CONFIRMACIÓN ???{COLOR_RESET}\n{prompt_text}\n¿Proceder? [S/N]: "
        )
        return response.strip().upper() == "S"
    except EOFError:
        return False
    except Exception as e:
        print(f"  Error en ask_confirmation: {e}")
        return False


def ask_user(question: str) -> str:
    if not question or not question.strip():
        return "Error: Pregunta vacía."
    print(f"\n{COLOR_MAGENTA}??? CONFIRMACIÓN ???{COLOR_RESET}\n{question}")
    try:
        response = input("Tu respuesta: ").strip()
        return response if response else "Usuario no proveyó respuesta."
    except EOFError:
        return "Usuario señaló EOF, no se proveyó respuesta."
    except Exception as e:
        return f"Error obteniendo respuesta del usuario: {e}"
