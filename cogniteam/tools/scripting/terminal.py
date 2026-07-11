import os
import re
import subprocess
import traceback
from typing import Dict

from cogniteam.config.settings import settings
from cogniteam.utils.io import ask_confirmation
from cogniteam.utils.security import is_command_blocked


# Patterns de comandos encadenados/peligrosos en toda la string
_CHAINED_PATTERNS = [
    r"\|\s*(bash|sh|python|python3)\b",
    r"(?<!\$)\|\s*",          # pipe (|)
    r";\s*(bash|sh|python|python3)\b",
    r"&&\s*(bash|sh|python|python3)\b",
    r"\|\|",
    r"(^|[;&|])\s*\w+\\s+-c\s+",
    r"`[^`]+`",
    r"\$\([^)]+\)",
    r"(python|python3)\s+-c\s+['\"]",
    r"(bash|sh)\s+-c\s+['\"]",
    r"(sudo|doas)\s+",
]


def _has_dangerous_chaining(command: str) -> bool:
    """Detecta pipes, subcomandos, chaining y code injection en la command."""
    for pattern in _CHAINED_PATTERNS:
        if re.search(pattern, command):
            return True
    return False


def execute_terminal_command_safe(command: str) -> Dict[str, str]:
    print(f"\n-- [execute_terminal_command_safe] Ejecutando: '{command}'")
    try:
        import shlex
        command_parts = shlex.split(command)
    except Exception as e:
        return {"result": f"Error parseando comando: {e}"}

    if not command_parts:
        return {"result": "Error: comando vacío."}

    # Usar la función centralizada de security.py
    if is_command_blocked(command):
        return {"result": "Error: comando bloqueado por seguridad."}

    # Detectar chaining peligroso en toda la string (no solo el primer token)
    if _has_dangerous_chaining(command):
        print(f"  Chaining peligroso detectado: '{command}'")
        return {"result": "Error: comando rechazado por seguridad (chaining/subcomando detectado)."}

    project_root = os.path.abspath(settings.project_root)
    first_word = command_parts[0].lower()
    if os.path.isabs(first_word) and not first_word.startswith(project_root):
        return {"result": f"Error: rutas absolutas fuera de '{settings.project_root}' bloqueadas."}

    if not ask_confirmation(
        f"¿Ejecutar `{command}` en '{settings.project_root}'?"
    ):
        return {"result": "Cancelado por el usuario."}

    try:
        proc = subprocess.run(
            command_parts,
            capture_output=True,
            text=True,
            cwd=settings.project_root,
            timeout=60,
            check=False,
        )
        parts = [
            f"Comando: `{command}`",
            f"Código: {proc.returncode}",
        ]
        if proc.stdout:
            parts.append(f"--- STDOUT ---\n{proc.stdout.strip()}")
        if proc.stderr:
            parts.append(f"--- STDERR ---\n{proc.stderr.strip()}")
        return {"result": "\n".join(parts)}
    except subprocess.TimeoutExpired:
        return {"result": f"Timeout (60s) ejecutando '{command}'."}
    except FileNotFoundError:
        return {"result": f"Comando '{command_parts[0]}' no encontrado en PATH."}
    except Exception as e:
        print(f"ERROR execute_terminal_command_safe: {e}")
        traceback.print_exc()
        return {"result": f"Error ejecutando '{command}': {e}"}
