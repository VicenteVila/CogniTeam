import subprocess
import traceback
from typing import Dict

from cogniteam.config.settings import settings
from cogniteam.utils.io import ask_confirmation


def execute_terminal_command_safe(command: str) -> Dict[str, str]:
    print(f"\n-- [execute_terminal_command_safe] Ejecutando: '{command}'")
    try:
        import shlex
        command_parts = shlex.split(command)
    except Exception as e:
        return {"result": f"Error parseando comando: {e}"}

    if not command_parts:
        return {"result": "Error: comando vacío."}

    first_word = command_parts[0].lower()
    blocked = {
        "sudo", "chmod", "chown", "shutdown", "reboot", "halt", "format",
        "mkfs", "fdisk", "diskpart", "dd", "rm", "del", "erase",
        "runas", "icacls", "cacls",
    }
    if first_word in blocked:
        return {"result": f"Error: comando '{first_word}' bloqueado por seguridad."}

    import os
    project_root = os.path.abspath(settings.project_root)
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
