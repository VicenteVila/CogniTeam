import os
import subprocess
import traceback
from typing import Dict, List, Optional

from cogniteam.config.settings import settings
from cogniteam.utils.io import ask_confirmation


def _ensure_git_repo() -> bool:
    git_dir = os.path.join(settings.project_root, ".git")
    if not (os.path.exists(git_dir) and os.path.isdir(git_dir)):
        return False
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, cwd=settings.project_root,
            timeout=10, check=False,
        )
        return proc.returncode == 0 and "true" in proc.stdout.strip().lower()
    except Exception:
        return False


def _execute_git_command_robust(command_list: List[str]) -> str:
    if not command_list or command_list[0].lower() != "git":
        return "Error: Solo comandos Git."

    print(f"\n-- [Git] `{' '.join(command_list)}` en '{settings.project_root}'")

    is_repo = _ensure_git_repo()
    allowed_without_repo = {"init", "version", "help", "--version", "--help"}
    git_subcommand = command_list[1] if len(command_list) > 1 else ""
    if git_subcommand not in allowed_without_repo and not is_repo:
        return (
            f"Error: '{settings.project_root}' no es un repositorio Git. "
            f"Ejecuta `git init` primero."
        )

    confirm_needed = {
        "init", "add", "commit", "push", "pull", "merge", "rebase",
        "reset", "checkout", "branch", "tag", "clean", "remote",
    }
    full_cmd = " ".join(command_list)
    if any(cmd in full_cmd for cmd in confirm_needed):
        if not ask_confirmation(f"¿Ejecutar `{full_cmd}` en '{settings.project_root}'?"):
            return "Cancelado por el usuario."

    try:
        proc = subprocess.run(
            command_list,
            capture_output=True, text=True, cwd=settings.project_root,
            timeout=60, check=False,
        )
        parts = [
            f"Comando: `{' '.join(command_list)}`",
            f"Código: {proc.returncode}",
        ]
        if proc.stdout.strip():
            parts.append(f"--- STDOUT ---\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            parts.append(f"--- STDERR ---\n{proc.stderr.strip()}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Timeout (60s) en `{' '.join(command_list)}`."
    except FileNotFoundError:
        return "Error: 'git' no encontrado en PATH."
    except Exception as e:
        print(f"ERROR _execute_git_command_robust: {e}")
        traceback.print_exc()
        return f"Error: {e}"


def git_status() -> Dict[str, str]:
    return {"result": _execute_git_command_robust(["git", "status"])}


def git_add(files: List[str]) -> Dict[str, str]:
    if not isinstance(files, list) or not files:
        return {"result": "Error: 'files' debe ser una lista no vacía."}
    return {"result": _execute_git_command_robust(["git", "add"] + files)}


def git_commit(message: str) -> Dict[str, str]:
    if not message or not message.strip():
        return {"result": "Error: mensaje de commit requerido."}
    return {"result": _execute_git_command_robust(["git", "commit", "-m", message])}


def git_diff() -> Dict[str, str]:
    return {"result": _execute_git_command_robust(["git", "diff"])}


def git_log(limit: Optional[int] = 10) -> Dict[str, str]:
    cmd = ["git", "log", f"-{max(1, limit if limit else 10)}", "--oneline"]
    return {"result": _execute_git_command_robust(cmd)}


def git_push() -> Dict[str, str]:
    return {"result": _execute_git_command_robust(["git", "push"])}


def git_pull() -> Dict[str, str]:
    return {"result": _execute_git_command_robust(["git", "pull"])}
