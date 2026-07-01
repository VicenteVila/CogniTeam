import difflib
import os
import traceback
from typing import Dict, List, Optional

from cogniteam.config.settings import settings
from cogniteam.tools.base import ToolResponse
from cogniteam.tools.scripting.terminal import execute_terminal_command_safe
from cogniteam.utils.io import ask_confirmation
from cogniteam.utils.security import is_path_safe


def propose_script(description: str, filepath: Optional[str] = None) -> Dict[str, str]:
    print(f"\n-- [propose_script] Creando script para: '{description[:100]}...'")
    try:
        from cogniteam.tools.utils.llm import llm_complete

        context = ""
        if filepath:
            safe = is_path_safe(filepath)
            if safe and os.path.isfile(safe):
                try:
                    with open(safe, "r", encoding="utf-8") as f:
                        context = f.read()
                except Exception:
                    pass

        prompt = (
            f"Eres un ingeniero de software experto. Genera UN script de bash/shell "
            f"que realice lo siguiente:\n{description}\n\n"
            f"{'Contexto del archivo:\n' + context[:3000] if context else ''}\n\n"
            f"Output ÚNICAMENTE el código del script, sin markdown, sin explicaciones."
        )

        raw = llm_complete(prompt, task="code", max_tokens=2000, timeout_seconds=60)
        if raw:
            code = raw.strip()
            import re
            for p in [r"```bash\s*([\s\S]*?)\s*```", r"```sh\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
                m = re.search(p, code, re.DOTALL)
                if m:
                    code = m.group(1).strip()
                    break
            return {"result": code}
        return {"result": "Error: respuesta vacía del LLM."}
    except Exception as e:
        print(f"ERROR propose_script: {e}")
        traceback.print_exc()
        return {"result": f"Error: {e}"}


def apply_script(script: str, filepath: Optional[str] = None) -> Dict[str, str]:
    print("\n-- [apply_script] Aplicando script")
    if not script:
        return {"result": "Error: script vacío."}

    print(f"Script a ejecutar:\n{script}")
    if not ask_confirmation("¿Ejecutar este script?"):
        return {"result": "Cancelado."}

    save_msg = ""
    if filepath:
        safe = is_path_safe(filepath)
        if safe:
            try:
                os.makedirs(os.path.dirname(safe), exist_ok=True)
                with open(safe, "w", encoding="utf-8") as f:
                    f.write(script)
                save_msg = f"Script guardado en '{filepath}'.\n"
            except Exception as e:
                save_msg = f"(No se pudo guardar: {e})\n"

    result = execute_terminal_command_safe(script)
    if save_msg:
        if "result" in result and not result["result"].startswith("Error"):
            result["result"] = save_msg + result["result"]
        elif "result" in result:
            result["result"] = result["result"] + f"\n{save_msg}"
    return result


def validate_script(script: str) -> Dict[str, str]:
    print("\n-- [validate_script] Validando script")
    if not script:
        return {"result": "Error: script vacío."}
    if len(script) > 50000:
        return {"result": "Error: script demasiado largo (>50000c)."}

    import subprocess
    import tempfile

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False,
                                         encoding="utf-8") as f:
            f.write(script)
            tmppath = f.name

        proc = subprocess.run(
            ["bash", "-n", tmppath],
            capture_output=True, text=True, timeout=30,
        )
        os.unlink(tmppath)

        if proc.returncode == 0:
            return {"result": "✓ Sintaxis válida."}
        else:
            msg = proc.stderr.strip() or proc.stdout.strip() or "Error de sintaxis"
            return {"result": f"✗ Error de sintaxis:\n{msg}"}
    except subprocess.TimeoutExpired:
        return {"result": "⚠ Timeout validando sintaxis."}
    except FileNotFoundError:
        return {"result": "⚠ bash no disponible en PATH."}
    except Exception as e:
        return {"result": f"Error validando: {e}"}


def view_script_diff(filepath: str, new_content: str) -> Dict[str, str]:
    print(f"\n-- [view_script_diff] Mostrando diff para '{filepath}'")
    safe = is_path_safe(filepath)
    if not safe or not os.path.isfile(safe):
        return {"result": f"Archivo '{filepath}' no encontrado o fuera del proyecto."}

    try:
        with open(safe, "r", encoding="utf-8") as f:
            old_lines = f.readlines()
    except Exception as e:
        return {"result": f"Error leyendo '{filepath}': {e}"}

    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filepath,
        tofile=filepath + " (nuevo)",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    return {"result": diff_text if diff_text else "Sin diferencias."}
