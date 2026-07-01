import os
import shutil
import traceback
from typing import Any, Dict, Optional

from cogniteam.config.settings import settings
from cogniteam.tools.base import ToolResponse
from cogniteam.utils.io import ask_confirmation
from cogniteam.utils.security import is_path_safe


def write_file_sandboxed(relative_filepath: str, content: Optional[str]) -> Dict[str, Any]:
    print(f"\n-- [write_file_sandboxed] Escribiendo en '{relative_filepath}'")
    safe_path = is_path_safe(relative_filepath)
    if not safe_path:
        return ToolResponse(
            success=False,
            message=f"Ruta '{relative_filepath}' inválida/insegura.",
            data=None,
        ).model_dump()

    if content is None:
        print(f"  Contenido None, omitiendo escritura de '{relative_filepath}'")
        return ToolResponse(
            success=True,
            message=f"Escritura omitida (content=None).",
            data={"filepath": safe_path, "action": "skipped"},
        ).model_dump()

    if not isinstance(content, str):
        try:
            import json
            content = json.dumps(content, indent=2, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        except Exception as e:
            return ToolResponse(
                success=False,
                message=f"No se pudo convertir content a string: {e}",
                data=None,
            ).model_dump()

    try:
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return ToolResponse(
            success=True,
            message=f"Archivo '{relative_filepath}' escrito en '{safe_path}'.",
            data={"filepath": safe_path, "action": "written"},
        ).model_dump()
    except Exception as e:
        print(f"ERROR write_file_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error escribiendo '{relative_filepath}': {e}",
            data=None,
        ).model_dump()


def read_file_sandboxed(relative_filepath: str) -> Dict[str, Any]:
    print(f"\n-- [read_file_sandboxed] Leyendo '{relative_filepath}'")
    safe_path = is_path_safe(relative_filepath)
    if not safe_path:
        if os.path.isabs(relative_filepath) and os.path.isfile(relative_filepath):
            print(f"  Usando ruta absoluta fuera del proyecto: '{relative_filepath}'")
            safe_path = relative_filepath
        else:
            return ToolResponse(
                success=False,
                message=f"Ruta '{relative_filepath}' inválida/insegura.",
                data=None,
            ).model_dump()

    try:
        if not os.path.isfile(safe_path):
            return ToolResponse(
                success=False,
                message=f"Archivo '{relative_filepath}' no encontrado.",
                data=None,
            ).model_dump()
        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolResponse(
            success=True,
            message=f"Archivo '{relative_filepath}' leído.",
            data=content,
        ).model_dump()
    except Exception as e:
        print(f"ERROR read_file_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error leyendo '{relative_filepath}': {e}",
            data=None,
        ).model_dump()


def list_files_sandboxed(relative_dirpath: Optional[str] = None) -> Dict[str, str]:
    path_to_list = relative_dirpath if relative_dirpath and relative_dirpath.strip() else "."
    print(f"\n-- [list_files_sandboxed] Listando '{path_to_list}'")
    safe_path = is_path_safe(path_to_list)
    if not safe_path:
        return {"result": f"Ruta '{path_to_list}' inválida o fuera del proyecto."}

    try:
        if not os.path.isdir(safe_path):
            return {"result": f"'{path_to_list}' no es un directorio."}
        items = os.listdir(safe_path)
        display = os.path.relpath(safe_path, os.path.abspath(settings.project_root))
        display_path = settings.project_root if display == "." else os.path.join(settings.project_root, display)
        parts = [f"Contenidos de '{display_path}':"]
        if not items:
            parts.append("  (vacío)")
        else:
            detailed = [
                {"name": i, "is_dir": os.path.isdir(os.path.join(safe_path, i))}
                for i in items
            ]
            detailed.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
            for item in detailed:
                parts.append(f"  [{'DIR' if item['is_dir'] else 'FILE'}] {item['name']}")
        return {"result": "\n".join(parts)}
    except Exception as e:
        print(f"ERROR list_files_sandboxed: {e}")
        traceback.print_exc()
        return {"result": f"Error listando '{path_to_list}': {e}"}


def create_directory_sandboxed(relative_dirpath: str) -> Dict[str, Any]:
    print(f"\n-- [create_directory_sandboxed] Creando '{relative_dirpath}'")
    safe_path = is_path_safe(relative_dirpath)
    if not safe_path:
        return ToolResponse(
            success=False,
            message=f"Ruta '{relative_dirpath}' inválida/insegura.",
        ).model_dump()

    try:
        if os.path.exists(safe_path) and os.path.isdir(safe_path):
            return ToolResponse(
                success=True,
                message=f"Directorio '{relative_dirpath}' ya existe.",
                data={"path": safe_path},
            ).model_dump()
        if os.path.exists(safe_path) and not os.path.isdir(safe_path):
            return ToolResponse(
                success=False,
                message=f"Ya existe un archivo en '{relative_dirpath}'.",
            ).model_dump()
        os.makedirs(safe_path, exist_ok=True)
        return ToolResponse(
            success=True,
            message=f"Directorio '{relative_dirpath}' creado.",
            data={"path": safe_path},
        ).model_dump()
    except Exception as e:
        print(f"ERROR create_directory_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error creando '{relative_dirpath}': {e}",
        ).model_dump()


def delete_file_sandboxed(relative_filepath: str) -> Dict[str, Any]:
    print(f"\n-- [delete_file_sandboxed] '{relative_filepath}'")
    safe_path = is_path_safe(relative_filepath)
    if not safe_path:
        return ToolResponse(
            success=False,
            message=f"Ruta '{relative_filepath}' inválida/insegura.",
        ).model_dump()
    if not (os.path.exists(safe_path) and os.path.isfile(safe_path)):
        return ToolResponse(
            success=False,
            message=f"'{relative_filepath}' no encontrado o no es archivo.",
        ).model_dump()
    if not ask_confirmation(f"¿Borrar '{relative_filepath}'? Irreversible."):
        return ToolResponse(
            success=False,
            message="Cancelado por el usuario.",
        ).model_dump()
    try:
        os.remove(safe_path)
        return ToolResponse(
            success=True,
            message=f"'{relative_filepath}' borrado.",
            data={"filepath": safe_path},
        ).model_dump()
    except Exception as e:
        print(f"ERROR delete_file_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error borrando '{relative_filepath}': {e}",
        ).model_dump()


def delete_directory_sandboxed(relative_dirpath: str) -> Dict[str, Any]:
    print(f"\n-- [delete_directory_sandboxed] '{relative_dirpath}'")
    safe_path = is_path_safe(relative_dirpath)
    if not safe_path:
        return ToolResponse(
            success=False,
            message=f"Ruta '{relative_dirpath}' inválida/insegura.",
        ).model_dump()
    root_abs = os.path.abspath(settings.project_root)
    if os.path.abspath(safe_path) == root_abs:
        return ToolResponse(
            success=False,
            message="No se permite borrar la raíz del proyecto.",
        ).model_dump()
    if not (os.path.exists(safe_path) and os.path.isdir(safe_path)):
        return ToolResponse(
            success=False,
            message=f"'{relative_dirpath}' no encontrado o no es directorio.",
        ).model_dump()
    if not ask_confirmation(f"¿Borrar directorio '{relative_dirpath}' y TODO su contenido? Irreversible."):
        return ToolResponse(
            success=False,
            message="Cancelado por el usuario.",
        ).model_dump()
    try:
        shutil.rmtree(safe_path)
        return ToolResponse(
            success=True,
            message=f"Directorio '{relative_dirpath}' borrado.",
            data={"path": safe_path},
        ).model_dump()
    except Exception as e:
        print(f"ERROR delete_directory_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error borrando '{relative_dirpath}': {e}",
        ).model_dump()


def move_or_rename_sandboxed(source_relative_path: str, destination_relative_path: str) -> Dict[str, Any]:
    print(f"\n-- [move_or_rename_sandboxed] '{source_relative_path}' -> '{destination_relative_path}'")
    safe_source = is_path_safe(source_relative_path)
    safe_dest = is_path_safe(destination_relative_path)
    if not safe_source:
        return ToolResponse(success=False, message=f"Origen '{source_relative_path}' inválido.").model_dump()
    if not safe_dest:
        return ToolResponse(success=False, message=f"Destino '{destination_relative_path}' inválido.").model_dump()

    root_abs = os.path.abspath(settings.project_root)
    if os.path.abspath(safe_source) == root_abs:
        return ToolResponse(success=False, message="No se puede mover la raíz del proyecto.").model_dump()
    if not os.path.exists(safe_source):
        return ToolResponse(success=False, message=f"Origen '{source_relative_path}' no existe.").model_dump()
    if os.path.exists(safe_dest) and not ask_confirmation(f"Destino '{destination_relative_path}' ya existe. ¿Sobrescribir?"):
        return ToolResponse(success=False, message="Cancelado.").model_dump()
    if not ask_confirmation(f"¿Mover/renombrar '{source_relative_path}' -> '{destination_relative_path}'?"):
        return ToolResponse(success=False, message="Cancelado.").model_dump()
    try:
        os.makedirs(os.path.dirname(safe_dest), exist_ok=True)
        shutil.move(safe_source, safe_dest)
        return ToolResponse(
            success=True,
            message=f"Movido a '{destination_relative_path}'.",
            data={"source": safe_source, "destination": safe_dest},
        ).model_dump()
    except Exception as e:
        print(f"ERROR move_or_rename_sandboxed: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error moviendo '{source_relative_path}': {e}",
        ).model_dump()
