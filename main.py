#!/usr/bin/env python3
"""CogniTeam - Entry point for the multi-agent system."""

import asyncio
import inspect
import os
import sys
import traceback
from typing import Any, Callable, Dict, List

from pathlib import Path

import traceforge

from cogniteam.agents import (
    create_debugger_agent,
    create_developer_agent,
    create_planner_agent,
    create_ui_designer_agent,
    generate_plan,
)
from cogniteam.config.settings import settings
from cogniteam.core.orchestrator import run_orchestrated_flow
from cogniteam.core.session import get_session_service
# Tools are loaded via _build_tool_map()


def _build_tool_map() -> Dict[str, Callable]:
    from cogniteam.tools.filesystem.operations import (
        create_directory_sandboxed,
        delete_directory_sandboxed,
        delete_file_sandboxed,
        list_files_sandboxed,
        move_or_rename_sandboxed,
        read_file_sandboxed,
        write_file_sandboxed,
    )
    from cogniteam.tools.git.operations import (
        git_add,
        git_commit,
        git_diff,
        git_log,
        git_pull,
        git_push,
        git_status,
    )
    from cogniteam.tools.integrations.api import call_api_real
    from cogniteam.tools.integrations.cocoguide import generar_guia_cocoguide
    from cogniteam.tools.integrations.pdf import create_pdf_from_text
    from cogniteam.tools.integrations.text_artifact import generate_textual_artifact
    from cogniteam.tools.integrations.tts import speak_text
    from cogniteam.tools.scripting.proposals import apply_script, propose_script, validate_script, view_script_diff
    from cogniteam.tools.scripting.terminal import execute_terminal_command_safe
    from cogniteam.tools.ui.analyze import analyze_html_js
    from cogniteam.tools.ui.combine import combine_ui_to_html
    from cogniteam.tools.ui.css import generate_css_code
    from cogniteam.tools.ui.fix import fix_ui_code
    from cogniteam.tools.ui.html import generate_ui_code
    from cogniteam.tools.ui.js import generate_js_code
    from cogniteam.tools.web.browse import browse_web_page
    from cogniteam.tools.web.extract import extract_info_from_text
    from cogniteam.tools.web.search import web_search_real

    funcs: List[Callable] = [
        web_search_real,
        browse_web_page,
        extract_info_from_text,
        generate_ui_code,
        generate_css_code,
        generate_js_code,
        combine_ui_to_html,
        generate_textual_artifact,
        analyze_html_js,
        fix_ui_code,
        write_file_sandboxed,
        read_file_sandboxed,
        list_files_sandboxed,
        create_directory_sandboxed,
        delete_file_sandboxed,
        delete_directory_sandboxed,
        move_or_rename_sandboxed,
        execute_terminal_command_safe,
        propose_script,
        apply_script,
        validate_script,
        view_script_diff,
        git_status,
        git_add,
        git_commit,
        git_diff,
        git_log,
        git_push,
        git_pull,
        call_api_real,
        create_pdf_from_text,
        generar_guia_cocoguide,
        speak_text,
    ]

    return {f.__name__: f for f in funcs if hasattr(f, "__name__")}


def _categorize_tools(tool_map: Dict[str, Callable]) -> Dict[str, List[Callable]]:
    ui_tools_names = {
        "generate_ui_code", "generate_css_code", "generate_js_code",
        "combine_ui_to_html",
        "generate_textual_artifact", "analyze_html_js", "fix_ui_code",
        "read_file_sandboxed", "write_file_sandboxed",
    }
    debugger_tools_names = {
        "analyze_html_js", "fix_ui_code", "read_file_sandboxed",
        "write_file_sandboxed", "generate_textual_artifact",
        "extract_info_from_text", "execute_terminal_command_safe",
        "generate_ui_code", "generate_css_code", "generate_js_code",
        "generar_guia_cocoguide", "speak_text", "propose_script",
        "apply_script", "validate_script", "view_script_diff",
    }
    developer_tools_names = set(tool_map.keys()) - ui_tools_names

    return {
        "ui": [tool_map[n] for n in ui_tools_names if n in tool_map],
        "debugger": [tool_map[n] for n in debugger_tools_names if n in tool_map],
        "developer": [tool_map[n] for n in developer_tools_names if n in tool_map],
        "planner": [],  # Planner doesn't use tools directly
    }


def _warmup_litellm():
    """Pre-import litellm to avoid 30s delay on first use."""
    if settings.use_ollama and settings.ollama_base_url:
        try:
            import litellm
            litellm.api_base = settings.ollama_base_url
            litellm.suppress_debug_info = True
            litellm.set_verbose = False
        except Exception:
            pass

def _save_result(flow_result: Dict, manifest, timestamp: str, snapshot_before: set):
    """Copy new/modified files to proyectos_finalizados/ and write a summary report."""
    import json, os, shutil, time
    from pathlib import Path

    out_dir = Path(settings.project_root) / "proyectos_finalizados" / f"RUN_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Find new/modified files ──
    new_files = []
    for root, dirs, files in os.walk(settings.project_root):
        rel = os.path.relpath(root, settings.project_root)
        if rel.startswith("proyectos_finalizados") or rel.startswith(".venv") or rel.startswith(".cogniteam") or rel.startswith("__pycache__") or rel.startswith(".git"):
            continue
        for f in files:
            fp = os.path.normpath(os.path.join(rel, f))
            if fp not in snapshot_before:
                new_files.append(fp)

    if new_files:
        artifacts_dir = out_dir / "artefactos"
        artifacts_dir.mkdir(exist_ok=True)
        for fp in new_files:
            src = os.path.join(settings.project_root, fp)
            dst = artifacts_dir / fp
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Excluir archivos del proyecto (main.py, etc.) que no deben moverse
            if fp.startswith("main.py") or fp.startswith("cogniteam/") or fp.startswith("config/"):
                print(f"  -> omitido (proyecto): {fp}")
                continue
            try:
                shutil.move(src, dst)
                print(f"  -> movido: {fp}")
            except Exception as e:
                try:
                    shutil.copy2(src, dst)
                    os.unlink(src)
                    print(f"  -> copiado (move falló: {e}): {fp}")
                except Exception as e2:
                    print(f"  -> ERROR moviendo {fp}: {e} | copy2: {e2}")

    # ── 2. Summary report ──
    success = flow_result.get("success", False)
    elapsed = flow_result.get("elapsed", 0)
    plan_summary = flow_result.get("plan_summary", "")
    steps = flow_result.get("steps", [])
    outputs = flow_result.get("outputs", {})

    lines = []
    lines.append(f"# Resultado CogniTeam — {timestamp}")
    lines.append(f"")
    lines.append(f"**Estado:** {'✅ Completado' if success else '❌ Fallido'}")
    lines.append(f"**Tiempo total:** {elapsed:.1f}s")
    lines.append(f"**Pasos ejecutados:** {len(steps)}")
    lines.append(f"")

    if manifest:
        lines.append(f"## Clasificación (Scoping Agent)")
        ci = manifest.classification
        lines.append(f"- **Dominio principal:** {ci.domain_key}.{ci.archetype_key}")
        lines.append(f"- **Confianza:** {ci.confidence:.0%}")
        if manifest.secondary_classifications:
            for sc in manifest.secondary_classifications:
                lines.append(f"- **Secundario:** {sc.domain_key}.{sc.archetype_key} (conf: {sc.confidence:.0%})")
        lines.append(f"")
        lines.append(f"**Tarea clarificada:** {manifest.clarified_task}")
        lines.append(f"")

    lines.append(f"## Plan ejecutado")
    if plan_summary:
        lines.append(f"```\n{plan_summary}\n```")
    else:
        for s in steps:
            tool = s.get("tool_to_use", "?")
            desc = s.get("action_description", "")[:120]
            lines.append(f"- **{tool}:** {desc}")
    lines.append(f"")

    if new_files:
        lines.append(f"## Archivos generados ({len(new_files)})")
        for fp in sorted(new_files):
            lines.append(f"- `{fp}`")
        lines.append(f"")

    if outputs:
        lines.append(f"## Outputs de pasos")
        for k, v in list(outputs.items())[:10]:
            v_str = str(v)[:200]
            lines.append(f"- **{k}:** {v_str}")
        if len(outputs) > 10:
            lines.append(f"- ... y {len(outputs)-10} outputs más")

    report_path = out_dir / "reporte.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Resultados guardados en: {out_dir}/")
    print(f"  Reporte: {report_path}")


async def main():
    _warmup_litellm()
    model_for = "Ollama" if settings.use_ollama else settings.model_name
    print(f"\n{'='*60}")
    print(f"  CogniTeam v0.2.0")
    print(f"  Proyecto: {settings.project_root}")
    print(f"  Modelo: {model_for}")
    print(f"  Modo: {'Local (Ollama)' if settings.use_ollama else 'API'}")
    print(f"{'='*60}")

    # Initialize session store
    _ = get_session_service()

    # Initialize memory modules
    try:
        from cogniteam.memory.hmem import get_hmem
        hmem = get_hmem()
        hmem.load()
        print(f"  H-MEM cargada ({len(hmem.nodes)} nodos, {len(hmem.knowledge_triplets)} tripletes)")

        from cogniteam.memory.graphrag import get_graphrag
        graphrag = get_graphrag()
        graphrag.load()
        print(f"  GraphRAG cargada ({graphrag.graph.number_of_nodes()} nodos)")

        from cogniteam.memory.skills import get_skills
        skills = get_skills()
        skills.load()
        print(f"  Skills cargada ({len(skills.skills)} skills, {len(skills.nodes)} nodos de grafo)")

        from cogniteam.memory.matm import get_matm
        matm = get_matm()
        matm.load()
        print(f"  MATM cargada ({len(matm.memories)} memorias)")

        from cogniteam.memory.fastslow import get_fastslow
        fs = get_fastslow()
        fs.load()
        print(f"  Fast-Slow cargada ({len(fs.population)} políticas)")

        print("[Memoria] Todos los módulos de memoria inicializados.")
    except Exception as e:
        print(f"  [Memoria] Error inicializando memoria: {e}")

    # ── Inicializar TraceForge ──
    TRACEFORGE_DB = os.path.join(settings.project_root, ".cogniteam", "traceforge.db")
    traceforge.configure(collector="sqlite", db_path=TRACEFORGE_DB)
    print(f"  TraceForge activado → {TRACEFORGE_DB}")

    # Build tool map
    tool_map = _build_tool_map()
    categorized = _categorize_tools(tool_map)

    agent_name = settings.app_name

    # Create agents (now data containers, no ADK)
    planner_agent = create_planner_agent(
        agent_name=f"PlannerAgent_{agent_name}",
        instruction="",
    )
    ui_agent = create_ui_designer_agent(
        agent_name=f"UIDesignerAgent_{agent_name}",
        tools=categorized["ui"],
    )
    debugger_agent = create_debugger_agent(
        agent_name=f"DebuggerAgent_{agent_name}",
        tools=categorized["debugger"],
    )
    developer_agent = create_developer_agent(
        agent_name=f"DeveloperAgent_{agent_name}",
        tools=categorized["developer"],
    )

    # Build tools/agents descriptions for planner prompt
    TOOL_DESCRIPTIONS = {
        "generate_ui_code": "Genera HTML completo desde descripcion en lenguaje natural. Output: HTML string.",
        "generate_css_code": "Genera CSS desde descripcion en lenguaje natural. Output: CSS string.",
        "generate_js_code": "Genera JS desde descripcion en lenguaje natural. Output: JS string.",
        "combine_ui_to_html": "Toma HTML, CSS, JS y los combina en un unico archivo HTML con CSS/JS inline. Output: HTML final.",
        "generate_textual_artifact": "Genera documentacion, descripciones o archivos de texto desde descripcion natural.",
        "analyze_html_js": "Analiza y depura codigo HTML/JS. Output: analisis con errores encontrados.",
        "write_file_sandboxed": "Escribe contenido en un archivo. Usa rutas relativas. Crea directorios automaticamente.",
        "read_file_sandboxed": "Lee el contenido de un archivo existente.",
        "list_files_sandboxed": "Lista archivos en un directorio.",
        "create_directory_sandboxed": "Crea un directorio (usualmente no necesario, write_file_sandboxed ya lo hace).",
        "validate_script": "Valida sintaxis de un script bash/shell. Corre sin confirmacion. Output: resultado de validacion.",
        "propose_script": "Genera un script bash desde descripcion en lenguaje natural.",
        "view_script_diff": "Muestra diferencias entre contenido actual y nuevo de un archivo.",
        "web_search_real": "Busca informacion en la web. Output: resultados de busqueda.",
        "browse_web_page": "Navega a una URL y extrae su contenido textual.",
        "extract_info_from_text": "Extrae informacion estructurada desde texto usando un schema dado.",
        "call_api_real": "Hace una llamada HTTP GET a una API externa.",
    }
    HIDDEN_TOOLS = {"fix_ui_code", "delete_file_sandboxed", "delete_directory_sandboxed",
                    "move_or_rename_sandboxed", "git_status", "git_add", "git_commit",
                    "git_diff", "git_log", "git_push", "git_pull",
                    "execute_terminal_command_safe", "apply_script",
                    "create_pdf_from_text", "generar_guia_cocoguide", "speak_text"}

    tools_desc_lines = []
    for name in sorted(tool_map.keys()):
        if name in HIDDEN_TOOLS:
            continue
        fn = tool_map[name]
        sig = inspect.signature(fn)
        params = []
        for pname, p in sig.parameters.items():
            kind = "req" if p.default is inspect.Parameter.empty else "opt"
            ptype = p.annotation if p.annotation is not inspect.Parameter.empty else "str"
            params.append(f"{pname}:{ptype}({kind})")
        param_str = ", ".join(params) if params else ""
        desc = TOOL_DESCRIPTIONS.get(name, "")
        if desc:
            tools_desc_lines.append(f"- `{name}({param_str})`: {desc}")
        else:
            tools_desc_lines.append(f"- `{name}`: {param_str}")
    tools_description = "\n".join(tools_desc_lines)

    agents_description = (
        f"- PlannerAgent_{agent_name}: Planificador\n"
        f"- UIDesignerAgent_{agent_name}: UI/Frontend\n"
        f"- DebuggerAgent_{agent_name}: Depuración\n"
        f"- DeveloperAgent_{agent_name}: Desarrollo general"
    )

    # Read user requirements
    print(
        "\nIntroduce la tarea (multilínea, 'FIN_TAREA' en línea nueva para terminar):"
    )
    lines = []
    while True:
        try:
            line = input("TAREA> ")
        except EOFError:
            print("\nEOF detectado.")
            break
        if line.strip().upper() == "FIN_TAREA":
            break
        lines.append(line)
    requirements = "\n".join(lines).strip()

    if not requirements:
        print("Sin tarea. Saliendo.")
        return

    print(f"\nTarea ({len(requirements)}c) recibida.")

    # --- Scoping Phase ---
    manifest = None
    if settings.use_scoping_agent:
        from cogniteam.scoping.agent import clarify_task
        from cogniteam.scoping.manifest import TaskManifest

        manifest: TaskManifest = clarify_task(requirements)
        requirements = manifest.clarified_task
        print(f"\nTarea clarificada ({len(requirements)}c). Iniciando ejecución...")

    # ── Snapshot de archivos antes de la ejecución ──
    import time as _time
    _timestamp = _time.strftime("%Y%m%d_%H%M%S")
    _snapshot_before = set()
    for _root, _dirs, _files in os.walk(settings.project_root):
        _rel = os.path.relpath(_root, settings.project_root)
        if _rel.startswith("proyectos_finalizados") or _rel.startswith(".venv") or _rel.startswith(".cogniteam") or _rel.startswith("__pycache__") or _rel.startswith(".git") or _rel.startswith(".venv"):
            continue
        for _f in _files:
            _snapshot_before.add(os.path.normpath(os.path.join(_rel, _f)))

    flow_result = await run_orchestrated_flow(
        requirements=requirements,
        planner_agent=planner_agent,
        planner_runner=None,
        tool_functions_map=tool_map,
        max_replanning=settings.max_replanning_attempts,
        tools_description=tools_description,
        agents_description=agents_description,
        memory_enabled=True,
    )

    # ── Generar reporte TraceForge ──
    try:
        trace_id = traceforge.get_last_trace_id()
        if trace_id:
            report_dir = Path(settings.project_root) / "proyectos_finalizados" / f"RUN_{_timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            traceforge.report(trace_id, format="html", output=str(report_dir / "traceforge_report.html"))
            traceforge.report(trace_id, format="markdown", output=str(report_dir / "traceforge_report.md"))
            print(f"  TraceForge reportes generados en {report_dir}/")
    except Exception as e:
        print(f"  TraceForge report error: {e}")

    # ── Guardar resultados en proyectos_finalizados/ ──
    _save_result(flow_result, manifest, _timestamp, _snapshot_before)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n--- Interrumpido por el usuario ---")
    except SystemExit as e:
        print(f"--- SystemExit: {e} ---")
    except Exception as e:
        print(f"--- ERROR CRÍTICO: {e} ---")
        traceback.print_exc()
    finally:
        print("--- CogniTeam finalizado ---")
