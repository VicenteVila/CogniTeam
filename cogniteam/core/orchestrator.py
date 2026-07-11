import asyncio
import json
import os
import random
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import traceforge

from cogniteam.config.settings import settings
from cogniteam.core.context import StepContext
from cogniteam.core.planner import generate_plan_with_world_model
from cogniteam.core.session import create_flow_session, get_session_service


@dataclass
class CalibrationEntry:
    domain: str
    archetype: str
    predicted_confidence: float
    actual_success: bool
    brier_error: float = 0.0

    def __post_init__(self):
        outcome = 1.0 if self.actual_success else 0.0
        self.brier_error = (self.predicted_confidence - outcome) ** 2


class CalibrationStore:
    """Tabla de calibración episódica (emulación de FC-RL Brier score).
    Persistible en H-MEM / MATM / Fast-Slow de CogniTeam.
    """

    def __init__(self):
        self._history: Dict[str, List[CalibrationEntry]] = defaultdict(list)

    def _key(self, domain: str, archetype: str) -> str:
        return f"{domain}.{archetype}"

    def record(self, domain: str, archetype: str, predicted_confidence: float, actual_success: bool):
        entry = CalibrationEntry(
            domain=domain,
            archetype=archetype,
            predicted_confidence=predicted_confidence,
            actual_success=actual_success,
        )
        self._history[self._key(domain, archetype)].append(entry)

    def get_threshold(self, domain: str, archetype: str, default: float = 0.5) -> float:
        """Calcula umbral dinámico basado en historial Brier con decaimiento.

        - Si hay pocas muestras (<5): retorna default.
        - Si el Brier promedio reciente es alto (mala calibración): sube el umbral.
        - Si el Brier reciente es bajo (buena calibración) y hay éxitos recientes:
          el umbral decae gradualmente hacia default para permitir recuperación.
        """
        entries = self._history.get(self._key(domain, archetype), [])
        if len(entries) < 5:
            return default
        recent = entries[-10:]
        brier_scores = [e.brier_error for e in recent]
        avg_brier = sum(brier_scores) / len(brier_scores)
        recent_successes = sum(1 for e in recent if e.actual_success)
        success_rate = recent_successes / len(recent) if recent else 0

        if avg_brier < 0.15 and success_rate > 0.7:
            decay = max(0, (default - avg_brier) * 0.3)
            return max(default, default + decay)
        adjustment = avg_brier * 0.5
        return min(0.95, default + adjustment)

    def get_report(self, domain: str, archetype: str) -> Dict[str, Any]:
        entries = self._history.get(self._key(domain, archetype), [])
        if not entries:
            return {"count": 0}
        briers = [e.brier_error for e in entries]
        return {
            "count": len(entries),
            "successes": sum(1 for e in entries if e.actual_success),
            "failures": sum(1 for e in entries if not e.actual_success),
            "mean_brier": sum(briers) / len(briers),
            "last_threshold": self.get_threshold(domain, archetype),
        }

    def to_dict(self) -> Dict[str, list]:
        return {
            key: [
                {
                    "domain": e.domain,
                    "archetype": e.archetype,
                    "predicted_confidence": e.predicted_confidence,
                    "actual_success": e.actual_success,
                }
                for e in entries
            ]
            for key, entries in self._history.items()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, list]) -> "CalibrationStore":
        store = cls()
        for key, entries in data.items():
            for e in entries:
                entry = CalibrationEntry(
                    domain=e.get("domain", ""),
                    archetype=e.get("archetype", ""),
                    predicted_confidence=e.get("predicted_confidence", 0.5),
                    actual_success=e.get("actual_success", False),
                )
                store._history[key].append(entry)
        return store

    def save(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationStore":
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


def _resolve_tool_name(raw_name: str, available: Dict[str, Callable]) -> Optional[str]:
    if raw_name in available:
        return raw_name
    m = re.match(r"^([\w-]+)", raw_name)
    if m and m.group(1) in available:
        return m.group(1)
    return None


def _validate_output(
    output: Any, expected_format: Optional[str], tool_name: str
) -> bool:
    if not expected_format:
        return True
    expected_format = expected_format.lower()
    check = output

    # Auto-parse JSON string if json format is expected
    if expected_format == "json" and isinstance(check, str):
        import json
        try:
            check = json.loads(check)
            output = check
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(output, dict) and "success" in output and "message" in output:
        if expected_format == "tool_response_dict":
            return True
        if output.get("success"):
            data_val = output.get("data")
            check = data_val if isinstance(data_val, str) else output.get("message", "")
        else:
            check = output.get("message") or ""
    elif isinstance(output, dict) and "result" in output:
        if expected_format == "json":
            check = output
        else:
            check = output.get("result") or ""
    elif isinstance(output, dict) and "success" in output:
        if output.get("success"):
            data_val = output.get("data")
            check = data_val if isinstance(data_val, str) else output.get("message", "")

    valid = False
    if expected_format == "html":
        valid = isinstance(check, str) and (
            "<html" in check.lower() or "<!doctype html" in check.lower()
        )
    elif expected_format in ("css", "javascript", "string"):
        valid = isinstance(check, str)
    elif expected_format == "text":
        valid = isinstance(check, (str, type(None)))
    elif expected_format == "json_string":
        valid = isinstance(check, str)
        if valid:
            try:
                json.loads(check); valid = True
            except json.JSONDecodeError:
                valid = False
    elif expected_format == "json":
        valid = isinstance(check, (dict, list))
    elif expected_format == "boolean":
        valid = isinstance(check, bool)
    elif expected_format == "tool_response_dict":
        valid = isinstance(output, dict) and "success" in output
    else:
        print(f"  Formato desconocido '{expected_format}'. Omitiendo validación.")
        return True
    if not valid:
        _display = str(check)[:100] if not isinstance(check, str) else check[:100]
        print(
            f"  VALIDACIÓN: esperado '{expected_format}', recibido "
            f"'{_display}' ({type(check).__name__})"
        )
    return valid


def _get_memory_context(requirements: str) -> str:
    """Retrieve relevant context from all memory modules."""
    parts = []
    try:
        from cogniteam.memory.hmem import get_hmem
        hmem = get_hmem()
        ctx = hmem.hybrid_retrieve(requirements, top_k_temporal=5, top_k_knowledge=5)
        if ctx.get("results"):
            temporal = "\n".join(
                r["content"] for r in ctx["results"][:5] if r["type"] == "temporal"
            )
            if temporal:
                parts.append(f"[H-MEM Temporal Memory]\n{temporal[:1500]}")
    except Exception as e:
        print(f"  Memory H-MEM context error: {e}")

    try:
        from cogniteam.memory.graphrag import get_graphrag
        graphrag = get_graphrag()
        gctx = graphrag.hybrid_search(requirements, top_k_global=2, top_k_local=3)
        if gctx.get("global_communities"):
            summaries = "\n".join(
                f"- {c['summary'][:200]}" for c in gctx["global_communities"][:2]
            )
            if summaries:
                parts.append(f"[GraphRAG Community Knowledge]\n{summaries}")
    except Exception as e:
        print(f"  Memory GraphRAG context error: {e}")

    try:
        from cogniteam.memory.skills import get_skills
        skills = get_skills()
        sctx = skills.retrieve_with_memory(requirements)
        if sctx.get("relevant_skills"):
            skills_list = "\n".join(
                f"- {s['name']}: {s['description'][:100]}"
                for s in sctx["relevant_skills"][:3]
            )
            if skills_list:
                parts.append(f"[Available Skills]\n{skills_list}")
    except Exception as e:
        print(f"  Memory Skills context error: {e}")

    if parts:
        return "\n\n".join(parts)
    return ""


def _get_artifacts_summary(outputs: Dict[str, Any]) -> str:
    """Genera un resumen de artefactos para verificación de grounding."""
    parts = []
    for key, val in outputs.items():
        if isinstance(val, dict):
            content = str(val.get("result", val.get("data", val.get("message", ""))))
        elif isinstance(val, str):
            content = val
        else:
            content = str(val)
        if content and len(content) > 20:
            parts.append(f"{key}: {content[:300]}")
    return "\n".join(parts[:5])


def _run_functional_validation(project_root: str, ctx) -> None:
    """Busca archivos .html generados y ejecuta validate_html_functional en cada uno.

    Safety-net post-ejecución — solo reporta, no modifica el flujo.
    """
    import os

    from cogniteam.tools.ui.validate import validate_html_functional

    html_files = sorted(
        f for f in os.listdir(project_root)
        if f.endswith(".html") and os.path.isfile(os.path.join(project_root, f))
    )
    if not html_files:
        return

    print(f"\n--- Validación funcional (Playwright) ---")
    all_passed = True
    for fname in html_files:
        result = validate_html_functional(
            filepath=fname,
            capture_screenshot=True,
            timeout_seconds=15,
        )
        if result.get("passed"):
            print(f"  ✅ {fname}: {result.get('data', 'OK')}")
        else:
            all_passed = False
            errors = result.get("console_errors", [])
            print(f"  ❌ {fname}: {result.get('data', 'FAIL')}")
            for err in errors[:5]:
                print(f"     {err}")
            if result.get("screenshot_path"):
                print(f"     Screenshot: {result['screenshot_path']}")

    ctx.outputs["functional_validation_passed"] = all_passed
    ctx.outputs["functional_validation_summary"] = (
        "Todos los HTML pasaron validación funcional" if all_passed
        else "Algunos HTML fallaron validación funcional"
    )
    print(f"  {'✅' if all_passed else '⚠'} Validación funcional: "
          f"{'todos los HTML pasaron' if all_passed else 'algunos HTML fallaron'}")


async def run_orchestrated_flow(
    requirements: str,
    planner_agent,
    planner_runner=None,
    tool_functions_map: Dict[str, Callable] = None,
    max_replanning: int = 2,
    tools_description: str = "",
    agents_description: str = "",
    memory_enabled: bool = True,
    domain: str = "",
    archetype: str = "",
    calibration_store: Optional[CalibrationStore] = None,
    calibration_threshold: Optional[float] = None,
    debugger_agent=None,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Iniciando flujo orquestado para: '{requirements[:100]}...'")
    print(f"{'='*60}")

    # Pre-retrieve memory context to augment the planner prompt
    memory_context = ""
    if memory_enabled:
        print("[Memoria] Recuperando contexto de módulos de memoria...")
        memory_context = _get_memory_context(requirements)
        if memory_context:
            print(f"  Contexto de memoria recuperado ({len(memory_context)}c).")
            requirements = (
                f"{requirements}\n\n"
                f"--- Memory Context (use as reference) ---\n{memory_context}"
            )

    original_requirements = requirements

    start = time.time()
    flow_id = f"{int(start)}_{random.randint(1000, 9999)}"
    session_id = create_flow_session(flow_id)
    artifacts_dir = os.path.join(settings.project_root, ".cogniteam", "artifacts", flow_id)
    os.makedirs(artifacts_dir, exist_ok=True)
    print(f"  Directorio de artefactos: {artifacts_dir}")
    ctx = StepContext(requirements)

    steps: List[Dict[str, Any]] = []
    flow_failed = False
    replan_count = 0
    failed_plan_summary = ""
    world_model = None
    grounding_result = None
    execution_success = False

    while replan_count <= max_replanning:
        replan_context = None
        if replan_count > 0:
            replan_context = ctx.outputs.get("last_deviation_analysis")
            if failed_plan_summary:
                replan_context = f"{replan_context}\n\nPlan anterior fallido:\n{failed_plan_summary}"
            print(f"\n[Re-planificación #{replan_count}]")

        plan_result = await generate_plan_with_world_model(
            planner_agent=planner_agent,
            requirements=requirements,
            domain=domain,
            archetype=archetype,
            session_id=session_id,
            replan_context=replan_context,
            tools_description=tools_description,
            agents_description=agents_description,
            calibration_threshold=calibration_threshold,
        )

        wm_status = plan_result.get("wm_status", "unknown")
        if wm_status == "failed":
            print(f"  [World Model] FALLÓ la generación. Planner procede sin simulación prospectiva.")

        if plan_result.get("action") == "RECLARIFY":
            print(f"  [World Model] {plan_result['reason']}")
            return {
                "status": "RECLARIFY",
                "reason": plan_result["reason"],
                "gaps": plan_result.get("gaps", []),
                "world_model": plan_result.get("world_model"),
                "elapsed": time.time() - start,
            }

        world_model = plan_result.get("world_model")
        plan = plan_result.get("plan")

        if not plan or not isinstance(plan.get("steps"), list) or not plan["steps"]:
            print("  Plan no generado o vacío.")
            if replan_count < max_replanning:
                replan_count += 1
                ctx.outputs["last_deviation_analysis"] = (
                    "Planner no produjo JSON válido."
                )
                continue
            else:
                print("  Límite de re-planificaciones alcanzado.")
                flow_failed = True
                break

        steps = plan["steps"]
        print(
            f"\nEjecutando plan {plan.get('plan_id', 'N/A')} "
            f"({len(steps)} pasos)..."
        )

        plan_summary_parts = []
        failed_step_detail = ""

        for step_info in steps:
            if flow_failed:
                break

            if not isinstance(step_info, dict):
                msg = f"Paso no es dict: {step_info}"
                print(f"  {msg}")
                ctx.store(f"malformed_step_{random.randint(1000, 9999)}", msg, "failed")
                flow_failed = True
                ctx.outputs["last_deviation_analysis"] = msg
                break

            step_num = step_info.get("step", "?")
            tool_name = step_info.get("tool_to_use")
            raw_inputs = step_info.get("inputs", {})
            var_name = step_info.get(
                "output_variable_name", f"step_{step_num}_out"
            )
            expected_fmt = step_info.get("expected_output_format")

            print(f"\n--- [Paso {step_num}] {tool_name} -> {var_name} ---")

            result = None
            step_ok = False

            resolved = _resolve_tool_name(tool_name, tool_functions_map or {}) if tool_name else None
            if resolved and resolved != "AgentLogic":
                with traceforge.span(agent="developer.execute_step", model=tool_name, tags=["execution", tool_name or "unknown"]) as _sp:
                    processed, warnings = ctx.resolve_inputs(raw_inputs)
                    if warnings:
                        print(f"  Advertencias: {', '.join(warnings)}")

                    if resolved in ("write_file_sandboxed", "read_file_sandboxed", "list_files_sandboxed", "delete_file_sandboxed"):
                        orig_path = processed.get("relative_filepath", processed.get("relative_dirpath", ""))
                        if orig_path and not orig_path.startswith(".") and not orig_path.startswith("/"):
                            if resolved == "write_file_sandboxed":
                                processed["relative_filepath"] = os.path.relpath(
                                    os.path.join(artifacts_dir, orig_path),
                                    settings.project_root,
                                )
                            elif resolved == "read_file_sandboxed":
                                processed["relative_filepath"] = os.path.relpath(
                                    os.path.join(artifacts_dir, orig_path),
                                    settings.project_root,
                                )
                            elif resolved == "delete_file_sandboxed":
                                processed["relative_filepath"] = os.path.relpath(
                                    os.path.join(artifacts_dir, orig_path),
                                    settings.project_root,
                                )
                    elif resolved in ("combine_ui_to_html", "validate_html_functional"):
                        orig_path = processed.get("filepath", "")
                        if orig_path and not orig_path.startswith(".") and not orig_path.startswith("/"):
                            processed["filepath"] = os.path.relpath(
                                os.path.join(artifacts_dir, orig_path),
                                settings.project_root,
                            )

                    try:
                        fn = tool_functions_map[resolved]
                        if asyncio.iscoroutinefunction(fn):
                            result = await fn(**processed)
                        else:
                            result = fn(**processed)
                    except Exception as e:
                        print(f"  ERROR ejecutando '{resolved}': {e}")
                        traceback.print_exc()
                        result = {
                            "success": False,
                            "message": f"Excepción: {e}",
                            "data": None,
                        }

                    if isinstance(result, dict) and result.get("success") is False:
                        err_msg = result.get('data', result.get('message', 'error desconocido'))
                        # print full result for debugging
                        if err_msg == 'error desconocido':
                            print(f"  Fallo explícito (result completo): {str(result)[:300]}")
                        else:
                            print(f"  Fallo explícito: {err_msg}")
                        ctx.store(var_name, result, "failed_explicit")
                        step_ok = False
                        failed_step_detail = f"Paso {step_num}: {resolved} falló con: {err_msg}"
                    elif expected_fmt and not _validate_output(
                        result, expected_fmt, tool_name
                    ):
                        print(f"  Fallo de formato (esperado: {expected_fmt})")
                        ctx.store(var_name, result, "format_mismatch")
                        step_ok = False
                        failed_step_detail = f"Paso {step_num}: {resolved} devolvió formato incorrecto (esperado: {expected_fmt})"
                    else:
                        ctx.store(var_name, result, "succeeded")
                        step_ok = True
                        print(f"  OK")
                        plan_summary_parts.append(f"Paso {step_num} ({resolved}): OK")

                        if memory_enabled and result:
                            try:
                                _store_in_memory(resolved, var_name, result, requirements)
                            except Exception:
                                pass
                    _sp.set_output(str(result)[:500])
            else:
                print(
                    f"  Herramienta '{tool_name}' no encontrada o es AgentLogic."
                )
                result = {
                    "success": False,
                    "message": f"'{tool_name}' no ejecutable",
                    "data": None,
                }
                ctx.store(var_name, result, "failed")
                step_ok = False
                failed_step_detail = f"Paso {step_num}: tool '{tool_name}' no encontrada"

            if not step_ok:
                flow_failed = True
                ctx.outputs["last_deviation_analysis"] = failed_step_detail
                failed_plan_summary = "\n".join(plan_summary_parts + [f">>> FALLO: {failed_step_detail}"])
                break

        if flow_failed and replan_count < max_replanning:
            replan_count += 1
            flow_failed = False
            continue
        elif flow_failed:
            execution_success = False
            break
        else:
            print(f"\nPlan completado exitosamente.")
            # --- GROUNDING GATE ---
            if world_model and world_model.get("keywords"):
                from cogniteam.agents.debugger_agent import GroundingValidator
                artifacts = _get_artifacts_summary(ctx.outputs)
                validator = GroundingValidator(max_replans=max_replanning)
                grounding_result = validator.validate(
                    artifacts_summary=artifacts,
                    expected_keywords=world_model["keywords"],
                    attempt=replan_count + 1,
                )
                ctx.outputs["grounding_result"] = {
                    "score": grounding_result.score,
                    "verdict": grounding_result.verdict.value,
                    "found": grounding_result.found_keywords,
                    "missing": grounding_result.missing_keywords,
                }
                print(f"\n[Grounding Gate] Score: {grounding_result.score:.2f} | "
                      f"Veredicto: {grounding_result.verdict.value} | "
                      f"Faltan: {grounding_result.missing_keywords}")

                if grounding_result.should_continue():
                    execution_success = True
                    break
                elif grounding_result.should_replan():
                    replan_count += 1
                    flow_failed = False
                    feedback = (f"Grounding: score {grounding_result.score:.2f}, "
                                f"faltan keywords: {grounding_result.missing_keywords}")
                    ctx.outputs["last_deviation_analysis"] = feedback
                    failed_plan_summary = f">>> GROUNDING: {feedback}"
                    print(f"  ↻ Grounding dispara re-planificación #{replan_count}")
                    continue
                elif grounding_result.should_abort():
                    print(f"  ✗ Grounding: abortando tras {replan_count + 1} intentos")
                    execution_success = False
                    flow_failed = True
                    break
            else:
                execution_success = True
                break

    # --- POST-EXECUTION FILE VALIDATION ---
    if execution_success and world_model and world_model.get("keywords"):
        file_keywords = [kw for kw in world_model["keywords"] if "." in kw]
        if file_keywords:
            missing = []
            for fname in file_keywords:
                fpath = os.path.join(artifacts_dir, fname)
                if not os.path.isfile(fpath):
                    missing.append(fname)
            if missing:
                print(f"\n[Validación Post-Ejecución] Archivos esperados NO encontrados en {artifacts_dir}: {missing}")
                print(f"  El plan reportó éxito pero estos archivos no existen en disco.")
            else:
                print(f"\n[Validación Post-Ejecución] Archivos verificados en {artifacts_dir}: {file_keywords} ✅")

    # --- HTML QUALITY VERIFICATION (Debugger) ---
    if execution_success:
        from cogniteam.agents.debugger_agent import verify_html_quality
        quality = verify_html_quality(project_root=artifacts_dir)
        if quality["total_conflicts"] > 0:
            print(f"\n[Calidad HTML] Score: {quality['quality_score']:.2f} | Conflictos: {quality['total_conflicts']} en {len(quality['files_with_conflicts'])} archivos")
            for fwc in quality["files_with_conflicts"]:
                for c in fwc["conflicts"]:
                    print(f"  ⚠ {fwc['file']}: {c}")
        elif quality["files_checked"] > 0:
            print(f"\n[Calidad HTML] {quality['files_checked']} archivos revisados, sin conflictos ✅")

    # --- FUNCTIONAL VALIDATION (Playwright headless) ---
    if execution_success:
        _run_functional_validation(artifacts_dir, ctx)

    # --- CALIBRATION (FC-RL emulado) ---
    if calibration_store is not None and domain and archetype and world_model:
        pred_conf = world_model.get("confidence", 50) / 100.0
        calibration_store.record(domain, archetype, pred_conf, execution_success)
        new_threshold = calibration_store.get_threshold(domain, archetype)
        print(f"\n[Calibración] {domain}.{archetype} | Confianza: {pred_conf:.2f} | Éxito: {execution_success} | Umbral: {new_threshold:.2f}")

    # Save memory state
    if memory_enabled:
        try:
            _save_all_memory()
        except Exception as e:
            print(f"  Error guardando memoria: {e}")

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    if flow_failed:
        print("RESULTADO: Flujo NO completado exitosamente.")
    else:
        print("RESULTADO: Flujo completado.")
    print(f"Tiempo total: {elapsed:.2f}s")
    print(f"{'='*60}")

    result = {
        "success": not flow_failed,
        "elapsed": elapsed,
        "steps": steps,
        "plan_summary": "\n".join(plan_summary_parts) if plan_summary_parts else "",
        "outputs": dict(ctx.outputs) if hasattr(ctx, "outputs") else {},
        "world_model": world_model,
        "grounding": grounding_result,
    }
    if not flow_failed and domain and archetype:
        result["domain"] = domain
        result["archetype"] = archetype
    return result


def _store_in_memory(tool_name: str, var_name: str, result: Any, requirements: str):
    """Store tool outputs in relevant memory modules."""
    content = str(result.get("result", result.get("data", str(result))))
    try:
        from cogniteam.memory.hmem import get_hmem
        hmem = get_hmem()
        hmem.add_memory(
            content=f"[{tool_name}] {content[:500]}",
            entities=[tool_name, var_name],
            importance=0.4,
        )
    except Exception:
        pass

    try:
        from cogniteam.memory.graphrag import get_graphrag
        graphrag = get_graphrag()
        graphrag.add_text(
            f"Tool {tool_name} executed for: {requirements[:200]}\nOutput: {content[:500]}",
            source=f"{tool_name}_{var_name}",
        )
    except Exception:
        pass


def _save_all_memory():
    try:
        from cogniteam.memory.hmem import get_hmem
        hmem = get_hmem()
        hmem.consolidate()
        hmem.save()
        from cogniteam.memory.graphrag import get_graphrag
        graphrag = get_graphrag()
        graphrag.save()
        from cogniteam.memory.skills import get_skills
        skills = get_skills()
        skills.save()
        from cogniteam.memory.matm import get_matm
        matm = get_matm()
        matm.save()
        from cogniteam.memory.fastslow import get_fastslow
        fs = get_fastslow()
        fs.save()
    except Exception as e:
        print(f"  Error saving memory: {e}")
