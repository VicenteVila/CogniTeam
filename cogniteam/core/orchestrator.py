import asyncio
import json
import random
import re
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

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
        """Calcula umbral dinámico basado en historial Brier."""
        entries = self._history.get(self._key(domain, archetype), [])
        if len(entries) < 5:
            return default
        brier_scores = [e.brier_error for e in entries[-20:]]
        avg_brier = sum(brier_scores) / len(brier_scores)
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
    m = re.match(r"^(\w+)", raw_name)
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
            check = output.get("data") or output.get("message") or ""
        else:
            check = output.get("message") or ""
    elif isinstance(output, dict) and "result" in output:
        if expected_format == "json":
            check = output
        else:
            check = output.get("result") or ""
    elif isinstance(output, dict) and "success" in output:
        if output.get("success"):
            check = output.get("data") or output.get("message") or ""

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

    start = time.time()
    flow_id = f"{int(start)}_{random.randint(1000, 9999)}"
    session_id = create_flow_session(flow_id)
    ctx = StepContext(requirements)
    original_requirements = requirements

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
                processed, warnings = ctx.resolve_inputs(raw_inputs)
                if warnings:
                    print(f"  Advertencias: {', '.join(warnings)}")

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
                    err_msg = result.get('message', 'error desconocido')
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
            execution_success = True
            break

    # --- GROUNDING VERIFICATION (Debugger) ---
    if execution_success and world_model and world_model.get("keywords"):
        artifacts = _get_artifacts_summary(ctx.outputs)
        if debugger_agent is not None and hasattr(debugger_agent, "verify_grounding"):
            try:
                grounding_result = debugger_agent.verify_grounding(
                    world_model["keywords"], artifacts
                )
            except Exception:
                from cogniteam.agents.debugger_agent import verify_grounding
                grounding_result = verify_grounding(world_model["keywords"], artifacts)
        else:
            from cogniteam.agents.debugger_agent import verify_grounding
            grounding_result = verify_grounding(world_model["keywords"], artifacts)

        if grounding_result:
            gs = grounding_result.get("grounding_score", 0)
            print(f"\n[Grounding] Score: {gs:.2f} | Encontradas: {grounding_result.get('found_keywords', [])}")
            if gs < 0.6:
                print(f"  [Grounding] Keywords faltantes: {grounding_result.get('missing_keywords', [])}")
                print(f"  [Grounding] Acción correctiva: {grounding_result.get('corrective_action', 'N/A')}")

    # --- POST-EXECUTION FILE VALIDATION ---
    if execution_success and world_model and world_model.get("keywords"):
        import os
        file_keywords = [kw for kw in world_model["keywords"] if "." in kw]
        if file_keywords:
            missing = []
            for fname in file_keywords:
                fpath = os.path.join(settings.project_root, fname)
                if not os.path.isfile(fpath):
                    missing.append(fname)
            if missing:
                print(f"\n[Validación Post-Ejecución] Archivos esperados NO encontrados: {missing}")
                print(f"  El plan reportó éxito pero estos archivos no existen en disco.")
            else:
                print(f"\n[Validación Post-Ejecución] Archivos verificados en disco: {file_keywords} ✅")

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
