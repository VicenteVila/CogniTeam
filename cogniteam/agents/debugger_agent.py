import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import traceforge

from cogniteam.config.settings import settings
from cogniteam.tools.ui.combine import detect_html_conflicts
from cogniteam.tools.utils.llm import llm_complete


DEBUGGER_INSTRUCTION_TEMPLATE = """Eres un Debugger y Analista de Código experto.

Herramientas disponibles:
{tools_description}

MODOS:
1. EJECUTAR PASO: Si se indica una herramienta, ejecútala con args EXACTOS. Tras ejecutarla UNA VEZ, finaliza.
2. CORREGIR: Analiza info, usa herramienta para corregir.

Output: resultado de herramienta o análisis.
"""


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class GroundingResult:
    score: float
    found_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    verdict: Verdict = Verdict.FAIL
    max_attempts_reached: bool = False
    corrective_action: str = "Re-plan from scratch"

    def should_continue(self) -> bool:
        return self.verdict == Verdict.PASS and self.score >= 0.7

    def should_replan(self) -> bool:
        return self.verdict in (Verdict.FAIL, Verdict.PARTIAL) and not self.max_attempts_reached

    def should_abort(self) -> bool:
        return self.verdict in (Verdict.FAIL, Verdict.PARTIAL) and self.max_attempts_reached


GROUNDING_VERIFICATION_PROMPT = """Eres el "Grounding Verifier" dentro del Debugger Agent de CogniTeam.

Tu trabajo es verificar si los artefactos generados cumplen con los requisitos de la tarea.

Distingue entre:
- TECHNOLOGIES: cómo está construido (ej: "Vanilla JS", "Canvas API", "CSS Grid")
- DELIVERABLES: qué archivos existen (ej: "index.html", "style.css")
- STANDARDS: qué estándares cumple (ej: "accesibilidad WCAG", "responsive design")

KEYWORDS A VERIFICAR: {predicted_keywords}

RESUMEN DE ARTEFACTOS GENERADOS:
{artifacts_summary}

REGLAS:
1. Una TECHNOLOGY está "presente" si el código la usa, no si existe un archivo con ese nombre.
2. Un DELIVERABLE está "presente" si existe el archivo con el nombre y contenido esperado.
3. Un STANDARD está "presente" si el artefacto lo implementa (ej: alt text para accesibilidad).
4. Retorna grounding_score [0.0, 1.0] basado en la fracción de keywords presentes.
5. Si keywords críticas faltan, el plan no está anclado a la realidad.

Responde ÚNICAMENTE JSON válido:
{{
  "grounding_score": 0.6,
  "verdict": "pass" | "fail" | "partial",
  "found_keywords": ["keyword1", "keyword2"],
  "missing_keywords": ["keyword3"],
  "corrective_action": "Re-plan from scratch" | "Regenerate specific section" | "None needed"
}}
"""


class DebuggerAgent:
    """Debugger agent — data-only container (no ADK)."""
    name: str = "DebuggerAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_debugger_agent(
    agent_name: str,
    tools: List[Any],
) -> DebuggerAgent:
    tool_names = sorted([t.__name__ for t in tools])
    instruction = DEBUGGER_INSTRUCTION_TEMPLATE.format(
        tools_description=", ".join(tool_names)
    )
    agent = DebuggerAgent(instruction=instruction)
    agent.name = agent_name
    return agent


@traceforge.trace(agent="debugger.diagnose", tags=["grounding"])
def verify_grounding(
    predicted_keywords: List[str],
    artifacts_summary: str,
) -> Dict[str, Any]:
    """Verifica si las keywords predichas aparecen en los artefactos generados.

    Retorna dict con grounding_score, found_keywords, missing_keywords, corrective_action.
    """
    if not predicted_keywords:
        return {
            "grounding_score": 1.0,
            "found_keywords": [],
            "missing_keywords": [],
            "corrective_action": "None needed",
        }

    prompt = GROUNDING_VERIFICATION_PROMPT.format(
        predicted_keywords=json.dumps(predicted_keywords),
        artifacts_summary=artifacts_summary or "(sin resumen de artefactos)",
    )
    raw = llm_complete(
        prompt=prompt,
        task="grounding",
        max_tokens=1024,
        temperature=0.2,
        timeout_seconds=60,
    )

    if not raw:
        return _grounding_fallback(predicted_keywords, artifacts_summary)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return _grounding_fallback(predicted_keywords, artifacts_summary)

    try:
        result = json.loads(match.group())
        result.setdefault("grounding_score", 0.0)
        result.setdefault("found_keywords", [])
        result.setdefault("missing_keywords", list(predicted_keywords))
        result.setdefault("corrective_action", "Re-plan from scratch")
        return result
    except (json.JSONDecodeError, ValueError):
        return _grounding_fallback(predicted_keywords, artifacts_summary)


class GroundingValidator:
    """Gate de validación: decide si el flujo debe continuar, re-planificar o abortar."""

    def __init__(self, max_replans: int = 2):
        self.max_replans = max_replans

    @traceforge.trace(agent="grounding.gate", tags=["grounding", "gate"])
    def validate(
        self,
        artifacts_summary: str,
        expected_keywords: List[str],
        attempt: int = 1,
    ) -> GroundingResult:
        raw = verify_grounding(expected_keywords, artifacts_summary)
        if not isinstance(raw, dict):
            raw = _grounding_fallback(expected_keywords, artifacts_summary)

        score = float(raw.get("grounding_score", 0.0))
        found = raw.get("found_keywords", [])
        missing = raw.get("missing_keywords", list(expected_keywords))

        if score >= 0.7 and len(missing) == 0:
            verdict = Verdict.PASS
        elif score >= 0.4:
            verdict = Verdict.PARTIAL
        else:
            verdict = Verdict.FAIL

        return GroundingResult(
            score=score,
            found_keywords=found,
            missing_keywords=missing,
            verdict=verdict,
            max_attempts_reached=(attempt > self.max_replans),
            corrective_action=raw.get("corrective_action", "Re-plan from scratch"),
        )


def _grounding_fallback(
    predicted_keywords: List[str],
    artifacts_summary: str,
) -> Dict[str, Any]:
    """Fallback determinista: busca keywords en el texto de artefactos."""
    if not artifacts_summary:
        return {
            "grounding_score": 0.0,
            "found_keywords": [],
            "missing_keywords": list(predicted_keywords),
            "corrective_action": "Re-plan from scratch",
        }
    summary_lower = artifacts_summary.lower()
    found = [kw for kw in predicted_keywords if kw.lower() in summary_lower]
    missing = [kw for kw in predicted_keywords if kw.lower() not in summary_lower]
    score = len(found) / len(predicted_keywords) if predicted_keywords else 1.0
    action = "None needed" if score >= 0.6 else "Regenerate specific section"
    return {
        "grounding_score": round(score, 2),
        "found_keywords": found,
        "missing_keywords": missing,
        "corrective_action": action,
    }


def verify_html_quality(project_root: str = "") -> Dict[str, Any]:
    """Escanea archivos .html generados y detecta conflictos de implementación.

    Retorna dict con:
      - files_checked: cantidad de archivos revisados
      - files_with_conflicts: lista de archivos con problemas
      - total_conflicts: cantidad total de conflictos
      - quality_score: 1.0 si no hay conflictos, menor si los hay
    """
    root = project_root or settings.project_root
    html_files = []
    for f in os.listdir(root):
        if f.endswith(".html") and os.path.isfile(os.path.join(root, f)):
            html_files.append(f)

    files_with_conflicts = []
    total_conflicts = 0

    for fname in sorted(html_files):
        fpath = os.path.join(root, fname)
        try:
            content = open(fpath, "r", encoding="utf-8").read()
        except Exception:
            continue
        conflicts = detect_html_conflicts(content)
        if conflicts:
            files_with_conflicts.append({
                "file": fname,
                "conflicts": conflicts,
            })
            total_conflicts += len(conflicts)
            for c in conflicts:
                print(f"  [Debugger] {fname}: {c}")

    total_files = len(html_files)
    quality_score = 1.0
    if total_conflicts > 0:
        quality_score = max(0.0, 1.0 - (total_conflicts / max(total_files, 1) * 0.5))

    return {
        "files_checked": total_files,
        "files_with_conflicts": files_with_conflicts,
        "total_conflicts": total_conflicts,
        "quality_score": round(quality_score, 2),
    }
