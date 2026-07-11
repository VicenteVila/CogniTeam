import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import traceforge

from cogniteam.config.settings import settings
from cogniteam.tools.utils.llm import llm_complete


ARCHETYPE_BLUEPRINT_INDEX = "archetype_blueprint_index"

GENERIC_BLUEPRINT = {
    "domain_key": "generic",
    "domain_name": "Generic / Sin clasificar",
    "archetype_key": "generic-task",
    "archetype_name": "Tarea genérica",
    "priority": "clarification_and_delivery",
    "domain_rules": [
        "No asumir información no explícita.",
        "Preferir formatos de salida estándar (JSON, Markdown, texto plano).",
    ],
    "required_parameters": [
        {"name": "task_goal", "question": "¿Cuál es el objetivo exacto de la tarea?", "type": "string"},
        {"name": "input_source", "question": "¿Qué datos o recursos de entrada están disponibles?", "type": "string"},
        {"name": "output_format", "question": "¿Qué formato debe tener el resultado?", "type": "choice", "options": ["json", "markdown", "text", "script", "html"]},
    ],
    "optional_parameters": [
        {"name": "constraints", "question": "¿Hay restricciones a considerar?", "type": "string"},
        {"name": "deadline", "question": "¿Hay fecha límite?", "type": "string"},
    ],
    "stack": {},
}


class BlueprintLoader:
    _instance: Optional["BlueprintLoader"] = None
    _data: Optional[Dict[str, Any]] = None
    _blueprints: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "BlueprintLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._data is not None:
            return
        self._load()

    def _load(self) -> None:
        root = Path(settings.project_root)
        candidates = [
            root / "antigravity_complete_system_7_domains.json",
            root / "config" / "antigravity_complete_system_7_domains.json",
        ]
        loaded = None
        for path in candidates:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                break
        if loaded is None:
            print("  [Scoping] Blueprint JSON no encontrado. Usando solo plantilla genérica.")
            loaded = {"dominios": {}}
        self._data = loaded
        self._build_index()

    def _build_index(self) -> None:
        self._blueprints = {}
        dominios = self._data.get("dominios", {})
        for dk, dv in dominios.items():
            domain_name = dv.get("domain_rules", [""])[0:1]
            domain_rules = dv.get("domain_rules", [])
            archetypes = dv.get("archetypes", {})
            for ak, av in archetypes.items():
                bp = self._archetype_to_blueprint(dk, dv, ak, av, domain_rules)
                self._blueprints[f"{dk}.{ak}"] = bp
        self._blueprints["generic.generic-task"] = dict(GENERIC_BLUEPRINT)

    def _archetype_to_blueprint(
        self, domain_key: str, domain_data: dict, archetype_key: str,
        archetype_data: dict, domain_rules: List[str],
    ) -> Dict[str, Any]:
        domain_name = domain_key.replace("_", " ").title()
        archetype_name = archetype_data.get("_key", archetype_key.replace("-", " ").title())
        required_raw = archetype_data.get("required_parameters", [])
        optional_raw = archetype_data.get("optional_parameters", [])
        required = [self._param_to_field(p, True) for p in required_raw]
        optional = [self._param_to_field(p, False) for p in optional_raw]
        return {
            "domain_key": domain_key,
            "domain_name": domain_name,
            "archetype_key": archetype_key,
            "archetype_name": archetype_name,
            "priority": archetype_data.get("priority", ""),
            "domain_rules": domain_rules or [],
            "stack": archetype_data.get("stack", {}),
            "keywords": archetype_data.get("keywords", []),
            "required_parameters": required,
            "optional_parameters": optional,
        }

    def _param_to_field(self, raw: str, required: bool) -> Dict[str, Any]:
        name = raw if isinstance(raw, str) else raw.get("name", str(raw))
        question = raw.get("question", f"¿Cuál es el valor para '{name}'?") if isinstance(raw, dict) else f"¿Cuál es el valor para '{name}'?"
        return {
            "name": name,
            "question": question,
            "required": required,
            "type": "string",
        }

    def get_all_blueprints(self) -> Dict[str, Any]:
        return dict(self._blueprints)

    def get_blueprint(self, domain_key: str, archetype_key: str) -> Optional[Dict[str, Any]]:
        return self._blueprints.get(f"{domain_key}.{archetype_key}")

    def get_generic_blueprint(self) -> Dict[str, Any]:
        return dict(GENERIC_BLUEPRINT)

    def get_domain_list(self) -> List[Dict[str, str]]:
        dominios = self._data.get("dominios", {})
        result = []
        for dk, dv in dominios.items():
            dn = dk.replace("_", " ").title()
            archetypes = list(dv.get("archetypes", {}).keys())
            result.append({"key": dk, "name": dn, "archetypes": archetypes})
        return result

    def get_raw_data(self) -> Dict[str, Any]:
        return dict(self._data) if self._data else {}


_CLASSIFICATION_PROMPT = """Clasifica esta tarea en un dominio+arquetipo.

DOMINIOS:
{domain_list}

TAREA: "{task}"

Responde solo JSON: {{"domain_key":"","archetype_key":"","confidence":0.0,"reasoning":""}}
Si no encaja: "generic.generic-task".
"""


def _domain_line(d: dict, loader: BlueprintLoader) -> str:
    arch_parts = []
    for ak in d["archetypes"]:
        bp = loader.get_blueprint(d["key"], ak)
        kw = bp.get("keywords", []) if bp else []
        if kw:
            arch_parts.append(f"{ak} ({', '.join(kw[:4])})")
        else:
            arch_parts.append(ak)
    arch_list = ", ".join(arch_parts)
    return f"  {d['key']} ({d['name']}): {arch_list}"


@traceforge.trace(agent="scoping.classify", tags=["llm", "classification"])
def classify_task(prompt: str) -> Tuple[str, str, float, str]:
    """Returns (domain_key, archetype_key, confidence, reasoning)."""
    loader = BlueprintLoader()
    domains = loader.get_domain_list()
    domain_lines = [_domain_line(d, loader) for d in domains]
    domain_list_str = "\n".join(domain_lines)

    full_prompt = _CLASSIFICATION_PROMPT.format(domain_list=domain_list_str, task=prompt)
    raw = llm_complete(prompt=full_prompt, task="extract", max_tokens=512, temperature=0.1, timeout_seconds=120)

    if not raw:
        return "generic", "generic-task", 0.0, "No se obtuvo respuesta del LLM"

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        return "generic", "generic-task", 0.0, "No se pudo extraer JSON de la respuesta"

    try:
        result = json.loads(json_match.group(0))
        dk = result.get("domain_key", "generic")
        ak = result.get("archetype_key", "generic-task")
        conf = float(result.get("confidence", 0.0))
        reason = result.get("reasoning", "")
        return dk, ak, conf, reason
    except (json.JSONDecodeError, ValueError, TypeError):
        return "generic", "generic-task", 0.0, "Error parseando JSON de clasificación"


_MULTI_CLASSIFICATION_PROMPT = """Eres un clasificador de tareas multidominio. Identifica hasta 3 arquetipos que cubran los distintos aspectos de la tarea del usuario.

DOMINIOS DISPONIBLES:
{domain_list}

TAREA: "{task}"

Analiza la tarea. Muchas tareas reales abarcan múltiples dominios. Identifica los arquetipos más relevantes.

Responde solo JSON:
[{{"domain_key":"","archetype_key":"","confidence":0.0,"reasoning":"","is_primary":true/falso}}]

- Máximo 3. El primero es el PRINCIPAL. Si no encaja, usa "generic.generic-task".
"""


def classify_task_multi(prompt: str) -> List[Dict[str, Any]]:
    """Returns a list of classification dicts: [{domain_key, archetype_key, confidence, reasoning, is_primary}, ...]."""
    loader = BlueprintLoader()
    domains = loader.get_domain_list()
    domain_lines = [_domain_line(d, loader) for d in domains]
    domain_list_str = "\n".join(domain_lines)

    full_prompt = _MULTI_CLASSIFICATION_PROMPT.format(domain_list=domain_list_str, task=prompt)
    print(f"  [Scoping] Clasificando en {len(domains)} dominios ({len(full_prompt)}c)...")
    raw = llm_complete(prompt=full_prompt, task="extract", max_tokens=1024, temperature=0.1, timeout_seconds=120)

    if not raw:
        print(f"  [Scoping] ⚠ LLM devolvió vacío. Usando fallback genérico.")
        return [{"domain_key": "generic", "archetype_key": "generic-task", "confidence": 0.0, "reasoning": "No se obtuvo respuesta del LLM", "is_primary": True}]

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            single = json.loads(json_match.group(0))
            return [{
                "domain_key": single.get("domain_key", "generic"),
                "archetype_key": single.get("archetype_key", "generic-task"),
                "confidence": float(single.get("confidence", 0.0)),
                "reasoning": single.get("reasoning", ""),
                "is_primary": True,
            }]
        return [{"domain_key": "generic", "archetype_key": "generic-task", "confidence": 0.0, "reasoning": "No se pudo extraer JSON", "is_primary": True}]

    try:
        results = json.loads(json_match.group(0))
        if not isinstance(results, list):
            results = [results]
        for r in results:
            r["confidence"] = float(r.get("confidence", 0.0))
            r.setdefault("is_primary", False)
        if not any(r.get("is_primary") for r in results):
            results[0]["is_primary"] = True
        primary = [r for r in results if r.get("is_primary")]
        secondary = [r for r in results if not r.get("is_primary") and r.get("confidence", 0) >= 0.35]
        results = primary + secondary
        return results[:3]
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return [{"domain_key": "generic", "archetype_key": "generic-task", "confidence": 0.0, "reasoning": "Error parseando JSON", "is_primary": True}]


def get_blueprint_for_task(prompt: str) -> Dict[str, Any]:
    """Classify the task and return the matching blueprint (or generic)."""
    dk, ak, conf, reason = classify_task(prompt)
    loader = BlueprintLoader()
    bp = loader.get_blueprint(dk, ak)
    if bp is None:
        bp = loader.get_generic_blueprint()
    bp["_classification"] = {
        "domain_key": dk,
        "archetype_key": ak,
        "confidence": conf,
        "reasoning": reason,
    }
    blueprints = get_blueprints_for_task_multi(prompt)
    if len(blueprints) > 1:
        bp["_multi_classifications"] = [b["_classification"] for b in blueprints]
    return bp


def get_blueprints_for_task_multi(prompt: str) -> List[Dict[str, Any]]:
    """Classify the task across multiple archetypes and return their blueprints."""
    classifications = classify_task_multi(prompt)
    loader = BlueprintLoader()
    result = []
    for c in classifications:
        dk = c["domain_key"]
        ak = c["archetype_key"]
        bp = loader.get_blueprint(dk, ak)
        if bp is None:
            bp = loader.get_generic_blueprint()
        bp = dict(bp)  # shallow copy
        bp["_classification"] = c
        result.append(bp)
    return result


def merge_blueprints(blueprints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple blueprints into one, deduplicating parameters and combining rules."""
    if not blueprints:
        return dict(GENERIC_BLUEPRINT)
    if len(blueprints) == 1:
        return dict(blueprints[0])

    primary = blueprints[0]
    domain_keys = set()
    archetype_keys = set()
    all_domain_rules = list(primary.get("domain_rules", []))
    seen_params = set()
    merged_required = []
    merged_optional = []

    for bp in blueprints:
        dk = bp.get("domain_key", "")
        ak = bp.get("archetype_key", "")
        domain_keys.add(dk)
        archetype_keys.add(ak)

        for rule in bp.get("domain_rules", []):
            if rule not in all_domain_rules:
                all_domain_rules.append(rule)

        for p in bp.get("required_parameters", []):
            if p["name"] not in seen_params:
                seen_params.add(p["name"])
                merged_required.append(p)

        for p in bp.get("optional_parameters", []):
            if p["name"] not in seen_params:
                seen_params.add(p["name"])
                merged_optional.append(p)

    merged = dict(primary)
    merged["domain_key"] = " + ".join(sorted(domain_keys))
    merged["domain_name"] = " + ".join(sorted(set(bp.get("domain_name", "") for bp in blueprints)))
    merged["archetype_key"] = " + ".join(sorted(archetype_keys))
    merged["archetype_name"] = " + ".join(sorted(set(bp.get("archetype_name", "") for bp in blueprints)))
    merged["domain_rules"] = all_domain_rules
    merged["required_parameters"] = merged_required
    merged["optional_parameters"] = merged_optional
    merged["_multi"] = True
    return merged
