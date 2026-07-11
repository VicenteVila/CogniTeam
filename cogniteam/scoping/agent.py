import json
import re
from typing import Any, Dict, List, Optional

import traceforge

from cogniteam.scoping.loader import (
    BlueprintLoader,
    get_blueprint_for_task,
    get_blueprints_for_task_multi,
    merge_blueprints,
)
from cogniteam.scoping.manifest import ClassificationInfo, TaskManifest
from cogniteam.tools.utils.llm import llm_complete


_QUESTION_GEN_PROMPT = """Eres un analista de requisitos. Dado uno o más blueprints de tarea y los parámetros ya conocidos, genera preguntas claras para obtener los parámetros que faltan.

ARQUETIPOS IDENTIFICADOS:
{archetypes_info}

TAREA ORIGINAL DEL USUARIO: "{original_task}"

PARÁMETROS YA CONOCIDOS:
{known_params}

PARÁMETROS FALTANTES (genera una pregunta para cada uno):
{missing_params}

Reglas para generar preguntas:
- Sé conciso, directo y específico del contexto de cada arquetipo.
- Si el parámetro tiene un conjunto limitado de opciones lógicas, incluye sugerencias.
- Si el usuario ya dio información relevante en la tarea original, no preguntes por ese parámetro.
- Agrupa las preguntas por arquetipo cuando tenga sentido.

Responde ÚNICAMENTE con un JSON array. Cada objeto debe tener:
  {{"parameter": "nombre", "question": "pregunta clara", "options": ["opcion1", "opcion2"] o null, "archetype": "nombre del arquetipo", "context": "por qué pregunto esto (opcional)"}}

Ejemplo:
[
  {{"parameter": "project_name", "question": "¿Cuál es el nombre del proyecto?", "options": null, "archetype": "Landing Page"}},
  {{"parameter": "business_type", "question": "¿Qué tipo de negocio es?", "options": ["SaaS", "E-commerce"], "archetype": "Landing Page"}},
  {{"parameter": "target_platforms", "question": "¿Qué plataformas móviles necesitas?", "options": ["iOS", "Android", "Ambos"], "archetype": "Mobile Apps"}}
]
"""

_CLARIFIED_TASK_PROMPT = """Eres un ingeniero de requisitos. Reescribe la tarea del usuario incorporando toda la información recopilada para que quede completamente especificada y sin ambigüedades.

TAREA ORIGINAL: "{original_task}"

ARQUETIPOS IDENTIFICADOS:
{archetypes_info}

PARÁMETROS RECOPILADOS:
{params_json}

RESTRICCIONES (reglas de todos los dominios aplicables):
{constraints}

Genera una descripción de tarea clara, completa y ejecutable que un planner IA pueda entender sin necesidad de más aclaraciones. Incluye todos los arquetipos identificados, las reglas aplicables de cada dominio, el stack tecnológico de cada arquetipo y todos los parámetros recopilados. No uses formato JSON, usa texto narrativo estructurado.
"""


def _param_name_to_human(name: str) -> str:
    s = name.replace("_", " ").replace("-", " ")
    return s[0].upper() + s[1:] if s else name or ""


def _collect_known_params(prompt: str, blueprints: List[Dict[str, Any]]) -> Dict[str, Any]:
    known = {}
    prompt_lower = prompt.lower()
    for bp in blueprints:
        for p in bp.get("required_parameters", []) + bp.get("optional_parameters", []):
            pname = p["name"]
            if pname in known:
                continue
            human = _param_name_to_human(pname)
            if human.lower() in prompt_lower:
                known[pname] = f"(posiblemente en el prompt: '{human}')"
    return known


def _ask_user(prompt_text: str, options: Optional[List[str]] = None, default: Optional[str] = None) -> str:
    from cogniteam.config.settings import settings

    if settings.auto_confirm:
        result = default or (options[0] if options else "")
        print(f"\n> {prompt_text}")
        print(f"  → {result} (auto_confirm)")
        return result

    if options:
        numbered = [f"{i+1}. {o}" for i, o in enumerate(options)]
        opts_str = " | ".join(numbered)
        prompt_text = f"{prompt_text}\n  Opciones: {opts_str}"
    if default:
        prompt_text = f"{prompt_text}\n  (default: {default})"
    prompt_text = f"\n> {prompt_text}\n  → "
    try:
        response = input(prompt_text).strip()
    except (EOFError, KeyboardInterrupt):
        return default or ""
    if not response and default:
        return default
    if options:
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(options):
                return options[idx]
        for o in options:
            if response.lower() == o.lower():
                return o
        return response
    return response


def _show_classification(blueprints: List[Dict[str, Any]]):
    print(f"\n{'='*50}")
    print(f"  [Scoping] Clasificación de la tarea")
    print(f"{'='*50}")
    for i, bp in enumerate(blueprints):
        cls = bp.get("_classification", {})
        dk = cls.get("domain_key", bp.get("domain_key", "?"))
        ak = cls.get("archetype_key", bp.get("archetype_key", "?"))
        dn = bp.get("domain_name", dk)
        an = bp.get("archetype_name", ak)
        conf = cls.get("confidence", 0)
        is_primary = cls.get("is_primary", i == 0)
        tag = "★ PRINCIPAL" if is_primary else "  SECUNDARIO"
        print(f"  {tag}")
        print(f"    Dominio:   {dn}")
        print(f"    Arquetipo: {an}")
        print(f"    Confianza: {conf:.0%}")
        print(f"    {cls.get('reasoning', '')}")
        if i < len(blueprints) - 1:
            print()
    print(f"{'='*50}")


def _confirm_classification() -> bool:
    resp = _ask_user("¿Es correcta esta clasificación?", options=["Sí", "No, quiero ajustarla"], default="Sí")
    return resp.lower() in ("sí", "si", "yes", "y", "1", "")


def _select_manual_classifications(current_blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    loader = BlueprintLoader()
    domains = loader.get_domain_list()

    print("\n  Puedes añadir/quitar arquetipos. Selecciona dominio y arquetipo.")
    print("  (escribe 'fin' para terminar de añadir)")

    selected = list(current_blueprints)

    while True:
        print(f"\n  Arquetipos actuales: {len(selected)}")
        for i, bp in enumerate(selected):
            tag = "★" if i == 0 else " "
            print(f"    {i+1}. {tag} {bp.get('domain_name', '?')} / {bp.get('archetype_name', '?')}")

        action = _ask_user("¿Añadir arquetipo, quitar uno, o continuar?", options=["Añadir", "Quitar", "Continuar"], default="Continuar")
        if action.lower() in ("continuar", "3", ""):
            break
        elif action.lower() in ("quitar", "2"):
            if len(selected) <= 1:
                print("  Debe haber al menos un arquetipo.")
                continue
            idx_str = _ask_user("Número del arquetipo a quitar", default="1")
            if idx_str.isdigit():
                idx = int(idx_str) - 1
                if 0 <= idx < len(selected):
                    removed = selected.pop(idx)
                    print(f"  Quitado: {removed.get('archetype_name', '?')}")
                    continue
            print("  Número inválido.")
        elif action.lower() in ("añadir", "add", "1"):
            print("\n  Dominios disponibles:")
            for i, d in enumerate(domains, 1):
                print(f"    {i}. {d['name']} ({d['key']})")
            d_resp = _ask_user("Selecciona un dominio (número o nombre)", default="1")
            if d_resp.isdigit():
                d_idx = int(d_resp) - 1
                if 0 <= d_idx < len(domains):
                    selected_domain = domains[d_idx]
                else:
                    print("  Opción inválida.")
                    continue
            else:
                matches = [d for d in domains if d_resp.lower() in d["name"].lower() or d_resp.lower() in d["key"].lower()]
                if len(matches) == 1:
                    selected_domain = matches[0]
                else:
                    print("  No se encontró el dominio.")
                    continue

            arch_names = selected_domain["archetypes"]
            if not arch_names:
                bp = loader.get_blueprint(selected_domain["key"], "generic-task")
                if bp:
                    bp = dict(bp)
                    bp["_classification"] = {"domain_key": selected_domain["key"], "archetype_key": "generic-task", "confidence": 1.0, "reasoning": "Selección manual", "is_primary": False}
                    selected.append(bp)
                    print("  Añadido arquetipo genérico.")
                continue

            print(f"\n  Arquetipos para {selected_domain['name']}:")
            for i, a in enumerate(arch_names, 1):
                print(f"    {i}. {a.replace('-', ' ').title()}")
            a_resp = _ask_user("Selecciona un arquetipo (número o nombre)", default="1")
            if a_resp.isdigit():
                a_idx = int(a_resp) - 1
                arch_key = arch_names[a_idx] if 0 <= a_idx < len(arch_names) else arch_names[0]
            else:
                matches = [a for a in arch_names if a_resp.lower() in a.lower()]
                arch_key = matches[0] if matches else arch_names[0]

            bp = loader.get_blueprint(selected_domain["key"], arch_key)
            if bp:
                bp = dict(bp)
                bp["_classification"] = {"domain_key": selected_domain["key"], "archetype_key": arch_key, "confidence": 1.0, "reasoning": "Selección manual", "is_primary": False}
                selected.append(bp)
                print(f"  Añadido: {selected_domain['name']} / {arch_key.replace('-', ' ').title()}")
            else:
                print("  No se encontró el blueprint.")

    return selected


@traceforge.trace(agent="scoping.questions", tags=["llm", "clarification"])
def _generate_questions(blueprints: List[Dict[str, Any]], original_prompt: str, known_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    required = []
    for bp in blueprints:
        for p in bp.get("required_parameters", []):
            if p["name"] not in known_params and p not in required:
                required.append(p)

    if not required:
        return []

    archetypes_lines = []
    for bp in blueprints:
        dk = bp.get("domain_key", "?")
        ak = bp.get("archetype_key", "?")
        dn = bp.get("domain_name", dk)
        an = bp.get("archetype_name", ak)
        priority = bp.get("priority", "")
        stack = json.dumps(bp.get("stack", {}), ensure_ascii=False)
        rules = "; ".join(bp.get("domain_rules", []))
        archetypes_lines.append(f"- {dn} / {an} (prioridad: {priority}, stack: {stack}, reglas: {rules})")
    archetypes_info = "\n".join(archetypes_lines)

    known_str = json.dumps(known_params, ensure_ascii=False, indent=2) if known_params else "(ninguno aún)"
    missing_str = json.dumps([p["name"] for p in required], ensure_ascii=False)

    prompt = _QUESTION_GEN_PROMPT.format(
        archetypes_info=archetypes_info,
        original_task=original_prompt,
        known_params=known_str,
        missing_params=missing_str,
    )

    raw = llm_complete(prompt=prompt, task="reasoning", max_tokens=2560, temperature=0.2)
    if not raw:
        return _fallback_questions(required, blueprints)

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        return _fallback_questions(required, blueprints)

    try:
        questions = json.loads(json_match.group(0))
        if not isinstance(questions, list):
            raise ValueError("Not a list")
        return questions
    except (json.JSONDecodeError, ValueError):
        return _fallback_questions(required, blueprints)


def _fallback_questions(required_params: List[Dict[str, Any]], blueprints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    archetype_map = {}
    for bp in blueprints:
        for p in bp.get("required_parameters", []):
            archetype_map[p["name"]] = bp.get("archetype_name", "")
    questions = []
    for p in required_params:
        arch = archetype_map.get(p["name"], "")
        prefix = f"[{arch}] " if arch else ""
        questions.append({
            "parameter": p["name"],
            "question": f"{prefix}¿Cuál es el valor para '{_param_name_to_human(p['name'])}'?",
            "options": None,
            "archetype": arch,
        })
    return questions


def _generate_clarified_task(blueprints: List[Dict[str, Any]], original_prompt: str, params: Dict[str, Any]) -> str:
    archetypes_lines = []
    for bp in blueprints:
        dk = bp.get("domain_key", "?")
        ak = bp.get("archetype_key", "?")
        dn = bp.get("domain_name", dk)
        an = bp.get("archetype_name", ak)
        cls = bp.get("_classification", {})
        tag = "★ PRINCIPAL" if cls.get("is_primary", False) else "  SECUNDARIO"
        priority = bp.get("priority", "")
        stack = json.dumps(bp.get("stack", {}), ensure_ascii=False)
        rules = "; ".join(bp.get("domain_rules", []))
        archetypes_lines.append(f"{tag} {dn} / {an}: prioridad={priority}, stack={stack}, reglas=[{rules}]")
    archetypes_info = "\n".join(archetypes_lines)

    params_json = json.dumps(params, ensure_ascii=False, indent=2)

    all_rules = []
    for bp in blueprints:
        for r in bp.get("domain_rules", []):
            if r not in all_rules:
                all_rules.append(r)
    constraints = "; ".join(all_rules)

    prompt = _CLARIFIED_TASK_PROMPT.format(
        original_task=original_prompt,
        archetypes_info=archetypes_info,
        params_json=params_json,
        constraints=constraints,
    )

    raw = llm_complete(prompt=prompt, task="reasoning", max_tokens=1536, temperature=0.3)
    if raw:
        return raw.strip()
    return _fallback_clarified(original_prompt, archetypes_info, params_json, constraints)


def _fallback_clarified(original_prompt: str, archetypes_info: str, params_json: str, constraints: str) -> str:
    return (
        f"Tarea: {original_prompt}\n\n"
        f"Arquetipos:\n{archetypes_info}\n\n"
        f"Parámetros:\n{params_json}\n\n"
        f"Restricciones: {constraints}"
    )


def _build_manifest(blueprints: List[Dict[str, Any]], original_prompt: str, params: Dict[str, Any], clarified_task: str) -> TaskManifest:
    primary_bp = blueprints[0] if blueprints else None
    if primary_bp is None:
        return TaskManifest(original_task=original_prompt, clarified_task=clarified_task)

    primary_cls = primary_bp.get("_classification", {})
    primary_classification = ClassificationInfo(
        domain_key=primary_bp.get("domain_key", primary_cls.get("domain_key", "")),
        domain_name=primary_bp.get("domain_name", ""),
        archetype_key=primary_bp.get("archetype_key", primary_cls.get("archetype_key", "")),
        archetype_name=primary_bp.get("archetype_name", ""),
        priority=primary_bp.get("priority", ""),
        confidence=primary_cls.get("confidence", 0),
        reasoning=primary_cls.get("reasoning", ""),
        domain_rules=primary_bp.get("domain_rules", []),
        stack=primary_bp.get("stack", {}),
        is_primary=True,
    )

    secondary = []
    for bp in blueprints[1:]:
        cls = bp.get("_classification", {})
        ci = ClassificationInfo(
            domain_key=bp.get("domain_key", cls.get("domain_key", "")),
            domain_name=bp.get("domain_name", ""),
            archetype_key=bp.get("archetype_key", cls.get("archetype_key", "")),
            archetype_name=bp.get("archetype_name", ""),
            priority=bp.get("priority", ""),
            confidence=cls.get("confidence", 0),
            reasoning=cls.get("reasoning", ""),
            domain_rules=bp.get("domain_rules", []),
            stack=bp.get("stack", {}),
            is_primary=False,
        )
        secondary.append(ci)

    all_rules = []
    for bp in blueprints:
        for r in bp.get("domain_rules", []):
            if r not in all_rules:
                all_rules.append(r)

    return TaskManifest(
        classification=primary_classification,
        secondary_classifications=secondary,
        parameters=params,
        constraints=all_rules,
        clarified_task=clarified_task,
        original_task=original_prompt,
    )


def clarify_task(original_prompt: str) -> TaskManifest:
    print(f"\n{'='*60}")
    print(f"  [Scoping Agent] Analizando tarea...")
    print(f"{'='*60}")

    # Step 1: Multi-classify
    blueprints = get_blueprints_for_task_multi(original_prompt)

    # Step 2: Show classification
    _show_classification(blueprints)

    # Step 3: Confirm with user (if primary confidence is low or generic)
    primary_cls = blueprints[0].get("_classification", {})
    primary_conf = primary_cls.get("confidence", 0)
    is_generic = (
        primary_cls.get("domain_key") == "generic"
        and primary_cls.get("archetype_key") == "generic-task"
    )

    needs_confirmation = is_generic or primary_conf < 0.5 or len(blueprints) > 1

    if needs_confirmation:
        if not _confirm_classification():
            adjusted = _select_manual_classifications(blueprints)
            if adjusted:
                blueprints = adjusted

    # Step 4: Merge and extract known params
    merged = merge_blueprints(blueprints)
    known_params = _collect_known_params(original_prompt, blueprints)

    # Step 5: Generate interview questions
    questions = _generate_questions(blueprints, original_prompt, known_params)
    if not questions:
        print("\n  [Scoping] No hay parámetros que aclarar. Tarea ya está completa.")
        clarified = _generate_clarified_task(blueprints, original_prompt, known_params)
        manifest = _build_manifest(blueprints, original_prompt, known_params, clarified)
        print(f"\n  [Scoping] Manifiesto generado.")
        return manifest

    # Step 6: Interactive Q&A
    all_params = dict(known_params)
    total = len(questions)
    print(f"\n  [Scoping] Voy a hacerte {total} pregunta(s) para asegurarme de tener todo claro:\n")

    for i, q in enumerate(questions, 1):
        param_name = q.get("parameter", f"param_{i}")
        question_text = q.get("question", f"¿Cuál es el valor para '{_param_name_to_human(param_name)}'?")
        options = q.get("options")
        arch_tag = q.get("archetype", "")

        if param_name in all_params:
            continue

        label = f"[{i}/{total}]"
        if arch_tag:
            label += f" [{arch_tag}]"
        label += f" {question_text}"

        answer = _ask_user(label, options=options)
        all_params[param_name] = answer

    # Step 7: Generate clarified task
    clarified = _generate_clarified_task(blueprints, original_prompt, all_params)

    # Step 8: Build manifest
    manifest = _build_manifest(blueprints, original_prompt, all_params, clarified)

    num_archetypes = 1 + len(blueprints) - 1
    print(f"\n  {'='*50}")
    print(f"  [Scoping] Tarea clarificada. Manifiesto generado.")
    print(f"  {num_archetypes} arquetipo(s) identificado(s).")
    print(f"  {len(all_params)} parámetros recopilados.")
    print(f"  Tarea clarificada: {clarified[:120]}...")
    print(f"  {'='*50}")

    return manifest
