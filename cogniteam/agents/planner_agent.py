import json
import random
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from cogniteam.config.settings import settings
from cogniteam.tools.utils.llm import llm_complete


PLANNER_INSTRUCTION_TEMPLATE = """Eres un Planificador Senior IA. Genera un plan JSON detallado paso a paso.

REGLAS:
1. Output ÚNICAMENTE JSON válido. Objeto raíz con clave "steps" (lista plana).
2. Cada paso tiene: "step" (int), "agent" (string), "action_description" (string), "tool_to_use" (string), "inputs" (dict), "output_variable_name" (string), "expected_output_format" (string).
3. NO te auto-asignes pasos (PlannerAgent nunca en steps).
4. Asigna tools a agentes según corresponda (los agentes se asignan en base a la tool).
5. En "inputs", usa los NOMBRES EXACTOS de los parámetros de la tool. Los parámetros req son obligatorios, opt son opcionales.
6. NO uses execute_terminal_command_safe, apply_script, delete_*, git_*, move_or_rename*. Prefiere write_file_sandboxed y validate_script.
7. IMPORTANTE: Usa SIEMPRE rutas RELATIVAS, sin "/" inicial ni "..". NO uses rutas absolutas como /tmp/ o /mnt/.
8. write_file_sandboxed ya crea directorios automáticamente. NO necesitas create_directory_sandboxed antes.
9. validate_script corre sin confirmación. Úsalo para validar sintaxis de scripts generados.

EJEMPLO de plan JSON:
{{"steps":[
  {{"step":1, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar HTML principal de la landing page", "tool_to_use":"generate_ui_code", "inputs":{{"description":"Landing page moderna con navbar, hero, galería y contacto"}}, "output_variable_name":"html_code", "expected_output_format":"html"}},
  {{"step":2, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar CSS para la landing page", "tool_to_use":"generate_css_code", "inputs":{{"description":"CSS oscuro con acentos dorados"}}, "output_variable_name":"css_code", "expected_output_format":"css"}},
  {{"step":3, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Generar JS para la landing page", "tool_to_use":"generate_js_code", "inputs":{{"description":"JS para smooth scroll y formulario de contacto"}}, "output_variable_name":"js_code", "expected_output_format":"javascript"}},
  {{"step":4, "agent":"UIDesignerAgent_CogniTeam", "action_description":"Combinar HTML, CSS y JS en un unico index.html", "tool_to_use":"combine_ui_to_html", "inputs":{{"html":"{{html_code}}", "css":"{{css_code}}", "js":"{{js_code}}", "filepath":"index.html"}}, "output_variable_name":"html_final", "expected_output_format":"html"}}
]}}

NO incluyas comentarios ni texto fuera del JSON. Output ÚNICAMENTE el objeto JSON.

Herramientas disponibles:
{tools_description}

Agentes:
{agents_description}"""


class PlannerAgent:
    """Planner agent — generates execution plans via LLM."""
    name: str = "PlannerAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_planner_agent(agent_name: str, instruction: str = "", tools=None) -> PlannerAgent:
    agent = PlannerAgent(instruction=instruction)
    agent.name = agent_name
    return agent


def _build_prompt(
    requirements: str,
    tools_description: str,
    agents_description: str,
    replan_context: Optional[str] = None,
) -> str:
    prompt = PLANNER_INSTRUCTION_TEMPLATE.format(
        tools_description=tools_description,
        agents_description=agents_description,
    )
    prompt = f"Genera un plan JSON para:\n{requirements}\n\n{prompt}"
    if replan_context:
        prompt += (
            f"\n\nContexto del intento anterior:\n{replan_context}\n"
            f"Genera un NUEVO plan CORREGIDO."
        )
    return prompt


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    if raw.lower().startswith("```json"):
        raw = raw[len("```json"):].strip()
    if raw.endswith("```"):
        raw = raw[:-len("```")].strip()
    first = raw.find("{")
    last = raw.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidate = raw[first: last + 1]
        try:
            json.loads(candidate)
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def generate_plan(
    planner_agent: PlannerAgent,
    requirements: str,
    tools_description: str = "",
    agents_description: str = "",
    replan_context: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate an execution plan via direct LLM call (no ADK)."""
    print(f"\n[Planificación] Generando plan...")

    prompt = _build_prompt(requirements, tools_description, agents_description, replan_context)

    raw_output = llm_complete(
        prompt=prompt,
        task="planning",
        max_tokens=4096,
        temperature=0.1,
        timeout_seconds=300,
    )

    if not raw_output:
        print("  ERROR: No se obtuvo respuesta del LLM.")

    plan = _extract_json(raw_output) if raw_output else None
    if plan:
        if "plan_id" not in plan:
            plan["plan_id"] = f"plan_{int(time.time())}_{random.randint(1000, 9999)}"
        print(f"  Plan generado: {plan.get('plan_id', 'N/A')}")
        print(f"  Pasos: {len(plan.get('steps', []))}")
        return plan

    print("  ERROR: No se pudo extraer JSON válido del Planner.")
    return None
