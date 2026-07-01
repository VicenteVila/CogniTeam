from typing import Callable, Dict, List, Optional


UI_DESIGNER_INSTRUCTION_TEMPLATE = """Eres un Diseñador y Desarrollador Frontend UI/UX experto.

Herramientas disponibles:
{tools_description}

REGLAS:
1. Ejecuta la ÚNICA herramienta indicada con los argumentos EXACTOS.
2. Produces código puro y completo, sin markdown ni explicaciones.
3. Una vez invocada la herramienta y devuelto el resultado, TU LABOR TERMINA.
4. Si generas HTML, debe ser completo (<!DOCTYPE html>).
"""


class UIDesignerAgent:
    """UI Designer agent — data-only container (no ADK)."""
    name: str = "UIDesignerAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_ui_designer_agent(
    agent_name: str,
    tools: List[Callable],
) -> UIDesignerAgent:
    tool_names = sorted([t.__name__ for t in tools])
    instruction = UI_DESIGNER_INSTRUCTION_TEMPLATE.format(
        tools_description=", ".join(tool_names)
    )
    agent = UIDesignerAgent(instruction=instruction)
    agent.name = agent_name
    return agent
