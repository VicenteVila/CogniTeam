from typing import Callable, List


DEVELOPER_INSTRUCTION_TEMPLATE = """Eres un Desarrollador Full-Stack experto en Python.

Herramientas disponibles:
{tools_description}

REGLAS:
1. Ejecuta la ÚNICA herramienta indicada con los argumentos EXACTOS.
2. NO te desvíes. NO intentes otras acciones.
3. Después de invocar la herramienta, tu labor finaliza.
4. Output: resultado directo de la herramienta.
"""


class DeveloperAgent:
    """Developer agent — data-only container (no ADK)."""
    name: str = "DeveloperAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_developer_agent(
    agent_name: str,
    tools: List[Callable],
) -> DeveloperAgent:
    tool_names = sorted([t.__name__ for t in tools])
    instruction = DEVELOPER_INSTRUCTION_TEMPLATE.format(
        tools_description=", ".join(tool_names)
    )
    agent = DeveloperAgent(instruction=instruction)
    agent.name = agent_name
    return agent
