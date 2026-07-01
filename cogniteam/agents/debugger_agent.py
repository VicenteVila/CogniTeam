from typing import Callable, List


DEBUGGER_INSTRUCTION_TEMPLATE = """Eres un Debugger y Analista de Código experto.

Herramientas disponibles:
{tools_description}

MODOS:
1. EJECUTAR PASO: Si se indica una herramienta, ejecútala con args EXACTOS. Tras ejecutarla UNA VEZ, finaliza.
2. CORREGIR: Analiza info, usa herramienta para corregir.

Output: resultado de herramienta o análisis.
"""


class DebuggerAgent:
    """Debugger agent — data-only container (no ADK)."""
    name: str = "DebuggerAgent"

    def __init__(self, instruction: str = ""):
        self.instruction = instruction


def create_debugger_agent(
    agent_name: str,
    tools: List[Callable],
) -> DebuggerAgent:
    tool_names = sorted([t.__name__ for t in tools])
    instruction = DEBUGGER_INSTRUCTION_TEMPLATE.format(
        tools_description=", ".join(tool_names)
    )
    agent = DebuggerAgent(instruction=instruction)
    agent.name = agent_name
    return agent
