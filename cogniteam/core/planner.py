from typing import Any, Dict, Optional

from cogniteam.agents.planner_agent import generate_plan as _generate_plan
from cogniteam.agents.planner_agent import generate_plan_with_world_model as _generate_plan_with_world_model


async def generate_plan(
    planner_agent,
    requirements: str,
    session_id: str = "",
    tools_description: str = "",
    agents_description: str = "",
    replan_context: Optional[str] = None,
    planner_runner=None,
) -> Optional[Dict[str, Any]]:
    """Generate plan via direct LLM call (ADK-free)."""
    return await _generate_plan(
        planner_agent=planner_agent,
        requirements=requirements,
        tools_description=tools_description,
        agents_description=agents_description,
        replan_context=replan_context,
    )


async def generate_plan_with_world_model(
    planner_agent,
    requirements: str,
    domain: str = "",
    archetype: str = "",
    tools_description: str = "",
    agents_description: str = "",
    replan_context: Optional[str] = None,
    calibration_threshold: Optional[float] = None,
    session_id: str = "",
    planner_runner=None,
) -> Dict[str, Any]:
    """Generate plan with world model pre-check (ADK-free)."""
    return await _generate_plan_with_world_model(
        planner_agent=planner_agent,
        requirements=requirements,
        domain=domain,
        archetype=archetype,
        tools_description=tools_description,
        agents_description=agents_description,
        replan_context=replan_context,
        calibration_threshold=calibration_threshold,
    )
