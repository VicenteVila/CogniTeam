from typing import Dict, Optional

from cogniteam.config.settings import settings

# Simple in-memory session store (replaces ADK InMemorySessionService)
_sessions: Dict[str, Dict] = {}
_flow_sessions: Dict[str, str] = {}


def get_session_service() -> Dict:
    """Return the session store (compatibility shim)."""
    return _sessions


def create_flow_session(flow_id: str) -> str:
    """Create and return a flow session ID."""
    session_id = f"flow_{settings.session_id}_{flow_id}"
    _flow_sessions[session_id] = {
        "app_name": settings.app_name,
        "user_id": settings.user_id,
        "session_id": session_id,
    }
    return session_id


def get_flow_context(session_id: str) -> Optional[Dict]:
    """Get stored context for a flow session."""
    return _flow_sessions.get(session_id)


def reset_session_service():
    _sessions.clear()
    _flow_sessions.clear()
