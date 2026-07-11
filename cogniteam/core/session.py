import json
import os
from pathlib import Path
from typing import Dict, Optional

from cogniteam.config.settings import settings

# Session store with optional JSON persistence
_sessions: Dict[str, Dict] = {}
_flow_sessions: Dict[str, Dict] = {}

_SESSION_FILE = Path(settings.project_root) / ".cogniteam" / "sessions.json"


def _save_sessions():
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": _sessions,
            "flow_sessions": _flow_sessions,
        }
        _SESSION_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _load_sessions():
    try:
        if _SESSION_FILE.exists():
            data = json.loads(_SESSION_FILE.read_text())
            _sessions.update(data.get("sessions", {}))
            _flow_sessions.update(data.get("flow_sessions", {}))
    except Exception:
        pass


def get_session_service() -> Dict:
    """Return the session store (compatibility shim)."""
    if not _sessions and _SESSION_FILE.exists():
        _load_sessions()
    return _sessions


def create_flow_session(flow_id: str) -> str:
    """Create and return a flow session ID."""
    if not _flow_sessions and _SESSION_FILE.exists():
        _load_sessions()
    session_id = f"flow_{settings.session_id}_{flow_id}"
    _flow_sessions[session_id] = {
        "app_name": settings.app_name,
        "user_id": settings.user_id,
        "session_id": session_id,
    }
    _save_sessions()
    return session_id


def get_flow_context(session_id: str) -> Optional[Dict]:
    """Get stored context for a flow session."""
    if not _flow_sessions and _SESSION_FILE.exists():
        _load_sessions()
    return _flow_sessions.get(session_id)


def reset_session_service():
    _sessions.clear()
    _flow_sessions.clear()
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()
