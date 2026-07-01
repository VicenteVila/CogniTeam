import json
import os
import time
from pathlib import Path
from typing import Dict, Optional


_STATE_DIR = Path.home() / ".cogniteam"


def _ensure_state_dir():
    _STATE_DIR.mkdir(parents=True, exist_ok=True)


def _state_path(service: str) -> Path:
    return _STATE_DIR / f"{service}_state.json"


def _load_state(service: str) -> Dict:
    path = _state_path(service)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"requests_today": 0, "date": "", "minute_log": []}


def _save_state(service: str, state: Dict):
    _ensure_state_dir()
    _state_path(service).write_text(json.dumps(state, indent=2))


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def check_rate_limit(
    service: str,
    max_per_day: int = 14000,
    max_per_minute: int = 30,
) -> bool:
    state = _load_state(service)
    today = _today()

    if state.get("date") != today:
        state["date"] = today
        state["requests_today"] = 0
        state["minute_log"] = []

    now = time.time()
    state["minute_log"] = [t for t in state["minute_log"] if now - t < 60]

    if state["requests_today"] >= max_per_day:
        return False
    if len(state["minute_log"]) >= max_per_minute:
        return False

    return True


def record_request(service: str):
    state = _load_state(service)
    today = _today()

    if state.get("date") != today:
        state["date"] = today
        state["requests_today"] = 0
        state["minute_log"] = []

    state["requests_today"] = state.get("requests_today", 0) + 1
    state["minute_log"] = state.get("minute_log", []) + [time.time()]

    _save_state(service, state)


def get_rate_status(service: str) -> Dict:
    state = _load_state(service)
    today = _today()

    if state.get("date") != today:
        return {"available": True, "used_today": 0, "used_this_minute": 0}

    now = time.time()
    recent = [t for t in state.get("minute_log", []) if now - t < 60]

    return {
        "available": True,
        "used_today": state.get("requests_today", 0),
        "used_this_minute": len(recent),
    }
