import json
import re
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from cogniteam.config.settings import settings


def get_model_for_task(task: str = "reasoning") -> str:
    """Get the appropriate Ollama model for a task type."""
    if not settings.use_ollama:
        return settings.model_name
    model_map = {
        "reasoning": settings.ollama_model_reasoning,
        "planning": settings.ollama_model_fast,
        "fast": settings.ollama_model_fast,
        "extract": settings.ollama_model_fast,
        "code": settings.ollama_model_code,
        "memory": settings.ollama_model_fast,
    }
    model = model_map.get(task, settings.ollama_model_reasoning)
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    return model


def get_litellm_model_name(model_name_str: str, ollama_model_name: Optional[str] = None) -> str:
    if model_name_str.startswith("gemini-") and not model_name_str.startswith("gemini/"):
        return f"gemini/{model_name_str}"
    if model_name_str.startswith("models/") and not model_name_str.startswith("gemini/"):
        name_part = model_name_str[len("models/"):]
        return f"gemini/{name_part}" if not name_part.startswith("gemini/") else name_part
    if model_name_str.startswith("ollama/"):
        return model_name_str
    if (
        ollama_model_name
        and model_name_str == ollama_model_name
        and not model_name_str.startswith("ollama/")
    ):
        return f"ollama/{model_name_str}"
    if settings.use_ollama and not model_name_str.startswith("gemini/"):
        return f"ollama/{model_name_str}"
    return model_name_str


def get_genai_model_name(model_name_str: str) -> str:
    if model_name_str.startswith("gemini/"):
        return model_name_str.split("gemini/", 1)[1]
    if model_name_str.startswith("models/"):
        if "gemini-" in model_name_str:
            return model_name_str.split("models/", 1)[1]
    return model_name_str


def _ollama_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 180,
) -> Optional[str]:
    """Call Ollama directly via HTTP, bypassing litellm compatibility issues."""
    import json as _json
    import time as _time
    import urllib.request as _request

    base = settings.ollama_base_url.rstrip("/")
    url = f"{base}/api/generate"
    payload = _json.dumps({
        "model": model_name,
        "system": "No pienses en voz alta. No muestres tu razonamiento interno. Responde directa y exclusivamente lo que se te pide.",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }).encode("utf-8")

    t0 = _time.time()
    try:
        req = _request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        resp = _request.urlopen(req, timeout=timeout_seconds)
        data = _json.loads(resp.read().decode("utf-8"))
        content = (data.get("response") or "").strip()
        if not content:
            content = (data.get("thinking") or "").strip()
        elapsed = _time.time() - t0
        display = content[:80]
        print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c): {display}...")
        return content or None
    except Exception as e:
        elapsed = _time.time() - t0
        print(f"  << LLM error tras {elapsed:.1f}s: {e}")
        return None


def _groq_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
) -> Optional[str]:
    """Call Groq cloud API."""
    import time as _time

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    t0 = _time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_seconds,
        )
        elapsed = _time.time() - t0
        content = response.choices[0].message.content or ""
        print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c, Groq)")
        return content.strip() or None
    except Exception as e:
        elapsed = _time.time() - t0
        print(f"  << LLM error (Groq/{model_name}) tras {elapsed:.1f}s: {e}")
        return None


def _groq_available(task: str) -> bool:
    """Check if Groq is configured and within rate limits for the given task."""
    from cogniteam.tools.utils.ratelimit import check_rate_limit

    if not settings.use_groq or not settings.groq_api_key:
        return False
    if task == "code":
        return False
    return check_rate_limit("groq", max_per_day=14000, max_per_minute=30)


def _record_groq():
    from cogniteam.tools.utils.ratelimit import record_request
    record_request("groq")


def llm_complete(
    prompt: str,
    task: str = "reasoning",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    response_format: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 180,
) -> Optional[str]:
    """Direct LLM call: Groq (if configured & within limits) → Ollama → litellm fallback."""
    import time as _time

    # Try Groq for non-code tasks first
    if _groq_available(task):
        groq_model = settings.groq_model_reasoning if task == "reasoning" else settings.groq_model_fast
        print(f"  >> LLM (Groq): {groq_model}, prompt={len(prompt)}c, max_tokens={max_tokens}")
        result = _groq_complete(
            model_name=groq_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        if result:
            _record_groq()
            return result
        print("  [LLM] Groq falló, usando Ollama...")

    # Fallback: Ollama
    if settings.use_ollama and settings.ollama_base_url:
        raw_model = get_model_for_task(task).replace("ollama/", "")
        print(f"  >> LLM: {raw_model}, prompt={len(prompt)}c, max_tokens={max_tokens}, timeout={timeout_seconds}s")
        return _ollama_complete(
            model_name=raw_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )

    # Fallback: litellm (legacy)
    import litellm

    model = get_model_for_task(task)
    display_model = model.replace("ollama/", "")
    print(f"  >> LLM: {display_model}, prompt={len(prompt)}c, max_tokens={max_tokens}, timeout={timeout_seconds}s")
    t0 = _time.time()

    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_seconds,
    )
    if response_format:
        kwargs["response_format"] = response_format
    try:
        response = litellm.completion(**kwargs)
        elapsed = _time.time() - t0
        if response and response.choices and response.choices[0].message:
            content = response.choices[0].message.content or ""
            print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c)")
            return content.strip() or None
        print(f"  << LLM respuesta vacía en {elapsed:.1f}s")
    except Exception as e:
        print(f"  << LLM error ({display_model}) tras {_time.time()-t0:.1f}s: {e}")
    return None


def _escape_control_chars_in_match(match_obj: re.Match) -> str:
    s = match_obj.group(0)
    if not (s.startswith('"') and s.endswith('"')):
        return s
    content = s[1:-1]
    control_char_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
    def replace_char(char_match):
        return control_char_map.get(char_match.group(0), char_match.group(0))
    return f'"{re.compile(r"[\n\r\t\b\f]").sub(replace_char, content)}"'


def sanitize_json_string_for_control_chars(json_string: str) -> str:
    if not json_string or not isinstance(json_string, str):
        return json_string
    json_string_regex = r'"(?>\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})|[^"\\\0-\x1F\x7F]+)*"'
    try:
        return re.sub(json_string_regex, _escape_control_chars_in_match, json_string)
    except Exception as e:
        print(f"  sanitize_json_string_for_control_chars: {e}")
        return json_string
