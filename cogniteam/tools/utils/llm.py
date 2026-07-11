import json
import re
import time
from typing import Any, Dict, List, Optional, Union

import traceforge

# Último proveedor/modelo que respondió (para diagnóstico)
_last_provider: str = ""
_last_model: str = ""


def get_last_provider() -> str:
    return _last_provider


def get_last_model() -> str:
    return _last_model

from cogniteam.config.settings import settings
from cogniteam.tools.utils.ratelimit import (
    check_rate_limit,
    is_circuit_open,
    record_failure,
    record_request,
    record_success,
)


# Límites por provider (free tier aprox)
_PROVIDER_LIMITS = {
    "nvidia": {"max_per_day": 5000, "max_per_minute": 40},
    "cerebras": {"max_per_day": 10000, "max_per_minute": 30},
    "mistral": {"max_per_day": 5000, "max_per_minute": 30},
}


def _provider_available(provider: str) -> bool:
    if is_circuit_open(provider):
        return False
    limits = _PROVIDER_LIMITS.get(provider)
    if limits:
        return check_rate_limit(provider, **limits)
    return True


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


def _openai_compatible_complete(
    api_key: str,
    base_url: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    provider_label: str,
) -> Optional[str]:
    """Call any OpenAI-compatible API (Cerebras, Mistral, etc.)."""
    import time as _time

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
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
        print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c, {provider_label})")
        return content.strip() or None
    except Exception as e:
        elapsed = _time.time() - t0
        print(f"  << LLM error ({provider_label}/{model_name}) tras {elapsed:.1f}s: {e}")
        return None


def _cerebras_available() -> bool:
    return settings.use_cerebras and bool(settings.cerebras_api_key) and _provider_available("cerebras")


def _cerebras_model(task: str) -> str:
    if task in ("reasoning", "planning", "world_model", "grounding", "code", "extract"):
        return settings.cerebras_model_reasoning
    return settings.cerebras_model_fast


def _cerebras_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
) -> Optional[str]:
    return _openai_compatible_complete(
        api_key=settings.cerebras_api_key,
        base_url="https://api.cerebras.ai/v1/",
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        provider_label="Cerebras",
    )


def _mistral_available() -> bool:
    return settings.use_mistral and bool(settings.mistral_api_key) and _provider_available("mistral")


def _mistral_model(task: str) -> str:
    if task in ("code", "grounding"):
        return settings.mistral_model_code
    if task in ("reasoning", "planning", "world_model"):
        return settings.mistral_model_reasoning
    return settings.mistral_model_fast


def _mistral_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
) -> Optional[str]:
    return _openai_compatible_complete(
        api_key=settings.mistral_api_key,
        base_url="https://api.mistral.ai/v1/",
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        provider_label="Mistral",
    )


def _nvidia_available() -> bool:
    return settings.use_nvidia and bool(settings.nvidia_api_key) and _provider_available("nvidia")


def _nvidia_model(task: str) -> str:
    if task in ("reasoning", "planning", "world_model", "code", "grounding"):
        return settings.nvidia_model_reasoning
    if task in ("extract", "fast", "memory"):
        return settings.nvidia_model_fast
    return settings.nvidia_model_reasoning


def _nvidia_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
) -> Optional[str]:
    return _openai_compatible_complete(
        api_key=settings.nvidia_api_key,
        base_url="https://integrate.api.nvidia.com/v1",
        model_name=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        provider_label="NVIDIA",
    )


def _google_available() -> bool:
    return settings.use_google and bool(settings.google_api_key)


def _google_model(task: str) -> str:
    if task in ("reasoning", "planning", "world_model", "code"):
        return settings.google_model_reasoning
    return settings.google_model_code


def _google_complete(
    model_name: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    timeout_seconds: int = 60,
) -> Optional[str]:
    """Call Google Gemini via google.genai SDK."""
    import time as _time

    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    t0 = _time.time()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        elapsed = _time.time() - t0
        content = response.text or ""
        print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c, Google/{model_name})")
        return content.strip() or None
    except Exception as e:
        elapsed = _time.time() - t0
        print(f"  << LLM error (Google/{model_name}) tras {elapsed:.1f}s: {e}")
        return None


def _get_primary_provider(task: str) -> Optional[str]:
    """Returns which provider should be tried first for this task, or None for Groq fallback."""
    routing = {
        "world_model": "nvidia",
        "reasoning": "nvidia",
        "planning": "mistral",
        "code": "mistral",
        "grounding": "mistral",
        "extract": "cerebras",
        "fast": "cerebras",
        "memory": "cerebras",
    }
    return routing.get(task)


def _groq_available(task: str) -> bool:
    """Check if Groq is configured and within rate limits for the given task."""
    from cogniteam.tools.utils.ratelimit import check_rate_limit

    if not settings.use_groq or not settings.groq_api_key:
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
    """Direct LLM call.

    Routing: Primary provider (Google/Mistral/Cerebras per task) → Groq → Ollama → litellm.
    Preserves free tiers: Google Flash (generous) for reasoning, Codestral only for code,
    Cerebras for fast ops, Groq/Ollama as fallback.
    """
    model_name = _resolve_model_name(task)
    with traceforge.span(agent=f"llm_{task}", model=model_name, tags=["llm_call", task, "cogniteam"]) as sp:
        t_start = time.time()
        result = _llm_complete_body(prompt, task, max_tokens, temperature, response_format, timeout_seconds)
        elapsed = time.time() - t_start
        if result:
            input_tokens = len(prompt) // 4
            output_tokens = len(result) // 4
            sp.set_tokens(input=input_tokens, output=output_tokens)
            sp.set_output(result[:500])
            return result
        sp.set_error("LLM returned None after all fallbacks")
        return None


def _llm_complete_body(
    prompt: str,
    task: str = "reasoning",
    max_tokens: int = 1024,
    temperature: float = 0.1,
    response_format: Optional[Dict[str, str]] = None,
    timeout_seconds: int = 180,
) -> Optional[str]:
    """Inner body of llm_complete (routing logic, no tracing)."""
    global _last_provider, _last_model
    import time as _time

    def _try_provider(provider: str, label: str, fn: callable, model: str) -> Optional[str]:
        global _last_provider, _last_model
        print(f"  >> LLM ({label}): {model}, prompt={len(prompt)}c, max_tokens={max_tokens}")
        result = fn(
            model_name=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        if result:
            _last_provider = provider
            _last_model = model
            record_success(provider)
            record_request(provider)
            return result
        record_failure(provider)
        print(f"  [LLM] {label} falló, siguiendo cadena...")
        return None

    # Step 1: Try primary provider for this task type
    primary = _get_primary_provider(task)
    if primary == "nvidia" and _nvidia_available():
        result = _try_provider("nvidia", "NVIDIA", _nvidia_complete, _nvidia_model(task))
        if result:
            return result
    elif primary == "google" and _google_available():
        result = _try_provider("google", "Google", _google_complete, _google_model(task))
        if result:
            return result
    elif primary == "mistral" and _mistral_available():
        result = _try_provider("mistral", "Mistral", _mistral_complete, _mistral_model(task))
        if result:
            return result
    elif primary == "cerebras" and _cerebras_available():
        result = _try_provider("cerebras", "Cerebras", _cerebras_complete, _cerebras_model(task))
        if result:
            return result

    # Step 2: Fallback — Groq
    if _groq_available(task):
        groq_model = settings.groq_model_reasoning if task in ("reasoning", "planning", "code", "extract") else settings.groq_model_fast
        print(f"  >> LLM (Groq): {groq_model}, prompt={len(prompt)}c, max_tokens={max_tokens}")
        result = _groq_complete(
            model_name=groq_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        if result:
            _last_provider = "groq"
            _last_model = groq_model
            record_success("groq")
            _record_groq()
            return result
        record_failure("groq")
        print("  [LLM] Groq falló, usando Ollama...")

    # Fallback: Ollama
    if settings.use_ollama and settings.ollama_base_url:
        raw_model = get_model_for_task(task).replace("ollama/", "")
        print(f"  >> LLM (Ollama): {raw_model}, prompt={len(prompt)}c, max_tokens={max_tokens}, timeout={timeout_seconds}s")
        result = _ollama_complete(
            model_name=raw_model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        if result:
            _last_provider = "ollama"
            _last_model = raw_model
            return result

    # Fallback: litellm (legacy)
    import litellm

    model = get_model_for_task(task)
    display_model = model.replace("ollama/", "")
    print(f"  >> LLM (litellm): {display_model}, prompt={len(prompt)}c, max_tokens={max_tokens}, timeout={timeout_seconds}s")
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
            _last_provider = "litellm"
            _last_model = model
            print(f"  << LLM respuesta en {elapsed:.1f}s ({len(content)}c)")
            return content.strip() or None
        print(f"  << LLM respuesta vacía en {elapsed:.1f}s")
    except Exception as e:
        print(f"  << LLM error ({display_model}) tras {_time.time()-t0:.1f}s: {e}")
    return None


_CTRL_CHAR_RE = re.compile(r"[\n\r\t\b\f]")

def _escape_control_chars_in_match(match_obj: re.Match) -> str:
    s = match_obj.group(0)
    if not (s.startswith('"') and s.endswith('"')):
        return s
    content = s[1:-1]
    control_char_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}
    def replace_char(char_match):
        return control_char_map.get(char_match.group(0), char_match.group(0))
    return f'"{_CTRL_CHAR_RE.sub(replace_char, content)}"'


def _resolve_model_name(task: str) -> str:
    """Resuelve qué modelo se usará según el routing interno de llm_complete."""
    primary = _get_primary_provider(task)
    if primary == "nvidia" and _nvidia_available():
        return _nvidia_model(task)
    elif primary == "google" and _google_available():
        return _google_model(task)
    elif primary == "mistral" and _mistral_available():
        return _mistral_model(task)
    elif primary == "cerebras" and _cerebras_available():
        return _cerebras_model(task)
    elif _groq_available(task):
        return settings.groq_model_reasoning if task in ("reasoning", "planning", "code", "extract") else settings.groq_model_fast
    elif settings.use_ollama and settings.ollama_base_url:
        return get_model_for_task(task).replace("ollama/", "")
    return "litellm-fallback"


def sanitize_json_string_for_control_chars(json_string: str) -> str:
    if not json_string or not isinstance(json_string, str):
        return json_string
    json_string_regex = r'"(?>\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})|[^"\\\0-\x1F\x7F]+)*"'
    try:
        return re.sub(json_string_regex, _escape_control_chars_in_match, json_string)
    except Exception as e:
        print(f"  sanitize_json_string_for_control_chars: {e}")
        return json_string
