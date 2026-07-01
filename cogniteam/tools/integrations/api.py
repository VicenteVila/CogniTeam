import json
import traceback
from typing import Any, Dict, Optional

import httpx

from cogniteam.tools.utils.retry import sync_retry_with_backoff
from cogniteam.utils.io import ask_confirmation
from cogniteam.utils.security import validate_url


@sync_retry_with_backoff
def call_api_real(
    url: str,
    method: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
) -> str:
    method_upper = (method or "GET").upper()
    print(f"\n-- [call_api_real] {method_upper} {url}")

    if not validate_url(url):
        return f"Error: URL inválida: '{url}'."

    prompt = f"Llamada API:\n  Método: {method_upper}\n  URL: {url}"
    if headers:
        prompt += f"\n  Headers: {json.dumps(headers, indent=1, ensure_ascii=False)}"
    if json_data:
        prompt += f"\n  Body: {json.dumps(json_data, indent=1, ensure_ascii=False)}"
    if not ask_confirmation(prompt):
        return "Cancelado por el usuario."

    try:
        with httpx.Client(timeout=httpx.Timeout(10.0, read=20.0)) as client:
            response = client.request(
                method_upper, url, headers=headers, json=json_data
            )
            response.raise_for_status()
        try:
            return json.dumps(response.json(), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = response.text
            return text[:2000] + ("..." if len(text) > 2000 else "")
    except httpx.HTTPStatusError as e:
        detail = f"Error HTTP {e.response.status_code}"
        try:
            detail += f": {json.dumps(e.response.json(), ensure_ascii=False)}"
        except json.JSONDecodeError:
            detail += f": {e.response.text[:500]}"
        return detail
    except httpx.TimeoutException:
        return f"Timeout llamando a {method_upper} {url}."
    except httpx.RequestError as e:
        return f"Error de red en {method_upper} {url}: {e}"
    except Exception as e:
        print(f"ERROR call_api_real: {e}")
        traceback.print_exc()
        return f"Error inesperado: {e}"
