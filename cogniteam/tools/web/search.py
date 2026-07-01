import os
from typing import Any, Dict, Optional

from cogniteam.tools.utils.retry import sync_retry_with_backoff
from cogniteam.tools.web._brave import brave_search


@sync_retry_with_backoff()
def web_search_real(query: str, num_results: Optional[int] = None) -> Dict[str, str]:
    num_results_actual = num_results if num_results is not None and 1 <= num_results <= 10 else 5
    print(f"\n-- [web_search_real] Buscando: '{query}' ({num_results_actual} resultados)")

    items = brave_search(query, num_results_actual)
    if items is None:
        from cogniteam.tools.web._cse import cse_api_key, cse_cx
        if not cse_api_key or not cse_cx:
            return {"result": "Error: No hay API key de Brave ni Google CSE configuradas."}
        return _google_cse_search(query, num_results_actual)

    if not items:
        return {"result": "No se encontraron resultados."}
    result_parts = ["Resultados de búsqueda (Brave):"]
    for i, item in enumerate(items):
        title = item.get("title", "N/A")
        link = item.get("url", "N/A")
        snippet = item.get("snippet", "N/A").replace(os.linesep, " ").replace("\n", " ")
        result_parts.append(f"{i+1}. {title}\n   URL: {link}\n   Snippet: {snippet}\n")
    return {"result": "\n".join(result_parts)}


def _google_cse_search(query: str, num_results: int) -> Dict[str, str]:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from cogniteam.tools.web._cse import google_search_service, cse_api_key, cse_cx

    print("  (usando Google CSE como fallback)")
    if not google_search_service:
        return {"result": "Error: Servicio Google Custom Search no configurado."}
    if not cse_api_key or not cse_cx:
        return {"result": "Error: Faltan credenciales CSE."}
    try:
        res = google_search_service.cse().list(q=query, cx=cse_cx, num=num_results).execute()
        items = res.get("items", [])
        if not items:
            return {"result": "No se encontraron resultados."}
        result_parts = ["Resultados de búsqueda (Google):"]
        for i, item in enumerate(items):
            title = item.get("title", "N/A")
            link = item.get("link", "N/A")
            snippet = item.get("snippet", "N/A").replace(os.linesep, " ").replace("\n", " ")
            result_parts.append(f"{i+1}. {title}\n   URL: {link}\n   Snippet: {snippet}\n")
        return {"result": "\n".join(result_parts)}
    except HttpError as e:
        if hasattr(e, "resp") and e.resp.status == 403:
            return {"result": "Error 403: Acceso denegado o cuota excedida."}
        return {"result": f"Error HTTP en búsqueda web: {e}"}
    except Exception as e:
        print(f"ERROR en web_search_real: {e}")
        return {"result": f"Error inesperado: {e}"}
