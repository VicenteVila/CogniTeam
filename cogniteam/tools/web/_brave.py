from typing import Any, Dict, List, Optional

import httpx

from cogniteam.config.settings import settings


def brave_search(
    query: str,
    num_results: int = 5,
    timeout_seconds: int = 15,
) -> Optional[List[Dict[str, Any]]]:
    from cogniteam.tools.utils.ratelimit import check_rate_limit, record_request

    api_key = settings.brave_api_key
    if not api_key:
        return None

    if not check_rate_limit("brave", max_per_day=100, max_per_minute=5):
        print("  [Brave Search] Límite diario o por minuto alcanzado, saltando búsqueda.")
        return None

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params: Dict[str, Any] = {
        "q": query,
        "count": min(num_results, 10),
        "safesearch": "moderate",
        "text_format": "plain",
    }

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        record_request("brave")

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        return results
    except Exception as e:
        print(f"  [Brave Search] Error: {e}")
        return None
