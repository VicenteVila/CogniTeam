from typing import Any, Optional

from cogniteam.config.settings import settings

google_search_service: Any = None
cse_api_key: Optional[str] = None
cse_cx: Optional[str] = None


def init_cse_service():
    global google_search_service, cse_api_key, cse_cx
    cse_api_key = settings.cse_api_key or None
    cse_cx = settings.cse_cx or None
    if cse_api_key and cse_cx:
        try:
            from googleapiclient.discovery import build
            google_search_service = build("customsearch", "v1", developerKey=cse_api_key)
            print("INFO: Google Custom Search Service inicializado.")
        except Exception as e:
            print(f"ERROR inicializando Google Custom Search: {e}")
            google_search_service = None
    else:
        print("ADVERTENCIA: CSE_API_KEY o CSE_CX no configurados. Búsqueda web no disponible.")
