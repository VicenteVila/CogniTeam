import re
from typing import Dict

import httpx
from bs4 import BeautifulSoup

from cogniteam.utils.security import validate_url
from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff
def browse_web_page(url: str) -> Dict[str, str]:
    print(f"\n-- [browse_web_page] Navegando URL: '{url}'")
    if not validate_url(url):
        return {"result": f"Error: URL inválida: '{url}'."}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=httpx.Timeout(15.0, read=20.0)) as client:
            response = client.get(url)
            response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "charset" in content_type:
            encoding = content_type.split("charset=")[-1].split(";")[0].strip()
        else:
            encoding = "utf-8"
        response.encoding = encoding
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in [
            "script", "style", "nav", "footer", "header", "aside",
            "form", "button", "input", "select", "textarea", "iframe",
            "noscript", "svg", "path",
            ".sidebar", "#sidebar",
            '[class*="ad"]', '[id*="ad"]', '[class*="banner"]',
        ]:
            for el in soup.select(selector):
                el.decompose()
        content_node = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", attrs={"role": "main"})
            or soup.find("div", class_=re.compile(r"(content|main|body|post|entry)", re.I))
            or soup.body
            or soup
        )
        text = content_node.get_text(separator="\n", strip=True) if content_node else ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = []
        for line in lines:
            line = re.sub(r"\s{3,}", "  ", line)
            if len(line) < 20 and not re.search(r"[.!?:]$", line) and not re.match(r"^(#+\s|\*\s|- )", line):
                if cleaned and len(cleaned[-1]) > 80:
                    continue
            cleaned.append(line)
        cleaned_text = "\n".join(cleaned)
        max_len = 25000
        final = cleaned_text[:max_len] + ("..." if len(cleaned_text) > max_len else "")
        return {"result": final if final.strip() else "No se pudo extraer contenido relevante."}
    except httpx.TimeoutException:
        return {"result": f"Error: Timeout en '{url}'."}
    except httpx.HTTPStatusError as e:
        return {"result": f"Error HTTP {e.response.status_code} en '{url}'."}
    except httpx.RequestError as e:
        return {"result": f"Error de red en '{url}': {e}"}
    except Exception as e:
        print(f"ERROR en browse_web_page: {e}")
        return {"result": f"Error procesando URL '{url}': {e}"}
