import os
import time
from typing import Any, Dict, List, Optional


def validate_html_functional(
    filepath: str,
    test_script: str = "",
    capture_screenshot: bool = False,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Carga un archivo HTML en Playwright headless y ejecuta validación funcional.

    Args:
        filepath: Ruta relativa al archivo HTML (dentro de project_root).
        test_script: Código JavaScript opcional que se inyecta tras el load.
                     Debe retornar un dict serializable con resultados.
        capture_screenshot: Si True, guarda un screenshot en el mismo directorio.
        timeout_seconds: Tiempo máximo de carga de página.

    Returns:
        Dict con:
          - success: bool (True si la página cargó sin errores fatales)
          - data: str resumen legible
          - passed: bool (test_script se ejecutó sin excepciones)
          - console_errors: List[str] mensajes de error en consola
          - screenshot_path: str (solo si capture_screenshot=True)
          - test_results: Dict (solo si test_script fue proporcionado)
          - page_title: str título de la página
    """
    from cogniteam.config.settings import settings

    root = settings.project_root
    abs_path = filepath if os.path.isabs(filepath) else os.path.join(root, filepath)

    if not os.path.isfile(abs_path):
        return {
            "success": False,
            "data": f"Archivo no encontrado: {filepath}",
            "passed": False,
            "console_errors": [f"FileNotFound: {abs_path}"],
            "screenshot_path": "",
            "page_title": "",
        }

    from playwright.sync_api import sync_playwright

    console_errors: List[str] = []
    screenshot_path = ""
    page_title = ""
    passed = False
    test_results: Dict[str, Any] = {}
    load_ok = False

    t0 = time.time()

    try:
        p = sync_playwright().start()
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})

        page.on("console", lambda msg: (
            console_errors.append(f"[{msg.type}] {msg.text}")
            if msg.type == "error"
            else None
        ))
        page.on("pageerror", lambda err: (
            console_errors.append(f"[PAGE_ERROR] {err}")
        ))

        page.goto(f"file://{abs_path}", wait_until="networkidle", timeout=timeout_seconds * 1000)
        time.sleep(0.5)
        page_title = page.title()
        load_ok = True

        if test_script:
            try:
                wrapped = f"(async () => {{{test_script}}})()"
                result = page.evaluate(wrapped)
                if isinstance(result, dict):
                    test_results = result
                elif result is not None:
                    test_results = {"return": result}
                else:
                    test_results = {}
                passed = True
            except Exception as e:
                console_errors.append(f"[TEST_SCRIPT_ERROR] {e}")
                test_results = {"error": str(e)}
                passed = False
        else:
            passed = len(console_errors) == 0

        if capture_screenshot:
            screenshot_dir = os.path.join(root, ".cogniteam", "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(filepath))[0]
            screenshot_path = os.path.join(screenshot_dir, f"{base}_{int(t0)}.png")
            page.screenshot(path=screenshot_path, full_page=True)

        browser.close()
        p.stop()

    except Exception as e:
        elapsed = time.time() - t0
        return {
            "success": False,
            "data": f"Error al cargar {filepath} en {elapsed:.1f}s: {e}",
            "passed": False,
            "console_errors": console_errors + [f"[BROWSER_ERROR] {e}"],
            "screenshot_path": screenshot_path,
            "page_title": "",
            "test_results": test_results,
        }

    elapsed = time.time() - t0
    summary_parts = [f"Página cargada en {elapsed:.1f}s: \"{page_title}\""]

    if console_errors:
        summary_parts.append(f"Errores de consola: {len(console_errors)}")
    if test_results:
        summary_parts.append(f"Test: {'PASÓ' if passed else 'FALLÓ'}")
    if screenshot_path:
        summary_parts.append(f"Screenshot: {screenshot_path}")

    return {
        "success": load_ok and passed,
        "data": " | ".join(summary_parts),
        "passed": passed,
        "console_errors": console_errors,
        "screenshot_path": screenshot_path,
        "test_results": test_results,
        "page_title": page_title,
        "load_time_seconds": round(elapsed, 2),
    }
