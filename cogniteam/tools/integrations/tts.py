import traceback
from typing import Any, Dict, Optional

from cogniteam.tools.base import ToolResponse

_tts_engine = None


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
        except Exception:
            _tts_engine = None
    return _tts_engine


from cogniteam.tools.utils.retry import sync_retry_with_backoff


@sync_retry_with_backoff(retries=1)
def speak_text(
    text_to_speak: str,
    wait_until_finished: Optional[bool] = True,
) -> Dict[str, Any]:
    print(f"\n-- [speak_text] '{text_to_speak[:100]}...'")
    engine = _get_tts_engine()
    if not engine:
        return ToolResponse(
            success=False,
            message="TTS (pyttsx3) no disponible.",
            data=None,
        ).model_dump()

    if not text_to_speak or not text_to_speak.strip():
        return ToolResponse(
            success=False,
            message="Texto vacío.",
            data=None,
        ).model_dump()

    try:
        engine.say(text_to_speak)
        if wait_until_finished:
            engine.runAndWait()
        return ToolResponse(
            success=True,
            message="Texto encolado en TTS.",
            data={"status": "completed" if wait_until_finished else "queued"},
        ).model_dump()
    except RuntimeError as e:
        try:
            _tts_engine.stop()
        except Exception:
            pass
        return ToolResponse(
            success=False,
            message=f"TTS RuntimeError: {e}",
            data={"error": str(e)},
        ).model_dump()
    except Exception as e:
        print(f"ERROR speak_text: {e}")
        traceback.print_exc()
        return ToolResponse(
            success=False,
            message=f"Error TTS: {e}",
            data={"error": str(e)},
        ).model_dump()
