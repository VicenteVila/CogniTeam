import functools
import random
import time
from typing import Callable, Tuple, Type

from cogniteam.tools.base import ToolResponse


def sync_retry_with_backoff(
    retries: int = 3,
    backoff_in_seconds: float = 1.0,
    catch_exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    def rwb_sync(f: Callable):
        @functools.wraps(f)
        def wrapper_sync(*args, **kwargs):
            attempts = 0
            last_exception = None
            while attempts <= retries:
                try:
                    return f(*args, **kwargs)
                except catch_exceptions as e:
                    last_exception = e
                    print(
                        f"  {f.__name__} falló (intento {attempts + 1}/{retries + 1}): {type(e).__name__} - {e}."
                    )
                    attempts += 1
                    if attempts > retries:
                        print(
                            f"  ERROR CRÍTICO en {f.__name__} tras {attempts} intento(s): {last_exception}"
                        )
                        return ToolResponse(
                            success=False,
                            message=f"Error final en {f.__name__} tras {attempts} intento(s): {last_exception}",
                            data={
                                "error_details": str(last_exception),
                                "function_args": args,
                                "function_kwargs": kwargs,
                            },
                        ).model_dump()
                    sleep_time = backoff_in_seconds * (2 ** (attempts - 1)) + random.uniform(0, 0.5)
                    print(f"    Reintentando en {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
            print(
                f"  ERROR INESPERADO: Bucle de reintentos para {f.__name__} finalizó sin devolver valor."
            )
            return ToolResponse(
                success=False,
                message=f"Error inesperado en {f.__name__} (fuera de reintentos).",
                data={"error_details": str(last_exception)},
            ).model_dump()
        return wrapper_sync
    return rwb_sync
