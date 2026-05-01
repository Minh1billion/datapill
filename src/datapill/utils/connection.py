import time
from typing import Callable, Awaitable


async def timed_connect(probe: Callable[[], Awaitable[None]]) -> tuple[bool, float | None, str | None]:
    t0 = time.perf_counter()
    try:
        await probe()
        return True, 1000 * (time.perf_counter() - t0), None
    except Exception as e:
        return False, None, str(e)