"""Distributed rate limiting primitives.

Production deployments use Redis so limits apply consistently across API replicas. The
in-memory implementation is deliberately available only for local/test environments.
"""

import time
from collections import defaultdict, deque

from redis.asyncio import Redis


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        now = time.monotonic()
        events = self._events[key]
        cutoff = now - window_seconds
        while events and events[0] <= cutoff:
            events.popleft()
        if len(events) >= limit:
            return False
        events.append(now)
        return True


class RedisRateLimiter:
    _script = """
    local current = redis.call('INCR', KEYS[1])
    if current == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
    return current
    """

    def __init__(self, redis: Redis) -> None:
        self.redis = redis

    async def allow(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        current = await self.redis.eval(self._script, 1, key, window_seconds)
        return int(current) <= limit
