import json
import time
from typing import Any, Optional

class CacheManager:
    def __init__(self, ttl_seconds: int = 3600):
        self._cache = {}
        self.ttl = ttl_seconds
    
    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['data']
            del self._cache[key]  # Remove expired
        return None
    
    async def set(self, key: str, value: Any):
        self._cache[key] = {
            'data': value,
            'timestamp': time.time()
        }
    
    async def get_all_keys(self) -> list:
        return list(self._cache.keys())