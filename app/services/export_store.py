import time
import uuid
import threading
from typing import Any, Dict, List, Optional

_TTL_SECONDS = 1800


class _Store:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}

    def put(
        self,
        rows: List[Dict[str, Any]],
        entity: str,
        fields: List[str],
        total_count: Optional[int] = None,
    ) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._purge_expired()
            self._data[key] = {
                "rows": rows,
                "entity": entity,
                "fields": fields,
                "total_count": total_count or len(rows),
                "expires": time.time() + _TTL_SECONDS,
            }
        return key

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if time.time() > entry["expires"]:
                del self._data[key]
                return None
            return entry

    def _purge_expired(self):
        now = time.time()
        expired = [k for k, v in self._data.items() if now > v["expires"]]
        for k in expired:
            del self._data[k]

ExportStore = _Store()
