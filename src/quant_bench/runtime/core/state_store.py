from pathlib import Path
from typing import Any

from quant_bench.runtime.core.atomic_io import atomic_write_json, safe_read_json


class JsonStateStore:
    """Small JSON-backed state store with atomic writes and corrupted-file recovery."""

    def __init__(self, path: str | Path, default: dict[str, Any] | None = None):
        self.path = Path(path)
        self.default = dict(default or {})
        self.state = safe_read_json(self.path, self.default)
        for key, value in self.default.items():
            self.state.setdefault(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def update(self, values: dict[str, Any]) -> None:
        self.state.update(values)

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def mark_done(self, collection_key: str, value: str) -> None:
        items = list(self.state.get(collection_key, []))
        if value not in items:
            items.append(value)
        self.state[collection_key] = items
        self.save()
