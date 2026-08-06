"""Atomic filesystem artifacts with explicit checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from quant_bench.contracts import ArtifactRef

ArtifactVisibility = Literal["public", "private", "trusted_local_only"]


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.artifacts: list[ArtifactRef] = []

    def _target(self, relative_path: str) -> Path:
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"artifact path must stay inside the run directory: {relative_path}")
        target = (self.root / Path(*pure.parts)).resolve()
        if self.root not in target.parents and target != self.root:
            raise ValueError(f"artifact path escapes the run directory: {relative_path}")
        return target

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def write_bytes(
        self,
        relative_path: str,
        data: bytes,
        media_type: str,
        *,
        schema_id: str | None = None,
        visibility: ArtifactVisibility = "public",
        track: bool = True,
    ) -> ArtifactRef:
        target = self._target(relative_path)
        self._atomic_write(target, data)
        reference = ArtifactRef(
            path=relative_path,
            media_type=media_type,
            schema_id=schema_id,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            visibility=visibility,
        )
        if track:
            self.artifacts.append(reference)
        return reference

    def write_text(
        self,
        relative_path: str,
        value: str,
        media_type: str = "text/plain; charset=utf-8",
        *,
        schema_id: str | None = None,
        visibility: ArtifactVisibility = "public",
        track: bool = True,
    ) -> ArtifactRef:
        return self.write_bytes(
            relative_path,
            value.encode("utf-8"),
            media_type,
            schema_id=schema_id,
            visibility=visibility,
            track=track,
        )

    def write_json(
        self,
        relative_path: str,
        value: BaseModel | dict[str, Any] | list[Any],
        *,
        schema_id: str | None = None,
        visibility: ArtifactVisibility = "public",
        track: bool = True,
    ) -> ArtifactRef:
        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
        return self.write_bytes(
            relative_path,
            encoded,
            "application/json",
            schema_id=schema_id,
            visibility=visibility,
            track=track,
        )

    def write_csv(
        self, relative_path: str, frame: pd.DataFrame, *, schema_id: str | None = None
    ) -> ArtifactRef:
        data = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        return self.write_bytes(relative_path, data, "text/csv; charset=utf-8", schema_id=schema_id)

    def read_bytes(self, relative_path: str) -> bytes:
        return self._target(relative_path).read_bytes()

    def write_checksums(self) -> ArtifactRef:
        lines = [
            f"{artifact.sha256}  {artifact.path}"
            for artifact in sorted(self.artifacts, key=lambda item: item.path)
        ]
        return self.write_text("checksums.sha256", "\n".join(lines) + "\n", track=False)


def verify_run(run_dir: Path) -> list[str]:
    root = run_dir.resolve()
    checksum_file = root / "checksums.sha256"
    if not checksum_file.is_file():
        return ["checksums.sha256 is missing"]
    errors: list[str] = []
    for line_number, line in enumerate(checksum_file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative_path = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid checksum line {line_number}")
            continue
        target = (root / relative_path).resolve()
        if root not in target.parents:
            errors.append(f"checksum path escapes run directory: {relative_path}")
        elif not target.is_file():
            errors.append(f"artifact is missing: {relative_path}")
        else:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if actual != expected:
                errors.append(f"checksum mismatch: {relative_path}")
    return errors
