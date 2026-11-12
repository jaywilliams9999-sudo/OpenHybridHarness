"""Artifact storage on the GPU machine.

Artifacts are written to disk here and returned to the orchestrator as a URL it
fetches. A 2MB PNG base64-encodes to roughly 2.7MB of text, which would
compete with progress ticks on the SSE channel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .errors import UnknownArtifact

ID_PATTERN = re.compile(r"\A[0-9a-f]{32}\Z")


class ArtifactStore:
    def __init__(
        self,
        root: Path,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._id_factory = id_factory

    def put_png(self, data: bytes, meta: dict[str, Any]) -> str:
        artifact_id = self._id_factory()
        (self._root / f"{artifact_id}.png").write_bytes(data)
        (self._root / f"{artifact_id}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        return artifact_id

    def path(self, artifact_id: str) -> Path:
        path = self._root / f"{self._validate(artifact_id)}.png"
        if not path.is_file():
            raise UnknownArtifact(
                f"No artifact {artifact_id!r}.",
                remedy="Re-generate the image; this artifact is gone.",
                artifact_id=artifact_id,
            )
        return path

    def meta(self, artifact_id: str) -> dict[str, Any]:
        self.path(artifact_id)  # existence + validation
        sidecar = self._root / f"{artifact_id}.json"
        return json.loads(sidecar.read_text(encoding="utf-8"))

    def url(self, artifact_id: str) -> str:
        return f"/artifacts/{self._validate(artifact_id)}.png"

    def _validate(self, artifact_id: str) -> str:
        if not ID_PATTERN.match(artifact_id):
            raise UnknownArtifact(
                f"Malformed artifact id {artifact_id!r}.",
                remedy="Artifact ids are 32 lowercase hex characters.",
                artifact_id=artifact_id,
            )
        return artifact_id
