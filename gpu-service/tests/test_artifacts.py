from itertools import count
from pathlib import Path

import pytest

from ohh_gpu.artifacts import ArtifactStore
from ohh_gpu.errors import UnknownArtifact

PNG = b"\x89PNG\r\n\x1a\nbody"


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    counter = count(1)
    return ArtifactStore(root=tmp_path, id_factory=lambda: f"{next(counter):032x}")


def test_put_writes_the_file_and_returns_an_id(store: ArtifactStore):
    artifact_id = store.put_png(PNG, meta={"prompt": "a lantern"})
    assert store.path(artifact_id).read_bytes() == PNG


def test_url_matches_the_wire_protocol(store: ArtifactStore):
    artifact_id = store.put_png(PNG, meta={})
    assert store.url(artifact_id) == f"/artifacts/{artifact_id}.png"


def test_meta_round_trips(store: ArtifactStore):
    artifact_id = store.put_png(PNG, meta={"prompt": "a lantern", "steps": 4})
    assert store.meta(artifact_id) == {"prompt": "a lantern", "steps": 4}


def test_creates_the_root_directory_if_absent(tmp_path: Path):
    store = ArtifactStore(root=tmp_path / "nested" / "artifacts")
    artifact_id = store.put_png(PNG, meta={})
    assert store.path(artifact_id).exists()


def test_unknown_artifact_raises(store: ArtifactStore):
    with pytest.raises(UnknownArtifact):
        store.path("0" * 32)


@pytest.mark.parametrize(
    "bad_id", ["../../etc/passwd", "abc", "", "zzzz" * 8, "0" * 31, "0" * 33]
)
def test_malformed_ids_are_rejected(store: ArtifactStore, bad_id: str):
    with pytest.raises(UnknownArtifact):
        store.path(bad_id)
