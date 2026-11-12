from itertools import count

import pytest

from ohh_gpu.ops import OpRegistry


@pytest.fixture
def ops() -> OpRegistry:
    counter = count(1)
    return OpRegistry(id_factory=lambda: f"op{next(counter)}")


def test_create_returns_a_running_op(ops):
    op = ops.create("load")
    assert op.id == "op1"
    assert op.kind == "load"
    assert op.status == "running"
    assert op.result is None
    assert op.error is None


def test_finish_records_the_result(ops):
    op = ops.create("load")
    ops.finish(op.id, {"model": "coder", "vram_mb": 4700})
    stored = ops.get(op.id)
    assert stored.status == "done"
    assert stored.result == {"model": "coder", "vram_mb": 4700}


def test_fail_records_the_error(ops):
    op = ops.create("generate_image")
    ops.fail(op.id, {"code": "VRAM_EXHAUSTED", "remedy": "proceed without it"})
    stored = ops.get(op.id)
    assert stored.status == "failed"
    assert stored.error["code"] == "VRAM_EXHAUSTED"


def test_unknown_op_returns_none(ops):
    assert ops.get("nope") is None


def test_registry_evicts_the_oldest_beyond_capacity():
    counter = count(1)
    ops = OpRegistry(id_factory=lambda: f"op{next(counter)}", capacity=2)
    first = ops.create("load")
    ops.create("load")
    ops.create("load")
    assert ops.get(first.id) is None
    assert ops.get("op3") is not None


def test_finish_on_an_evicted_op_id_is_a_noop_not_a_keyerror():
    # A long-running op can be pushed out by capacity before it completes.
    # finish() runs inside a background task with no caller watching for an
    # exception, so a missing op must be silently ignored rather than raise.
    counter = count(1)
    ops = OpRegistry(id_factory=lambda: f"op{next(counter)}", capacity=1)
    first = ops.create("load")
    ops.create("load")  # evicts `first`
    assert ops.get(first.id) is None

    ops.finish(first.id, {"model": "coder"})  # must not raise

    assert ops.get(first.id) is None


def test_fail_on_an_evicted_op_id_is_a_noop_not_a_keyerror():
    counter = count(1)
    ops = OpRegistry(id_factory=lambda: f"op{next(counter)}", capacity=1)
    first = ops.create("load")
    ops.create("load")  # evicts `first`
    assert ops.get(first.id) is None

    ops.fail(first.id, {"code": "VRAM_EXHAUSTED", "remedy": "proceed without it"})

    assert ops.get(first.id) is None
