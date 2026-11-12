"""Operation lifecycle.

``/load`` and ``/generate/image`` return 202 with an op_id and run in the
background; this registry is where the caller reads the outcome. Bounded so a
long-lived GPU service does not accumulate op records forever.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4


@dataclass
class Op:
    id: str
    kind: str
    status: str = "running"
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class OpRegistry:
    def __init__(
        self,
        id_factory: Callable[[], str] = lambda: uuid4().hex,
        capacity: int = 512,
    ) -> None:
        self._id_factory = id_factory
        self._capacity = capacity
        self._ops: OrderedDict[str, Op] = OrderedDict()

    def create(self, kind: str) -> Op:
        op = Op(id=self._id_factory(), kind=kind)
        self._ops[op.id] = op
        while len(self._ops) > self._capacity:
            self._ops.popitem(last=False)
        return op

    def get(self, op_id: str) -> Op | None:
        return self._ops.get(op_id)

    def finish(self, op_id: str, result: dict[str, Any]) -> None:
        # A long-running op can be evicted by capacity before it completes;
        # a missing op here is a no-op, not a KeyError raised inside a
        # background task where nothing would ever see it.
        op = self._ops.get(op_id)
        if op is None:
            return
        op.status = "done"
        op.result = result

    def fail(self, op_id: str, error: dict[str, Any]) -> None:
        op = self._ops.get(op_id)
        if op is None:
            return
        op.status = "failed"
        op.error = error
