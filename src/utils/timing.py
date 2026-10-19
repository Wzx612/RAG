"""Latency collection.

Collection only -- turning a list of durations into P50/P95/P99 is
:mod:`src.evaluation.metrics`' job, so the two concerns do not overlap.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timer(sink: list[float] | None = None) -> Iterator[dict[str, float]]:
    """Time a block in milliseconds.

    Yields a mutable dict whose ``ms`` key is populated on exit, so the value is
    readable after the ``with`` block::

        with timer() as t:
            do_work()
        print(t["ms"])
    """
    box: dict[str, float] = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["ms"] = (time.perf_counter() - start) * 1000.0
        if sink is not None:
            sink.append(box["ms"])


class StageLatencies:
    """Accumulates per-stage durations (embed / dense / sparse / fusion / rerank / total)."""

    def __init__(self) -> None:
        """Start with no samples recorded."""
        self._stages: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one stage and record it under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._stages[name].append((time.perf_counter() - start) * 1000.0)

    def record(self, name: str, milliseconds: float) -> None:
        """Record a duration measured elsewhere."""
        self._stages[name].append(milliseconds)

    @property
    def stages(self) -> dict[str, list[float]]:
        """Return a copy of the collected samples, keyed by stage name."""
        return dict(self._stages)

    def merge(self, other: StageLatencies) -> None:
        """Fold another recorder's samples into this one."""
        for name, values in other._stages.items():
            self._stages[name].extend(values)
