"""Generic multi-experiment benchmark runner (planned, NOT implemented).

Intended contract: run every configured experiment over all eval queries,
recording per-query metrics and per-stage latency. Orchestration only.

Phase 6B instead uses the concrete, tested
:mod:`src.evaluation.hybrid_benchmark` runner and
``scripts/benchmark_hybrid.py``. This module remains a placeholder.
"""
