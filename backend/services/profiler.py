"""
ClipMatch Pipeline Profiler
Lightweight timing instrumentation for each pipeline stage.
"""

import time
import logging
from collections import OrderedDict

logger = logging.getLogger('clipmatch.profiler')


class PipelineProfiler:
    """
    Records wall-clock time for named pipeline stages.

    Usage:
        profiler = PipelineProfiler()
        profiler.start('frame_extraction')
        ...  # do work
        profiler.stop('frame_extraction')

        profiler.start('hash_extraction')
        ...
        profiler.stop('hash_extraction')

        print(profiler.summary())
    """

    def __init__(self):
        self._starts = {}
        self._elapsed = OrderedDict()

    # ── Public API ──────────────────────────────────────────────

    def start(self, stage: str):
        """Begin timing a named stage."""
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        """
        End timing for *stage* and accumulate its elapsed time.
        Silently ignores stages that were never started (defensive).
        """
        t0 = self._starts.pop(stage, None)
        if t0 is None:
            return 0.0
        elapsed = time.perf_counter() - t0
        self._elapsed[stage] = self._elapsed.get(stage, 0.0) + elapsed
        return elapsed

    def get(self, stage: str) -> float:
        """Return elapsed seconds for *stage* (0 if not recorded)."""
        return self._elapsed.get(stage, 0.0)

    def summary(self) -> dict:
        """
        Return an OrderedDict of ``{stage_name: elapsed_seconds}``
        rounded to 3 decimal places, plus a ``total_sec`` key.
        """
        result = OrderedDict()
        total = 0.0
        for stage, elapsed in self._elapsed.items():
            rounded = round(elapsed, 3)
            result[f'{stage}_sec'] = rounded
            total += elapsed
        result['total_sec'] = round(total, 3)
        return result

    def log_summary(self, prefix: str = 'Pipeline'):
        """Log a one-line summary at INFO level."""
        parts = []
        for stage, elapsed in self._elapsed.items():
            label = stage.replace('_', ' ').title()
            parts.append(f'{label}: {elapsed:.2f}s')
        logger.info('[%s] %s', prefix, ' | '.join(parts))
