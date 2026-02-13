"""Pipeline context — cancellation, shutdown, and inter-stage coordination.

Passed through the clone pipeline so each stage can check for
cancellation / shutdown requests between steps.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


class PipelineCancelled(Exception):
    """Raised when a pipeline is cancelled by user request."""


class PipelineShutdown(Exception):
    """Raised when the application is shutting down gracefully."""


@dataclass
class PipelineContext:
    """Shared context for a single pipeline execution."""

    job_id: str
    mentor_id: str
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)

    def check_cancelled(self) -> None:
        """Check if the pipeline should stop.

        Call this between stages — it's a synchronous check that raises
        if cancellation or shutdown has been requested.
        """
        if self.cancel_event.is_set():
            raise PipelineCancelled(f"Job {self.job_id} cancelled by user")
        if self.shutdown_event.is_set():
            raise PipelineShutdown(f"Job {self.job_id} interrupted by shutdown")

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    @property
    def is_shutdown(self) -> bool:
        return self.shutdown_event.is_set()
