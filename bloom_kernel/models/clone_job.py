"""CloneJob ORM model — tracks AI clone training pipeline stages."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from bloom_kernel.models.base import Base


class CloneJob(Base):
    __tablename__ = "clone_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mentor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Pipeline stage tracking
    stage: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default="pending"
    )
    substage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # L1 per-generator checkpoints
    l1_topics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    l1_shades: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    l1_bio: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # L1 output: knowledge profile (combined after all L1 generators)
    l1_profile: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # L2 output: synthesis stats
    l2_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    # L2 per-type completion tracking: {"qa": true, "identity": true, ...}
    l2_completed_types: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    # Heartbeat for orphan detection
    heartbeat_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Cancellation flag
    cancelled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    # Training configuration
    training_config: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default="{}")

    # Output model path
    model_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Timestamps
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    def to_dict(self) -> dict:
        """Convert to API response dictionary."""
        return {
            "id": str(self.id),
            "mentor_id": str(self.mentor_id),
            "stage": self.stage,
            "substage": self.substage,
            "progress_pct": self.progress_pct,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "l1_topics": self.l1_topics,
            "l1_shades": self.l1_shades,
            "l1_bio": self.l1_bio,
            "l1_profile": self.l1_profile,
            "l2_stats": self.l2_stats,
            "l2_completed_types": self.l2_completed_types,
            "training_config": self.training_config,
            "model_path": self.model_path,
            "cancelled": self.cancelled,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
