"""Clone Service — durable, resumable AI clone pipeline.

Orchestrates L1→L2→Train with per-stage checkpointing, heartbeat-based
orphan detection, cancellation, and graceful shutdown support.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bloom_kernel.config import KernelSettings
from bloom_kernel.models.clone_job import CloneJob
from bloom_kernel.models.mentor import Mentor
from bloom_kernel.services.pipeline_context import (
    PipelineCancelled,
    PipelineContext,
    PipelineShutdown,
)
from bloom_kernel.utils.database import Database

# L2 data type names (order matters — this is the execution sequence)
L2_TYPES = ("qa", "identity", "counseling", "preference", "diversity", "context")

# Terminal stages that should never be resumed
_TERMINAL_STAGES = ("completed", "failed")


class CloneService:
    """Orchestrate the AI clone training pipeline with durability."""

    def __init__(
        self,
        settings: KernelSettings,
        db: Database,
        llm_client: object | None = None,
        ingestion_pipeline: object | None = None,
    ) -> None:
        self._settings = settings
        self._db = db
        self._llm = llm_client
        self._ingestion = ingestion_pipeline
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._contexts: dict[str, PipelineContext] = {}
        self._log_buffers: dict[str, deque] = {}
        # Global shutdown event shared across all pipelines
        self._shutdown_event = asyncio.Event()

    # ── Public API ────────────────────────────────────────────────────────

    async def start_clone(self, mentor_id: str, resume: bool = False) -> dict:
        """Start or resume the AI clone training pipeline for a mentor.

        Args:
            mentor_id: Mentor UUID string
            resume: If True, resume from the last checkpoint of an existing job.
                    If False, mark any existing incomplete job as failed and create new.

        Returns:
            Clone job dict with job_id, mentor_id, stage.
        """
        # Check for active in-memory task
        if mentor_id in self._active_tasks and not self._active_tasks[mentor_id].done():
            raise ValueError(f"Clone training already in progress for mentor {mentor_id}")

        # Verify mentor exists and has knowledge chunks
        async with self._db.session() as session:
            mentor = await self._get_mentor(session, mentor_id)
            if mentor is None:
                raise ValueError(f"Mentor not found: {mentor_id}")

        chunk_count = await self._count_chunks(mentor_id)
        if chunk_count < 3:
            raise ValueError(
                f"Insufficient knowledge: {chunk_count} chunks (minimum 3 required). "
                "Upload more documents first."
            )

        if resume:
            job_dict = await self._resume_existing_job(mentor_id)
            if job_dict:
                return job_dict
            # No resumable job found — fall through to create new
            logger.info(f"No resumable job for mentor={mentor_id}, creating new")

        # Mark any non-terminal jobs as failed before creating new
        await self._fail_incomplete_jobs(mentor_id, "Superseded by new job")

        # Create new job
        async with self._db.session() as session:
            job = CloneJob(
                mentor_id=uuid.UUID(mentor_id),
                stage="pending",
                progress_pct=0,
                training_config={
                    "base_model": self._settings.CLONE_BASE_MODEL,
                    "lora_rank": self._settings.CLONE_LORA_RANK,
                    "lora_alpha": self._settings.CLONE_LORA_ALPHA,
                    "learning_rate": self._settings.CLONE_LEARNING_RATE,
                    "num_epochs": self._settings.CLONE_NUM_EPOCHS,
                },
            )
            session.add(job)
            await session.flush()
            job_id = str(job.id)

            await session.execute(
                update(Mentor)
                .where(Mentor.id == uuid.UUID(mentor_id))
                .values(clone_status="processing", updated_at=datetime.now(timezone.utc))
            )

        self._launch_pipeline(mentor_id, job_id)
        logger.info(f"Clone pipeline started: mentor={mentor_id}, job={job_id}")
        return {"job_id": job_id, "mentor_id": mentor_id, "stage": "pending"}

    async def cancel_job(self, mentor_id: str) -> dict | None:
        """Cancel the active clone job for a mentor.

        Sets the cancel event so the pipeline stops at the next checkpoint.
        """
        ctx = self._contexts.get(mentor_id)
        if ctx is not None:
            ctx.cancel_event.set()
            logger.info(f"Cancel requested: mentor={mentor_id}, job={ctx.job_id}")

        # Also set DB flag in case the in-memory context is gone
        async with self._db.session() as session:
            stmt = (
                select(CloneJob)
                .where(
                    CloneJob.mentor_id == uuid.UUID(mentor_id),
                    CloneJob.stage.notin_(_TERMINAL_STAGES),
                    CloneJob.cancelled == False,  # noqa: E712
                )
                .order_by(CloneJob.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                return None

            job.cancelled = True
            job.stage = "failed"
            job.error_message = "Cancelled by user"
            job.completed_at = datetime.now(timezone.utc)
            await session.commit()

            # Update mentor status
            await self._update_mentor_status(mentor_id, "failed")
            return job.to_dict()

    async def graceful_shutdown(self) -> None:
        """Signal all active pipelines to stop and wait for checkpoint save."""
        if not self._active_tasks:
            return

        logger.info(f"Graceful shutdown: signaling {len(self._active_tasks)} active pipelines")
        self._shutdown_event.set()

        # Also signal each individual context
        for ctx in self._contexts.values():
            ctx.shutdown_event.set()

        # Wait for tasks to finish (they should save checkpoint and exit)
        timeout = self._settings.CLONE_SHUTDOWN_TIMEOUT
        tasks = list(self._active_tasks.values())
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                logger.warning(f"Graceful shutdown: {len(pending)} tasks didn't finish in {timeout}s")
                for t in pending:
                    t.cancel()

    async def recover_orphan_jobs(self) -> int:
        """Find and recover orphaned jobs (stale heartbeat, non-terminal stage).

        Called at application startup.
        Returns number of jobs recovered.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self._settings.CLONE_ORPHAN_TIMEOUT
        )
        max_retries = self._settings.CLONE_MAX_RETRIES

        async with self._db.session() as session:
            stmt = select(CloneJob).where(
                and_(
                    CloneJob.stage.notin_(_TERMINAL_STAGES),
                    CloneJob.cancelled == False,  # noqa: E712
                    CloneJob.retry_count < max_retries,
                    # Either heartbeat is stale or was never set (old jobs)
                    (CloneJob.heartbeat_at < cutoff) | (CloneJob.heartbeat_at.is_(None)),
                )
            )
            result = await session.execute(stmt)
            orphans = result.scalars().all()

        recovered = 0
        for job in orphans:
            mentor_id = str(job.mentor_id)
            job_id = str(job.id)

            # Skip if this mentor already has an active task
            if mentor_id in self._active_tasks and not self._active_tasks[mentor_id].done():
                continue

            # Check if retry limit is reached
            if job.retry_count >= max_retries:
                logger.warning(f"Orphan job {job_id} exceeded max retries, marking failed")
                await self._update_job(
                    job_id,
                    stage="failed",
                    error_message=f"Exceeded max retries ({max_retries})",
                    completed_at=datetime.now(timezone.utc),
                )
                await self._update_mentor_status(mentor_id, "failed")
                continue

            # Increment retry count and resume
            await self._update_job(job_id, retry_count=job.retry_count + 1)
            self._launch_pipeline(mentor_id, job_id)
            recovered += 1
            logger.info(
                f"Recovered orphan job: {job_id}, mentor={mentor_id}, "
                f"retry={job.retry_count + 1}, stage={job.stage}"
            )

        if recovered:
            logger.info(f"Recovered {recovered} orphan clone jobs")
        return recovered

    # ── Query methods (unchanged signatures) ──────────────────────────────

    async def get_progress(self, mentor_id: str) -> dict | None:
        async with self._db.session() as session:
            stmt = (
                select(CloneJob)
                .where(CloneJob.mentor_id == uuid.UUID(mentor_id))
                .order_by(CloneJob.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            return job.to_dict() if job else None

    async def get_job(self, job_id: str) -> dict | None:
        async with self._db.session() as session:
            stmt = select(CloneJob).where(CloneJob.id == uuid.UUID(job_id))
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            return job.to_dict() if job else None

    async def list_jobs(self, mentor_id: str) -> list[dict]:
        async with self._db.session() as session:
            stmt = (
                select(CloneJob)
                .where(CloneJob.mentor_id == uuid.UUID(mentor_id))
                .order_by(CloneJob.created_at.desc())
            )
            result = await session.execute(stmt)
            return [j.to_dict() for j in result.scalars().all()]

    async def get_l1_results(self, mentor_id: str) -> dict | None:
        async with self._db.session() as session:
            stmt = (
                select(CloneJob)
                .where(CloneJob.mentor_id == uuid.UUID(mentor_id))
                .where(CloneJob.l1_profile.isnot(None))
                .order_by(CloneJob.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None or job.l1_profile is None:
                return None
            return {"job_id": str(job.id), "stage": job.stage, "l1_profile": job.l1_profile}

    def get_logs(self, job_id: str) -> list[tuple[str, str]]:
        return list(self._log_buffers.get(job_id, deque()))

    async def get_logs_stream(self, job_id: str):
        seen = 0
        while True:
            buffer = self._log_buffers.get(job_id, deque())
            logs = list(buffer)
            if len(logs) > seen:
                for ts, msg in logs[seen:]:
                    yield f"data: {json.dumps({'timestamp': ts, 'message': msg}, ensure_ascii=False)}\n\n"
                seen = len(logs)

            job = await self.get_job(job_id)
            if job and job["stage"] in _TERMINAL_STAGES:
                yield f"data: {json.dumps({'done': True, 'stage': job['stage']})}\n\n"
                break

            await asyncio.sleep(1)

    # ── Pipeline launcher ─────────────────────────────────────────────────

    def _launch_pipeline(self, mentor_id: str, job_id: str) -> None:
        """Create context, log buffer, and background task for a pipeline."""
        ctx = PipelineContext(
            job_id=job_id,
            mentor_id=mentor_id,
            shutdown_event=self._shutdown_event,
        )
        self._contexts[mentor_id] = ctx
        self._log_buffers[job_id] = deque(maxlen=500)

        task = asyncio.create_task(
            self._run_pipeline(ctx),
            name=f"clone_{mentor_id}",
        )
        self._active_tasks[mentor_id] = task

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def _heartbeat_loop(self, ctx: PipelineContext) -> None:
        """Update heartbeat_at periodically while the pipeline runs."""
        interval = self._settings.CLONE_HEARTBEAT_INTERVAL
        while not ctx.is_cancelled and not ctx.is_shutdown:
            try:
                await self._update_job(
                    ctx.job_id, heartbeat_at=datetime.now(timezone.utc)
                )
            except Exception:
                pass  # Non-critical — next heartbeat will retry
            try:
                await asyncio.wait_for(ctx.cancel_event.wait(), timeout=interval)
                break  # Event was set
            except asyncio.TimeoutError:
                continue

    # ── Pipeline Execution (checkpoint state machine) ─────────────────────

    async def _run_pipeline(self, ctx: PipelineContext) -> None:
        """Execute L1→L2→Train with per-stage checkpointing."""
        mentor_id = ctx.mentor_id
        job_id = ctx.job_id
        heartbeat_task: asyncio.Task | None = None

        try:
            self._log(job_id, "Pipeline started")

            # Start heartbeat
            heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(ctx), name=f"heartbeat_{job_id}"
            )

            # Load checkpoint state from DB
            checkpoint = await self._load_checkpoint(job_id)

            # Get mentor info
            async with self._db.session() as session:
                mentor = await self._get_mentor(session, mentor_id)
                mentor_name = mentor.name if mentor else ""
                mentor_bio = mentor.bio if mentor else ""

            # Get all chunks
            chunks = await self._get_all_chunks(mentor_id)
            self._log(job_id, f"Loaded {len(chunks)} knowledge chunks")

            # ── L1: Topics ──
            if not checkpoint.get("l1_topics"):
                ctx.check_cancelled()
                await self._update_job(
                    job_id, stage="l1_topics", progress_pct=5,
                    started_at=datetime.now(timezone.utc),
                )
                self._log(job_id, "L1 Topics: clustering + summarization")

                topics = await self._run_l1_topics(chunks)
                topics_data = [t.to_dict() for t in topics]

                await self._update_job(job_id, l1_topics=topics_data, progress_pct=12)
                self._log(job_id, f"L1 Topics checkpoint: {len(topics)} topics saved")
            else:
                topics_data = checkpoint["l1_topics"]
                self._log(job_id, f"L1 Topics: restored {len(topics_data)} from checkpoint")
                topics = self._rebuild_topics(topics_data)

            # ── L1: Shades ──
            if not checkpoint.get("l1_shades"):
                ctx.check_cancelled()
                await self._update_job(job_id, stage="l1_shades", progress_pct=15)
                self._log(job_id, "L1 Shades: generating personality dimensions")

                shades = await self._run_l1_shades(topics, chunks, mentor_name, mentor_bio)
                shades_data = [s.to_dict() for s in shades]

                await self._update_job(job_id, l1_shades=shades_data, progress_pct=22)
                self._log(job_id, f"L1 Shades checkpoint: {len(shades)} shades saved")
            else:
                shades_data = checkpoint["l1_shades"]
                self._log(job_id, f"L1 Shades: restored {len(shades_data)} from checkpoint")
                shades = self._rebuild_shades(shades_data)

            # ── L1: Biography ──
            if not checkpoint.get("l1_bio"):
                ctx.check_cancelled()
                await self._update_job(job_id, stage="l1_bio", progress_pct=26)
                self._log(job_id, "L1 Bio: generating mentor biography")

                biography = await self._run_l1_bio(topics, shades, mentor_name, mentor_bio)
                bio_data = biography.to_dict()

                await self._update_job(job_id, l1_bio=bio_data, progress_pct=30)
                self._log(job_id, "L1 Bio checkpoint saved")
            else:
                bio_data = checkpoint["l1_bio"]
                self._log(job_id, "L1 Bio: restored from checkpoint")
                biography = self._rebuild_bio(bio_data)

            # ── L1: Profile + Persona (combined into l1_profile) ──
            if not checkpoint.get("l1_profile"):
                ctx.check_cancelled()
                await self._update_job(job_id, stage="l1_profile", progress_pct=30)
                self._log(job_id, "L1 Profile: generating profile + persona")

                l1_result = await self._run_l1_profile_and_persona(
                    topics, shades, biography, mentor_name, mentor_bio
                )
                l1_profile_data = l1_result.to_dict()

                await self._update_job(job_id, l1_profile=l1_profile_data, progress_pct=33)
                self._log(job_id, "L1 complete: profile + persona checkpoint saved")

                # Update mentor persona_prompt
                if l1_result.persona_prompt:
                    async with self._db.session() as session:
                        await session.execute(
                            update(Mentor)
                            .where(Mentor.id == uuid.UUID(mentor_id))
                            .values(persona_prompt=l1_result.persona_prompt,
                                    updated_at=datetime.now(timezone.utc))
                        )
            else:
                l1_profile_data = checkpoint["l1_profile"]
                self._log(job_id, "L1 Profile: restored from checkpoint")

            # Reconstruct L1Result for L2
            l1_result = self._rebuild_l1_result(
                topics_data, shades_data, bio_data, l1_profile_data
            )

            # ── L2: Per-type synthesis with individual checkpoints ──
            completed_types = checkpoint.get("l2_completed_types", {})
            training_dir = Path(self._settings.TRAINING_DIR) / mentor_id
            training_dir.mkdir(parents=True, exist_ok=True)

            from bloom_kernel.pipeline.l2_training_data import L2TrainingDataSynthesizer

            l2 = L2TrainingDataSynthesizer(self._settings, self._llm)
            l2_generators = {
                "qa": lambda: l2._generate_qa_pairs(chunks, mentor_name, l1_result),
                "identity": lambda: l2._generate_identity_pairs(l1_result, mentor_name),
                "counseling": lambda: l2._generate_counseling_dialogues(l1_result, mentor_name),
                "preference": lambda: l2._generate_preference_pairs(l1_result, mentor_name),
                "diversity": lambda: l2._generate_diversity_responses(l1_result, mentor_name),
                "context": lambda: l2._generate_context_dialogues(chunks, mentor_name, l1_result),
            }

            type_pct_start = 34
            type_pct_step = 5  # ~30% range for 6 types

            for i, dtype in enumerate(L2_TYPES):
                if completed_types.get(dtype):
                    self._log(job_id, f"L2 {dtype}: restored from checkpoint")
                    continue

                ctx.check_cancelled()
                pct = type_pct_start + i * type_pct_step
                await self._update_job(
                    job_id, stage="l2_synthesis", substage=dtype, progress_pct=pct
                )
                self._log(job_id, f"L2 {dtype}: generating...")

                samples = await l2_generators[dtype]()

                # Save this type as individual file
                type_path = training_dir / f"{dtype}.json"
                await asyncio.to_thread(
                    l2._save_json, samples, str(type_path)
                )

                # Update checkpoint
                completed_types[dtype] = True
                await self._update_job(job_id, l2_completed_types=completed_types)
                self._log(job_id, f"L2 {dtype} checkpoint: {len(samples)} samples saved")

            # ── L2: Merge all types into merged.json ──
            ctx.check_cancelled()
            await self._update_job(job_id, stage="l2_merge", progress_pct=64)
            self._log(job_id, "L2: Merging all types into merged.json")

            all_samples = []
            for dtype in L2_TYPES:
                type_path = training_dir / f"{dtype}.json"
                if type_path.exists():
                    with open(type_path, encoding="utf-8") as f:
                        all_samples.extend(json.load(f))

            persona_prompt = l1_profile_data.get("persona_prompt", "")
            chatml_data = l2._to_chatml(all_samples, persona_prompt)
            merged_path = str(training_dir / "merged.json")
            await asyncio.to_thread(l2._save_json, chatml_data, merged_path)

            l2_stats = {
                "qa_count": len(json.loads((training_dir / "qa.json").read_text()))
                    if (training_dir / "qa.json").exists() else 0,
                "identity_count": len(json.loads((training_dir / "identity.json").read_text()))
                    if (training_dir / "identity.json").exists() else 0,
                "dialogue_count": len(json.loads((training_dir / "counseling.json").read_text()))
                    if (training_dir / "counseling.json").exists() else 0,
                "preference_count": len(json.loads((training_dir / "preference.json").read_text()))
                    if (training_dir / "preference.json").exists() else 0,
                "diversity_count": len(json.loads((training_dir / "diversity.json").read_text()))
                    if (training_dir / "diversity.json").exists() else 0,
                "context_count": len(json.loads((training_dir / "context.json").read_text()))
                    if (training_dir / "context.json").exists() else 0,
                "total_samples": len(chatml_data),
                "output_path": merged_path,
            }
            await self._update_job(job_id, l2_stats=l2_stats, progress_pct=66)
            self._log(job_id, f"L2 complete: {l2_stats['total_samples']} total samples merged")

            # ── Training ──
            ctx.check_cancelled()
            await self._update_job(job_id, stage="training", progress_pct=67)
            self._log(job_id, "Training: Starting LoRA fine-tuning")

            from bloom_kernel.pipeline.trainer import MentorTrainer

            trainer = MentorTrainer(self._settings)

            # Check for existing HF checkpoint to resume from
            checkpoint_dir = Path(self._settings.MODEL_DIR) / mentor_id / "checkpoints"
            resume_from = None
            if checkpoint_dir.exists():
                checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"))
                if checkpoints:
                    resume_from = str(checkpoints[-1])
                    self._log(job_id, f"Training: resuming from {resume_from}")

            train_result = await trainer.train(
                mentor_id=mentor_id,
                data_path=merged_path,
                on_progress=lambda pct, msg: self._progress_callback(job_id, pct, msg),
            )

            if not train_result.success:
                raise RuntimeError(f"Training failed: {train_result.error}")

            self._log(job_id, f"Training complete: {train_result.model_path}")

            # ── Finalize ──
            await self._update_job(
                job_id,
                stage="completed",
                progress_pct=100,
                model_path=train_result.model_path,
                completed_at=datetime.now(timezone.utc),
            )

            await self._update_mentor_status(mentor_id, "ready")
            self._log(job_id, "Pipeline completed successfully!")
            logger.info(f"Clone pipeline completed: mentor={mentor_id}, job={job_id}")

        except PipelineCancelled:
            logger.info(f"Clone pipeline cancelled: mentor={mentor_id}, job={job_id}")
            self._log(job_id, "Pipeline cancelled by user")
            await self._update_job(
                job_id,
                stage="failed",
                error_message="Cancelled by user",
                cancelled=True,
                completed_at=datetime.now(timezone.utc),
            )
            await self._update_mentor_status(mentor_id, "failed")

        except PipelineShutdown:
            # Don't mark as failed — leave in current stage for recovery on restart
            logger.info(f"Clone pipeline interrupted by shutdown: mentor={mentor_id}, job={job_id}")
            self._log(job_id, "Pipeline interrupted by shutdown (will resume on restart)")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Clone pipeline failed: mentor={mentor_id}, error={error_msg}")
            self._log(job_id, f"Pipeline failed: {error_msg}")
            await self._update_job(
                job_id,
                stage="failed",
                error_message=error_msg,
                completed_at=datetime.now(timezone.utc),
            )
            await self._update_mentor_status(mentor_id, "failed")

        finally:
            # Stop heartbeat
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            self._active_tasks.pop(mentor_id, None)
            self._contexts.pop(mentor_id, None)

    # ── L1 sub-stage runners ──────────────────────────────────────────────

    async def _run_l1_topics(self, chunks: list[dict]):
        from bloom_kernel.pipeline.l1.topics_generator import TopicsGenerator
        gen = TopicsGenerator(self._settings, self._llm,
                             self._get_embedding_model())
        return await gen.generate(chunks)

    async def _run_l1_shades(self, topics, chunks, mentor_name, mentor_bio):
        from bloom_kernel.pipeline.l1.shade_generator import ShadeGenerator
        gen = ShadeGenerator(self._llm)
        return await gen.generate(topics, chunks, mentor_name, mentor_bio)

    async def _run_l1_bio(self, topics, shades, mentor_name, mentor_bio):
        from bloom_kernel.pipeline.l1.bio_generator import BioGenerator
        gen = BioGenerator(self._llm)
        return await gen.generate(topics, shades, mentor_name, mentor_bio)

    async def _run_l1_profile_and_persona(self, topics, shades, biography, mentor_name, mentor_bio):
        """Run profile + persona generation and return complete L1Result."""
        from bloom_kernel.pipeline.l1 import L1Generator, L1Result

        gen = L1Generator(self._settings, self._llm, self._get_embedding_model())
        # Generate profile and persona using the L1Generator's methods
        profile_data = await gen._generate_profile(topics, mentor_name, mentor_bio)
        persona = await gen._generate_persona(profile_data, shades, biography, mentor_name)

        return L1Result(
            topics=topics,
            shades=shades,
            biography=biography,
            background=profile_data.get("background", ""),
            style=profile_data.get("style", ""),
            core_beliefs=profile_data.get("core_beliefs", []),
            expertise=profile_data.get("expertise", []),
            tone=profile_data.get("tone", ""),
            counseling_approach=profile_data.get("counseling_approach", ""),
            persona_prompt=persona,
        )

    # ── Checkpoint loading ────────────────────────────────────────────────

    async def _load_checkpoint(self, job_id: str) -> dict:
        """Load checkpoint state from DB for resume."""
        async with self._db.session() as session:
            stmt = select(CloneJob).where(CloneJob.id == uuid.UUID(job_id))
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                return {}
            return {
                "stage": job.stage,
                "l1_topics": job.l1_topics,
                "l1_shades": job.l1_shades,
                "l1_bio": job.l1_bio,
                "l1_profile": job.l1_profile,
                "l2_completed_types": job.l2_completed_types or {},
            }

    async def _resume_existing_job(self, mentor_id: str) -> dict | None:
        """Find and resume an existing incomplete job."""
        async with self._db.session() as session:
            stmt = (
                select(CloneJob)
                .where(
                    CloneJob.mentor_id == uuid.UUID(mentor_id),
                    CloneJob.stage.notin_(_TERMINAL_STAGES),
                    CloneJob.cancelled == False,  # noqa: E712
                    CloneJob.retry_count < self._settings.CLONE_MAX_RETRIES,
                )
                .order_by(CloneJob.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            job = result.scalar_one_or_none()
            if job is None:
                return None

            job_id = str(job.id)
            job.retry_count += 1
            await session.commit()

        self._launch_pipeline(mentor_id, job_id)
        logger.info(
            f"Resuming clone pipeline: mentor={mentor_id}, job={job_id}, "
            f"stage={job.stage}, retry={job.retry_count}"
        )
        return {"job_id": job_id, "mentor_id": mentor_id, "stage": job.stage, "resumed": True}

    # ── Rebuild dataclasses from checkpoint JSONB ─────────────────────────

    @staticmethod
    def _rebuild_topics(data: list[dict]):
        from bloom_kernel.pipeline.l1.topics_generator import TopicCluster
        return [
            TopicCluster(
                topic_name=t.get("topic_name", ""),
                summary=t.get("summary", ""),
                key_points=t.get("key_points", []),
                expertise_areas=t.get("expertise_areas", []),
                chunk_ids=t.get("chunk_ids", []),
                chunk_count=t.get("chunk_count", 0),
            )
            for t in data
        ]

    @staticmethod
    def _rebuild_shades(data: list[dict]):
        from bloom_kernel.pipeline.l1.shade_generator import Shade
        return [
            Shade(
                title=s.get("title", ""),
                description_2p=s.get("description_2p", ""),
                description_3p=s.get("description_3p", ""),
                keywords=s.get("keywords", []),
            )
            for s in data
        ]

    @staticmethod
    def _rebuild_bio(data: dict):
        from bloom_kernel.pipeline.l1.bio_generator import MentorBiography
        return MentorBiography(
            second_person=data.get("second_person", ""),
            third_person=data.get("third_person", ""),
            summary=data.get("summary", ""),
        )

    @staticmethod
    def _rebuild_l1_result(topics_data, shades_data, bio_data, profile_data):
        from bloom_kernel.pipeline.l1 import L1Result
        from bloom_kernel.pipeline.l1.bio_generator import MentorBiography
        from bloom_kernel.pipeline.l1.shade_generator import Shade
        from bloom_kernel.pipeline.l1.topics_generator import TopicCluster

        topics = [
            TopicCluster(
                topic_name=t.get("topic_name", ""),
                summary=t.get("summary", ""),
                key_points=t.get("key_points", []),
                expertise_areas=t.get("expertise_areas", []),
                chunk_ids=t.get("chunk_ids", []),
                chunk_count=t.get("chunk_count", 0),
            )
            for t in (topics_data or [])
        ]
        shades = [
            Shade(
                title=s.get("title", ""),
                description_2p=s.get("description_2p", ""),
                description_3p=s.get("description_3p", ""),
                keywords=s.get("keywords", []),
            )
            for s in (shades_data or [])
        ]
        bio = MentorBiography(
            second_person=bio_data.get("second_person", "") if bio_data else "",
            third_person=bio_data.get("third_person", "") if bio_data else "",
            summary=bio_data.get("summary", "") if bio_data else "",
        )

        return L1Result(
            topics=topics,
            shades=shades,
            biography=bio,
            background=profile_data.get("background", "") if profile_data else "",
            style=profile_data.get("style", "") if profile_data else "",
            core_beliefs=profile_data.get("core_beliefs", []) if profile_data else [],
            expertise=profile_data.get("expertise", []) if profile_data else [],
            tone=profile_data.get("tone", "") if profile_data else "",
            counseling_approach=profile_data.get("counseling_approach", "") if profile_data else "",
            persona_prompt=profile_data.get("persona_prompt", "") if profile_data else "",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _get_embedding_model(self):
        if self._ingestion and hasattr(self._ingestion, '_embedding_model'):
            return self._ingestion._embedding_model
        return None

    async def _fail_incomplete_jobs(self, mentor_id: str, reason: str) -> None:
        """Mark all non-terminal jobs for a mentor as failed."""
        async with self._db.session() as session:
            await session.execute(
                update(CloneJob)
                .where(
                    CloneJob.mentor_id == uuid.UUID(mentor_id),
                    CloneJob.stage.notin_(_TERMINAL_STAGES),
                )
                .values(
                    stage="failed",
                    error_message=reason,
                    completed_at=datetime.now(timezone.utc),
                )
            )

    async def _update_mentor_status(self, mentor_id: str, status: str) -> None:
        try:
            async with self._db.session() as session:
                await session.execute(
                    update(Mentor)
                    .where(Mentor.id == uuid.UUID(mentor_id))
                    .values(clone_status=status, updated_at=datetime.now(timezone.utc))
                )
        except Exception:
            pass

    async def _progress_callback(self, job_id: str, pct: int, message: str) -> None:
        self._log(job_id, message)
        await self._update_job(job_id, progress_pct=pct)

    def _log(self, job_id: str, message: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        buffer = self._log_buffers.get(job_id)
        if buffer is not None:
            buffer.append((ts, message))
        logger.info(f"[clone:{job_id[:8]}] {message}")

    async def _update_job(self, job_id: str, **kwargs) -> None:
        kwargs["updated_at"] = datetime.now(timezone.utc)
        try:
            async with self._db.session() as session:
                await session.execute(
                    update(CloneJob)
                    .where(CloneJob.id == uuid.UUID(job_id))
                    .values(**kwargs)
                )
        except Exception as e:
            logger.error(f"Failed to update clone job {job_id}: {e}")

    async def _get_mentor(self, session: AsyncSession, mentor_id: str) -> Mentor | None:
        stmt = select(Mentor).where(Mentor.id == uuid.UUID(mentor_id))
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def _count_chunks(self, mentor_id: str) -> int:
        if self._ingestion is None or self._ingestion._chroma_client is None:
            return 0
        try:
            collection = self._ingestion._chroma_client.get_or_create_collection(
                f"mentor_{mentor_id}"
            )
            return collection.count()
        except Exception:
            return 0

    async def _get_all_chunks(self, mentor_id: str) -> list[dict]:
        if self._ingestion is None or self._ingestion._chroma_client is None:
            return []
        try:
            collection = self._ingestion._chroma_client.get_or_create_collection(
                f"mentor_{mentor_id}"
            )
            total = collection.count()
            if total == 0:
                return []

            result = collection.get(
                limit=total,
                include=["documents", "metadatas"],
            )

            chunks = []
            for i, chunk_id in enumerate(result["ids"]):
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": result["documents"][i] if result["documents"] else "",
                    "metadata": result["metadatas"][i] if result["metadatas"] else {},
                })
            return chunks
        except Exception as e:
            logger.error(f"Failed to get chunks for mentor {mentor_id}: {e}")
            return []
