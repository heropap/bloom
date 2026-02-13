"""Training routes — AI clone pipeline management."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from bloom_kernel.utils.jwt import get_current_user_id
from bloom_kernel.utils.response import fail, ok

router = APIRouter(prefix="/mentors", tags=["training"])


@router.post("/{mentor_id}/clone/start")
async def start_clone_training(
    mentor_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Start the AI clone training pipeline for a mentor.

    Prerequisites: at least 3 knowledge chunks uploaded.
    Pipeline: L1 Knowledge Extraction → L2 Data Synthesis → LoRA Training
    """
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    try:
        result = await clone_svc.start_clone(mentor_id)
        return ok(result)
    except ValueError as e:
        return fail(400, str(e))
    except Exception as e:
        return fail(500, f"Failed to start clone training: {e}", status_code=500)


@router.get("/{mentor_id}/clone/progress")
async def get_clone_progress(mentor_id: str):
    """Get the latest clone training progress for a mentor."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    progress = await clone_svc.get_progress(mentor_id)
    if progress is None:
        return ok({
            "stage": "none",
            "progress_pct": 0,
            "message": "No clone training jobs found for this mentor",
        })
    return ok(progress)


@router.get("/{mentor_id}/clone/jobs")
async def list_clone_jobs(mentor_id: str):
    """List all clone training jobs for a mentor."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    jobs = await clone_svc.list_jobs(mentor_id)
    return ok(jobs)


@router.get("/{mentor_id}/clone/jobs/{job_id}")
async def get_clone_job(mentor_id: str, job_id: str):
    """Get details of a specific clone training job."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    job = await clone_svc.get_job(job_id)
    if job is None:
        return fail(404, "Clone job not found", status_code=404)
    return ok(job)


@router.post("/{mentor_id}/clone/resume")
async def resume_clone_training(
    mentor_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Resume clone training from the last checkpoint."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    try:
        result = await clone_svc.start_clone(mentor_id, resume=True)
        return ok(result)
    except ValueError as e:
        return fail(400, str(e))
    except Exception as e:
        return fail(500, f"Failed to resume clone training: {e}", status_code=500)


@router.post("/{mentor_id}/clone/cancel")
async def cancel_clone_training(
    mentor_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Cancel the active clone training pipeline."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    result = await clone_svc.cancel_job(mentor_id)
    if result is None:
        return fail(404, "No active training job to cancel", status_code=404)
    return ok(result)


@router.get("/{mentor_id}/clone/logs/{job_id}")
async def stream_clone_logs(mentor_id: str, job_id: str):
    """Stream clone training logs via Server-Sent Events (SSE)."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    return StreamingResponse(
        clone_svc.get_logs_stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
