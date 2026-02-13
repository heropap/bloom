"""Studio routes — mentor self-service portal.

All routes are scoped to the authenticated mentor via require_mentor().
No {mentor_id} param needed — auto-resolved from JWT.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from loguru import logger
from pydantic import BaseModel

from bloom_kernel.services.mentor_service import MentorService
from bloom_kernel.utils.auth_deps import MentorContext, require_mentor
from bloom_kernel.utils.database import db
from bloom_kernel.utils.response import fail, ok

router = APIRouter(prefix="/studio/mentor", tags=["studio"])

_svc = MentorService()


# ── Request Models ───────────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    name: str | None = None
    bio: str | None = None
    avatar_url: str | None = None
    persona_prompt: str | None = None
    greeting_text: str | None = None
    style_tags: list[str] | None = None


class UpdateVoiceRequest(BaseModel):
    voice_model_id: str
    voice_sample_url: str | None = None


class IngestTextRequest(BaseModel):
    content: str
    source: str = "manual"
    metadata: dict[str, str] = {}


# ── Profile ──────────────────────────────────────────────────────────

@router.get("/profile")
async def get_profile(ctx: MentorContext = Depends(require_mentor)):
    """Get the authenticated mentor's profile."""
    async with db.session() as session:
        mentor = await _svc.get_by_id(session, ctx.mentor_id)
        if mentor is None:
            return fail(404, "Mentor not found", status_code=404)
        return ok(mentor)


@router.put("/profile")
async def update_profile(
    req: UpdateProfileRequest,
    ctx: MentorContext = Depends(require_mentor),
):
    """Update the authenticated mentor's profile."""
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    if not data:
        return fail(400, "No fields to update")

    async with db.session() as session:
        updated = await _svc.update_mentor(session, ctx.mentor_id, data)
        if updated is None:
            return fail(404, "Mentor not found", status_code=404)
        return ok(updated)


# ── Voice ────────────────────────────────────────────────────────────

@router.get("/voice")
async def get_voice_status(ctx: MentorContext = Depends(require_mentor)):
    """Get voice model status for the authenticated mentor."""
    async with db.session() as session:
        status = await _svc.get_voice_status(session, ctx.mentor_id)
        if status is None:
            return fail(404, "Mentor not found", status_code=404)
        return ok(status)


@router.put("/voice")
async def update_voice(
    req: UpdateVoiceRequest,
    ctx: MentorContext = Depends(require_mentor),
):
    """Update voice model after training."""
    async with db.session() as session:
        await _svc.update_voice_model(
            session, ctx.mentor_id, req.voice_model_id, req.voice_sample_url
        )
        return ok({"mentor_id": ctx.mentor_id, "voice_model_id": req.voice_model_id})


@router.post("/voice/test")
async def test_voice(ctx: MentorContext = Depends(require_mentor)):
    """Synthesize a test phrase with the mentor's voice model."""
    # TODO: integrate with TTS engine for voice preview
    return ok({"status": "not_implemented", "mentor_id": ctx.mentor_id})


# ── Knowledge ────────────────────────────────────────────────────────

@router.post("/knowledge/upload")
async def upload_knowledge(
    file: UploadFile = File(...),
    source: str = Form(None),
    ctx: MentorContext = Depends(require_mentor),
):
    """Upload a document for knowledge ingestion."""
    from bloom_kernel.api.mentors import _extract_docx_text, _extract_pdf_text
    from bloom_kernel.main import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    if pipeline is None:
        return fail(503, "Ingestion pipeline not available", status_code=503)

    allowed = {".txt", ".md", ".text", ".pdf", ".docx"}
    filename = file.filename or "upload.txt"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed:
        return fail(400, f"Unsupported file type: {ext}. Allowed: {', '.join(allowed)}")

    raw_bytes = await file.read()
    if ext in (".txt", ".md", ".text"):
        content = raw_bytes.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        content = _extract_pdf_text(raw_bytes)
    elif ext == ".docx":
        content = _extract_docx_text(raw_bytes)
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    if not content.strip():
        return fail(400, "Document is empty or could not be parsed")

    file_source = source or filename
    chunk_ids = await pipeline.ingest(ctx.mentor_id, content, file_source)
    logger.info(f"Studio knowledge upload: mentor={ctx.mentor_id}, file={filename}, chunks={len(chunk_ids)}")
    return ok({"chunk_ids": chunk_ids, "chunk_count": len(chunk_ids), "source": file_source})


@router.post("/knowledge/text")
async def ingest_text(
    req: IngestTextRequest,
    ctx: MentorContext = Depends(require_mentor),
):
    """Ingest raw text into the mentor's knowledge base."""
    from bloom_kernel.main import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    if pipeline is None:
        return fail(503, "Ingestion pipeline not available", status_code=503)

    if not req.content.strip():
        return fail(400, "Content is empty")

    chunk_ids = await pipeline.ingest(ctx.mentor_id, req.content, req.source, req.metadata)
    return ok({"chunk_ids": chunk_ids, "chunk_count": len(chunk_ids), "source": req.source})


@router.get("/knowledge/chunks")
async def list_chunks(
    page: int = 1,
    page_size: int = 20,
    ctx: MentorContext = Depends(require_mentor),
):
    """List knowledge chunks for the authenticated mentor."""
    from bloom_kernel.main import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    if pipeline is None:
        return fail(503, "Ingestion pipeline not available", status_code=503)

    if pipeline._chroma_client is None:
        return ok({"chunks": [], "total": 0, "page": page})

    try:
        collection = pipeline._chroma_client.get_or_create_collection(f"mentor_{ctx.mentor_id}")
        total = collection.count()
        offset = (page - 1) * page_size
        result = collection.get(limit=page_size, offset=offset, include=["documents", "metadatas"])
        chunks = []
        for i, chunk_id in enumerate(result["ids"]):
            chunks.append({
                "chunk_id": chunk_id,
                "content": result["documents"][i] if result["documents"] else "",
                "metadata": result["metadatas"][i] if result["metadatas"] else {},
            })
        return ok({"chunks": chunks, "total": total, "page": page, "page_size": page_size})
    except Exception as e:
        logger.error(f"Failed to list chunks: {e}")
        return fail(500, f"Failed to list chunks: {e}", status_code=500)


@router.delete("/knowledge/chunks/{chunk_id}")
async def delete_chunk(
    chunk_id: str,
    ctx: MentorContext = Depends(require_mentor),
):
    """Delete a knowledge chunk."""
    from bloom_kernel.main import get_ingestion_pipeline

    pipeline = get_ingestion_pipeline()
    if pipeline is None:
        return fail(503, "Ingestion pipeline not available", status_code=503)

    if pipeline._chroma_client is None:
        return fail(503, "ChromaDB not available", status_code=503)

    try:
        collection = pipeline._chroma_client.get_or_create_collection(f"mentor_{ctx.mentor_id}")
        collection.delete(ids=[chunk_id])
        return ok({"deleted": chunk_id})
    except Exception as e:
        logger.error(f"Failed to delete chunk: {e}")
        return fail(500, f"Failed to delete chunk: {e}", status_code=500)


# ── Training ─────────────────────────────────────────────────────────

@router.post("/training/start")
async def start_training(ctx: MentorContext = Depends(require_mentor)):
    """Start AI clone training pipeline for the authenticated mentor."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    try:
        job = await clone_svc.start_clone(ctx.mentor_id)
        return ok(job)
    except ValueError as e:
        return fail(400, str(e), status_code=400)
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        return fail(500, f"Training start failed: {e}", status_code=500)


@router.post("/training/resume")
async def resume_training(ctx: MentorContext = Depends(require_mentor)):
    """Resume AI clone training from the last checkpoint."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    try:
        job = await clone_svc.start_clone(ctx.mentor_id, resume=True)
        return ok(job)
    except ValueError as e:
        return fail(400, str(e), status_code=400)
    except Exception as e:
        logger.error(f"Failed to resume training: {e}")
        return fail(500, f"Training resume failed: {e}", status_code=500)


@router.post("/training/cancel")
async def cancel_training(ctx: MentorContext = Depends(require_mentor)):
    """Cancel the active AI clone training pipeline."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    result = await clone_svc.cancel_job(ctx.mentor_id)
    if result is None:
        return fail(404, "No active training job to cancel", status_code=404)
    return ok(result)


@router.get("/training/progress")
async def get_training_progress(ctx: MentorContext = Depends(require_mentor)):
    """Get latest clone training progress."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    progress = await clone_svc.get_progress(ctx.mentor_id)
    return ok(progress)


@router.get("/training/jobs")
async def list_training_jobs(ctx: MentorContext = Depends(require_mentor)):
    """List all clone training jobs for the authenticated mentor."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    jobs = await clone_svc.list_jobs(ctx.mentor_id)
    return ok(jobs)


@router.get("/training/l1-results")
async def get_l1_results(ctx: MentorContext = Depends(require_mentor)):
    """Get L1 extraction results (topics, shades, biography) for review."""
    from bloom_kernel.main import get_clone_service

    clone_svc = get_clone_service()
    if clone_svc is None:
        return fail(503, "Clone service not available", status_code=503)

    results = await clone_svc.get_l1_results(ctx.mentor_id)
    return ok(results)


@router.get("/training/logs/{job_id}")
async def stream_training_logs(
    job_id: str,
    ctx: MentorContext = Depends(require_mentor),
):
    """Stream clone training logs via Server-Sent Events (SSE)."""
    from fastapi.responses import StreamingResponse

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
