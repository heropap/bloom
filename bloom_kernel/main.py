"""Bloom Kernel — unified FastAPI application entry point.

Replaces 5 Go services + 5 Python services with a single process.
Run with: uvicorn bloom_kernel.main:app --host 0.0.0.0 --port 8002
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from fastapi.responses import JSONResponse

from bloom_kernel.config import settings
from bloom_kernel.utils.database import db
from bloom_kernel.utils.response import ok

# Import all ORM models so SQLAlchemy metadata is complete
import bloom_kernel.models  # noqa: F401

# ── Global engine/service instances (initialized in lifespan) ────────────

_tts_synthesizer = None
_asr_engine = None
_emotion_recognizer = None
_llm_client = None
_retriever = None
_dialogue_service = None
_ingestion_pipeline = None
_clone_service = None


def get_tts_synthesizer():
    return _tts_synthesizer


def get_asr_engine():
    return _asr_engine


def get_emotion_recognizer():
    return _emotion_recognizer


def get_dialogue_service():
    return _dialogue_service


def get_ingestion_pipeline():
    return _ingestion_pipeline


def get_llm_client():
    return _llm_client


def get_clone_service():
    return _clone_service


# ── Lifespan ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down all resources."""
    global _tts_synthesizer, _asr_engine, _emotion_recognizer
    global _llm_client, _retriever, _dialogue_service
    global _ingestion_pipeline, _clone_service

    logger.info("Bloom Kernel starting up...")

    # ── Database connections ──
    try:
        await db.connect(settings.POSTGRES_DSN, settings.REDIS_URL)
    except Exception as e:
        logger.warning(f"Database connection failed (will retry on first request): {e}")

    # ── TTS Engine ──
    try:
        from bloom_kernel.engine.tts.synthesizer import create_synthesizer

        _tts_synthesizer = await create_synthesizer(
            gpt_sovits_api_url=settings.GPT_SOVITS_API_URL,
            fallback_mode=settings.TTS_FALLBACK_MODE,
            doubao_app_id=settings.DOUBAO_APP_ID,
            doubao_api_key=settings.DOUBAO_API_KEY,
            doubao_cluster=settings.DOUBAO_CLUSTER,
            doubao_resource_id=settings.DOUBAO_RESOURCE_ID,
            doubao_voice_type=settings.DOUBAO_VOICE_TYPE,
        )
        logger.info(f"TTS engine ready: {_tts_synthesizer.backend_name}")
    except Exception as e:
        logger.warning(f"TTS initialization failed: {e}")

    # ── ASR Engine ──
    try:
        from bloom_kernel.engine.asr.transcriber import Transcriber

        _asr_engine = Transcriber(
            model_size=settings.ASR_MODEL_SIZE,
            device=settings.ASR_DEVICE,
            compute_type=settings.ASR_COMPUTE_TYPE,
            beam_size=settings.ASR_BEAM_SIZE,
            language=settings.ASR_LANGUAGE,
        )
        _asr_engine.load_model()
        logger.info("ASR engine ready")
    except Exception as e:
        logger.warning(f"ASR initialization failed: {e}")

    # ── Emotion Recognizer ──
    try:
        from bloom_kernel.engine.emotion.recognizer import create_recognizer

        _emotion_recognizer = create_recognizer(
            use_model=settings.EMOTION_USE_MODEL,
            model_name=settings.EMOTION_MODEL_NAME,
        )
        await _emotion_recognizer.startup()
        logger.info("Emotion recognizer ready")
    except Exception as e:
        logger.warning(f"Emotion recognizer initialization failed: {e}")

    # ── LLM Client ──
    try:
        from bloom_kernel.engine.llm_client import LLMClient

        _llm_client = LLMClient(settings)
        await _llm_client.startup()
        logger.info("LLM client ready")
    except Exception as e:
        logger.warning(f"LLM client initialization failed: {e}")

    # ── RAG Retriever ──
    try:
        from bloom_kernel.engine.rag.retriever import HybridRetriever

        _retriever = HybridRetriever(settings)
        await _retriever.startup()
        logger.info("RAG retriever ready")
    except Exception as e:
        logger.warning(f"RAG retriever initialization failed: {e}")

    # ── Dialogue Service ──
    try:
        from bloom_kernel.services.dialogue_service import DialogueService
        from bloom_kernel.services.mentor_service import MentorService
        from bloom_kernel.services.safety_service import SafetyService
        from bloom_kernel.services.session_service import SessionService

        _dialogue_service = DialogueService(
            settings=settings,
            db=db,
            llm_client=_llm_client,
            retriever=_retriever,
            safety_service=SafetyService(),
            mentor_service=MentorService(),
            session_service=SessionService(redis=db.redis),
        )
        logger.info("Dialogue service ready")
    except Exception as e:
        logger.warning(f"Dialogue service initialization failed: {e}")

    # ── Ingestion Pipeline ──
    try:
        from bloom_kernel.pipeline.l0_ingest import IngestionPipeline

        _ingestion_pipeline = IngestionPipeline(settings)
        await _ingestion_pipeline.startup()
        logger.info("Ingestion pipeline ready")
    except Exception as e:
        logger.warning(f"Ingestion pipeline initialization failed: {e}")

    # ── Clone Service ──
    try:
        from bloom_kernel.services.clone_service import CloneService

        _clone_service = CloneService(
            settings=settings,
            db=db,
            llm_client=_llm_client,
            ingestion_pipeline=_ingestion_pipeline,
        )
        logger.info("Clone service ready")

        # Recover orphaned clone jobs from previous crashes
        try:
            recovered = await _clone_service.recover_orphan_jobs()
            if recovered:
                logger.info(f"Recovered {recovered} orphan clone jobs")
        except Exception as e:
            logger.warning(f"Orphan job recovery failed: {e}")
    except Exception as e:
        logger.warning(f"Clone service initialization failed: {e}")

    logger.info("Bloom Kernel startup complete")

    yield

    # ── Shutdown ──
    logger.info("Bloom Kernel shutting down...")

    # Gracefully stop active clone pipelines (save checkpoints)
    if _clone_service:
        try:
            await _clone_service.graceful_shutdown()
        except Exception as e:
            logger.warning(f"Clone service shutdown error: {e}")

    if _ingestion_pipeline:
        await _ingestion_pipeline.shutdown()
    if _llm_client:
        await _llm_client.shutdown()
    if _retriever:
        await _retriever.shutdown()
    if _emotion_recognizer:
        await _emotion_recognizer.shutdown()
    if _asr_engine:
        _asr_engine.unload_model()

    await db.disconnect()
    logger.info("Bloom Kernel shutdown complete")


# ── App creation ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Bloom Kernel",
    description="Bloom (绽放) — AI voice companion platform unified backend",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers for infrastructure issues
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc):
    msg = str(exc)
    if "not connected" in msg.lower() or "database" in msg.lower():
        return JSONResponse(
            status_code=503,
            content={"code": 503, "message": f"Database unavailable: {msg}", "data": None},
        )
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": msg, "data": None},
    )


@app.exception_handler(ConnectionRefusedError)
async def connection_refused_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"code": 503, "message": "Infrastructure unavailable (postgres/redis not running)", "data": None},
    )


@app.exception_handler(ConnectionError)
async def connection_error_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={"code": 503, "message": f"Connection error: {exc}", "data": None},
    )

# Register API routes
from bloom_kernel.api.router import create_api_router

app.include_router(create_api_router())


# ── Health check ─────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Global health check endpoint."""
    return ok({
        "status": "ok",
        "service": "bloom-kernel",
        "tts": _tts_synthesizer.backend_name if _tts_synthesizer else "unavailable",
        "asr": "ready" if _asr_engine and _asr_engine._model is not None else "unavailable",
        "emotion": "ready" if _emotion_recognizer else "unavailable",
        "llm": "ready" if _llm_client else "unavailable",
        "rag": "ready" if _retriever else "unavailable",
        "dialogue": "ready" if _dialogue_service else "unavailable",
        "ingestion": "ready" if _ingestion_pipeline else "unavailable",
        "clone": "ready" if _clone_service else "unavailable",
    })


@app.get("/")
async def root():
    """Root endpoint."""
    return ok({"service": "bloom-kernel", "version": "0.1.0"})
