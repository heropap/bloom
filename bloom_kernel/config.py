"""Unified configuration for Bloom kernel.

Merges all previous Settings classes (Common, TTS, ASR, RAG, Emotion,
Dialogue) and Go service environment variables into a single class.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class KernelSettings(BaseSettings):
    """Single configuration class for the entire bloom_kernel backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    LOG_LEVEL: str = "INFO"
    DEV_MODE: bool = True  # enables /auth/dev-login

    # ── Database ────────────────────────────────────────────────────────
    POSTGRES_DSN: str = "postgresql+asyncpg://bloom:bloom_dev_password@localhost:5432/bloom"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT / Auth ──────────────────────────────────────────────────────
    JWT_SECRET: str = "bloom-dev-secret"
    JWT_EXPIRY_HOURS: int = 168  # 7 days

    # ── WeChat OAuth ────────────────────────────────────────────────────
    WECHAT_APP_ID: str = ""
    WECHAT_APP_SECRET: str = ""

    # ── LLM Providers ───────────────────────────────────────────────────
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com/v1"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"

    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    )
    DASHSCOPE_MODEL: str = "qwen-plus"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:3b"

    # LLM routing (environment-driven switching)
    LLM_PROVIDER_L1L2: str = "ollama"  # deepseek | dashscope | ollama
    LLM_PROVIDER_L3L5: str = "ollama"  # anthropic | dashscope | ollama

    # ── TTS ──────────────────────────────────────────────────────────────
    TTS_FALLBACK_MODE: str = "edge-tts"
    GPT_SOVITS_API_URL: str = "http://localhost:9880"

    DOUBAO_APP_ID: str = ""
    DOUBAO_API_KEY: str = ""
    DOUBAO_CLUSTER: str = "volcano_tts"
    DOUBAO_RESOURCE_ID: str = "zh_female_jitangnv_saturn_bigtts"
    DOUBAO_VOICE_TYPE: str = "zh_female_jitangnv_saturn_bigtts"

    EDGE_TTS_VOICE_FEMALE: str = "zh-CN-XiaoxiaoNeural"
    EDGE_TTS_VOICE_MALE: str = "zh-CN-YunxiNeural"

    # ── ASR ──────────────────────────────────────────────────────────────
    ASR_MODEL_SIZE: str = "base"
    ASR_DEVICE: str = "cpu"
    ASR_COMPUTE_TYPE: str = "int8"
    ASR_BEAM_SIZE: int = 5
    ASR_LANGUAGE: str = "zh"

    # ── Emotion ─────────────────────────────────────────────────────────
    EMOTION_USE_MODEL: bool = False
    EMOTION_MODEL_NAME: str = ""
    EMOTION_SAMPLE_RATE: int = 16000

    # ── RAG / Embeddings ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    USE_LIGHTWEIGHT_MODELS: bool = True
    RERANKER_FALLBACK: bool = True

    # ── L0 Ingestion ──────────────────────────────────────────────────
    L0_CHUNK_SIZE: int = 4000       # ~4000 chars ≈ 2000-4000 tokens for Chinese
    L0_CHUNK_OVERLAP: int = 200     # overlap between adjacent chunks

    # ── Dialogue ────────────────────────────────────────────────────────
    CONTEXT_WINDOW_SIZE: int = 10

    # ── File Storage ────────────────────────────────────────────────────
    DATA_DIR: str = "./data"
    UPLOAD_DIR: str = "./data/uploads"
    MODEL_DIR: str = "./data/models"

    # ── AI Clone Training ─────────────────────────────────────────────
    TRAINING_DIR: str = "./data/training"
    PROGRESS_DIR: str = "./data/progress"
    CLONE_BASE_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    CLONE_LORA_RANK: int = 16
    CLONE_LORA_ALPHA: int = 32
    CLONE_LEARNING_RATE: float = 2e-4
    CLONE_NUM_EPOCHS: int = 3
    CLONE_BATCH_SIZE: int = 4
    CLONE_MAX_SEQ_LENGTH: int = 2048

    # Clone pipeline durability
    CLONE_MAX_RETRIES: int = 5          # max job-level retries before permanent failure
    CLONE_HEARTBEAT_INTERVAL: int = 30  # seconds between heartbeat updates
    CLONE_ORPHAN_TIMEOUT: int = 300     # seconds before a job is considered orphaned (5 min)
    CLONE_LLM_MAX_RETRIES: int = 3     # max retries per LLM call
    CLONE_SHUTDOWN_TIMEOUT: int = 10    # seconds to wait for graceful shutdown

    # L1 Knowledge Extraction
    CLONE_L1_CLUSTER_COUNT: int = 8
    CLONE_L1_MAX_CHUNKS: int = 500

    # L2 Training Data Synthesis (6 types, aligned with Second-Me)
    CLONE_L2_QA_COUNT: int = 300
    CLONE_L2_IDENTITY_COUNT: int = 80
    CLONE_L2_DIALOGUE_COUNT: int = 60
    CLONE_L2_PREFERENCE_COUNT: int = 60
    CLONE_L2_DIVERSITY_COUNT: int = 40
    CLONE_L2_CONTEXT_COUNT: int = 40


# Singleton instance, created once on import
settings = KernelSettings()
