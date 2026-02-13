"""Multi-model LLM client for Bloom AI.

Routes requests between:
- DeepSeek-V3 for daily conversations (lower cost, fast)
- Claude Sonnet 4.5 for deep emotional support (higher quality empathy)
- DashScope (Qwen-Plus) for Chinese-optimized conversations

Model selection is based on the emotional state / conversation context.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import httpx
from loguru import logger

from bloom_kernel.config import KernelSettings


class ModelProvider(str, enum.Enum):
    """Supported LLM providers."""

    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    DASHSCOPE = "dashscope"
    OLLAMA = "ollama"


@dataclass
class LLMMessage:
    """A single message in a conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    provider: ModelProvider
    usage: dict[str, int] = field(default_factory=dict)


def select_model(emotion_level: int) -> ModelProvider:
    """Route to the appropriate model based on emotional context.

    Args:
        emotion_level: The current emotion intensity level (1-5).
            L1-L2: Normal / slight fluctuation -> DeepSeek (cost-effective)
            L3-L5: Anxiety through crisis -> Anthropic Claude (superior empathy)

    Returns:
        The model provider to use for this interaction.
    """
    match emotion_level:
        case 1 | 2:
            return ModelProvider.DEEPSEEK
        case 3 | 4 | 5:
            return ModelProvider.ANTHROPIC
        case _:
            logger.warning(f"Unknown emotion level {emotion_level}, defaulting to DeepSeek")
            return ModelProvider.DEEPSEEK


class LLMClient:
    """Async multi-model LLM client.

    Manages HTTP connections to multiple LLM providers and routes
    requests based on conversation context.
    """

    def __init__(self, settings: KernelSettings) -> None:
        self._settings = settings
        self._deepseek_client: httpx.AsyncClient | None = None
        self._anthropic_client: httpx.AsyncClient | None = None
        self._dashscope_client: httpx.AsyncClient | None = None
        self._ollama_client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        """Initialize HTTP clients. Call during application startup."""
        self._deepseek_client = httpx.AsyncClient(
            base_url=self._settings.DEEPSEEK_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._settings.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._anthropic_client = httpx.AsyncClient(
            base_url=self._settings.ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": self._settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
        if self._settings.DASHSCOPE_API_KEY:
            self._dashscope_client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._settings.DASHSCOPE_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        if self._settings.OLLAMA_BASE_URL:
            self._ollama_client = httpx.AsyncClient(
                base_url=f"{self._settings.OLLAMA_BASE_URL}/v1",
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        logger.info("LLM clients initialized")

    async def shutdown(self) -> None:
        """Close HTTP clients. Call during application shutdown."""
        if self._deepseek_client:
            await self._deepseek_client.aclose()
        if self._anthropic_client:
            await self._anthropic_client.aclose()
        if self._dashscope_client:
            await self._dashscope_client.aclose()
        if self._ollama_client:
            await self._ollama_client.aclose()
        logger.info("LLM clients closed")

    async def chat_with_retry(
        self,
        messages: list[LLMMessage],
        emotion_level: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        provider_override: ModelProvider | None = None,
        max_retries: int | None = None,
    ) -> LLMResponse:
        """Send a chat request with automatic retry on transient errors.

        Same signature as chat() plus an optional max_retries override.
        """
        from bloom_kernel.engine.llm_retry import retry_llm_call

        if max_retries is None:
            from bloom_kernel.config import settings
            max_retries = settings.CLONE_LLM_MAX_RETRIES

        return await retry_llm_call(
            self.chat,
            messages,
            emotion_level=emotion_level,
            max_tokens=max_tokens,
            temperature=temperature,
            provider_override=provider_override,
            max_retries=max_retries,
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        emotion_level: int = 1,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        provider_override: ModelProvider | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the appropriate model."""
        provider = provider_override or self._select_model(emotion_level)
        logger.debug(f"Routing to {provider.value} (emotion_level={emotion_level})")

        match provider:
            case ModelProvider.DEEPSEEK:
                return await self._call_deepseek(messages, max_tokens, temperature)
            case ModelProvider.ANTHROPIC:
                return await self._call_anthropic(messages, max_tokens, temperature)
            case ModelProvider.DASHSCOPE:
                return await self._call_dashscope(messages, max_tokens, temperature)
            case ModelProvider.OLLAMA:
                return await self._call_ollama(messages, max_tokens, temperature)

    def _select_model(self, emotion_level: int) -> ModelProvider:
        """Route to the appropriate model based on emotional context and config."""
        try:
            l1l2_provider = ModelProvider(self._settings.LLM_PROVIDER_L1L2)
        except ValueError:
            l1l2_provider = ModelProvider.DEEPSEEK

        try:
            l3l5_provider = ModelProvider(self._settings.LLM_PROVIDER_L3L5)
        except ValueError:
            l3l5_provider = ModelProvider.ANTHROPIC

        if emotion_level <= 2:
            return l1l2_provider
        return l3l5_provider

    async def _call_deepseek(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call DeepSeek API (OpenAI-compatible format)."""
        if not self._deepseek_client:
            raise RuntimeError("DeepSeek client not initialized. Call startup() first.")

        payload = {
            "model": self._settings.DEEPSEEK_MODEL,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await self._deepseek_client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()

        return LLMResponse(
            content=body["choices"][0]["message"]["content"],
            model=body.get("model", self._settings.DEEPSEEK_MODEL),
            provider=ModelProvider.DEEPSEEK,
            usage=body.get("usage", {}),
        )

    async def _call_anthropic(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call Anthropic Messages API."""
        if not self._anthropic_client:
            raise RuntimeError("Anthropic client not initialized. Call startup() first.")

        system_text = ""
        conversation: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                system_text += msg.content + "\n"
            else:
                conversation.append({"role": msg.role, "content": msg.content})

        payload: dict = {
            "model": self._settings.ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system_text.strip():
            payload["system"] = system_text.strip()

        response = await self._anthropic_client.post("/messages", json=payload)
        response.raise_for_status()
        body = response.json()

        content_blocks = body.get("content", [])
        text = "".join(
            block["text"] for block in content_blocks if block.get("type") == "text"
        )

        return LLMResponse(
            content=text,
            model=body.get("model", self._settings.ANTHROPIC_MODEL),
            provider=ModelProvider.ANTHROPIC,
            usage=body.get("usage", {}),
        )

    async def _call_dashscope(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call DashScope API (Qwen-Plus, NON-OpenAI-compatible format).

        DashScope uses ``input.prompt`` (a single string) rather than a
        ``messages`` array.
        """
        if not self._dashscope_client:
            raise RuntimeError("DashScope client not initialized. Call startup() first.")

        prompt_parts: list[str] = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"[系统指令]\n{msg.content}\n")
            elif msg.role == "user":
                prompt_parts.append(f"[用户]\n{msg.content}\n")
            elif msg.role == "assistant":
                prompt_parts.append(f"[助手]\n{msg.content}\n")
        prompt = "\n".join(prompt_parts)

        payload = {
            "model": self._settings.DASHSCOPE_MODEL,
            "input": {"prompt": prompt},
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        }

        response = await self._dashscope_client.post(
            self._settings.DASHSCOPE_BASE_URL,
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

        return LLMResponse(
            content=body["output"]["text"],
            model=self._settings.DASHSCOPE_MODEL,
            provider=ModelProvider.DASHSCOPE,
            usage=body.get("usage", {}),
        )

    async def _call_ollama(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        """Call Ollama API (OpenAI-compatible format)."""
        if not self._ollama_client:
            raise RuntimeError("Ollama client not initialized. Call startup() first.")

        payload = {
            "model": self._settings.OLLAMA_MODEL,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await self._ollama_client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()

        return LLMResponse(
            content=body["choices"][0]["message"]["content"],
            model=body.get("model", self._settings.OLLAMA_MODEL),
            provider=ModelProvider.OLLAMA,
            usage=body.get("usage", {}),
        )
