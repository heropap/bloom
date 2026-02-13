"""Shade Generator — personality dimensions from topics + chunks.

Aligned with Second-Me's ShadeGenerator: extracts unique personality
dimensions ("shades") that differentiate this mentor from others.
Each shade has second-person and third-person descriptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from bloom_kernel.engine.prompt_templates.l1_topic_extraction import (
    SHADE_GENERATION_PROMPT,
)
from bloom_kernel.pipeline.l1.topics_generator import TopicCluster, _parse_json


@dataclass
class Shade:
    """A personality dimension of a mentor."""

    title: str
    description_2p: str  # Second-person ("你...")
    description_3p: str  # Third-person ("导师名...")
    keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description_2p": self.description_2p,
            "description_3p": self.description_3p,
            "keywords": self.keywords,
        }


class ShadeGenerator:
    """Generate personality dimensions from L1 topics and L0 chunks."""

    def __init__(self, llm_client: object) -> None:
        self._llm = llm_client

    async def generate(
        self,
        topics: list[TopicCluster],
        chunks: list[dict],
        mentor_name: str = "",
        mentor_bio: str = "",
        on_progress: object | None = None,
    ) -> list[Shade]:
        """Generate shades from topics and representative chunks.

        Args:
            topics: L1 topic clusters
            chunks: Original L0 chunks (for sampling representative text)
            mentor_name: Mentor display name
            mentor_bio: Mentor bio text
            on_progress: Optional async callback(pct, message)

        Returns:
            List of Shade objects (typically 5-8).
        """
        if on_progress:
            await on_progress(15, "Generating personality dimensions (shades)...")

        topics_json = json.dumps(
            [{"topic_name": t.topic_name, "summary": t.summary, "key_points": t.key_points}
             for t in topics],
            ensure_ascii=False,
            indent=2,
        )

        # Sample representative chunks (up to 6000 chars)
        sample_texts: list[str] = []
        total_len = 0
        for c in chunks:
            text = c["content"][:600]
            if total_len + len(text) > 6000:
                break
            sample_texts.append(text)
            total_len += len(text)
        sample_chunks = "\n---\n".join(sample_texts)

        prompt = SHADE_GENERATION_PROMPT.format(
            mentor_name=mentor_name or "Unknown",
            mentor_bio=mentor_bio or "No bio provided",
            topics_json=topics_json,
            sample_chunks=sample_chunks,
        )

        try:
            response = await self._call_llm(prompt)
            parsed = _parse_json(response)
            shades_raw = parsed.get("shades", [])

            shades = [
                Shade(
                    title=s.get("title", f"Shade {i + 1}"),
                    description_2p=s.get("description_2p", ""),
                    description_3p=s.get("description_3p", ""),
                    keywords=s.get("keywords", []),
                )
                for i, s in enumerate(shades_raw)
            ]

            logger.info(f"ShadeGenerator: produced {len(shades)} shades")

            if on_progress:
                await on_progress(22, f"Generated {len(shades)} personality dimensions")

            return shades

        except Exception as e:
            logger.warning(f"Shade generation failed: {e}")
            # Fallback: generate one shade per topic
            return [
                Shade(
                    title=t.topic_name,
                    description_2p=f"你在{t.topic_name}方面有深入的见解和丰富的经验。",
                    description_3p=f"{mentor_name or '该导师'}擅长{t.topic_name}领域的指导和陪伴。",
                    keywords=t.expertise_areas[:3],
                )
                for t in topics[:8]
            ]

    async def _call_llm(self, prompt: str) -> str:
        from bloom_kernel.engine.llm_client import LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.chat_with_retry(
            messages=messages, max_tokens=3000, temperature=0.7, emotion_level=1
        )
        return response.content
