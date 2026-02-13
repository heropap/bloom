"""Bio Generator — mentor biography from topics + shades.

Aligned with Second-Me's StatusBioGenerator: synthesizes a mentor
biography in multiple forms (second-person, third-person, summary).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from bloom_kernel.engine.prompt_templates.l1_topic_extraction import (
    BIOGRAPHY_GENERATION_PROMPT,
)
from bloom_kernel.pipeline.l1.shade_generator import Shade
from bloom_kernel.pipeline.l1.topics_generator import TopicCluster, _parse_json


@dataclass
class MentorBiography:
    """Multi-form mentor biography."""

    second_person: str  # "你..."
    third_person: str   # "导师名..."
    summary: str        # One-liner

    def to_dict(self) -> dict:
        return {
            "second_person": self.second_person,
            "third_person": self.third_person,
            "summary": self.summary,
        }


class BioGenerator:
    """Generate mentor biography from L1 topics and shades."""

    def __init__(self, llm_client: object) -> None:
        self._llm = llm_client

    async def generate(
        self,
        topics: list[TopicCluster],
        shades: list[Shade],
        mentor_name: str = "",
        mentor_bio: str = "",
        on_progress: object | None = None,
    ) -> MentorBiography:
        """Generate mentor biography from topics and shades.

        Args:
            topics: L1 topic clusters
            shades: L1 personality dimensions
            mentor_name: Mentor display name
            mentor_bio: Mentor bio text
            on_progress: Optional async callback(pct, message)

        Returns:
            MentorBiography with three forms.
        """
        if on_progress:
            await on_progress(26, "Generating mentor biography...")

        topics_json = json.dumps(
            [{"topic_name": t.topic_name, "summary": t.summary}
             for t in topics],
            ensure_ascii=False,
            indent=2,
        )

        shades_json = json.dumps(
            [{"title": s.title, "description_2p": s.description_2p}
             for s in shades],
            ensure_ascii=False,
            indent=2,
        )

        prompt = BIOGRAPHY_GENERATION_PROMPT.format(
            mentor_name=mentor_name or "Unknown",
            mentor_bio=mentor_bio or "No bio provided",
            topics_json=topics_json,
            shades_json=shades_json,
        )

        try:
            response = await self._call_llm(prompt)
            parsed = _parse_json(response)

            bio = MentorBiography(
                second_person=parsed.get("second_person", ""),
                third_person=parsed.get("third_person", ""),
                summary=parsed.get("summary", ""),
            )

            logger.info(
                f"BioGenerator: biography generated "
                f"(2p={len(bio.second_person)} chars, 3p={len(bio.third_person)} chars)"
            )

            if on_progress:
                await on_progress(30, "Biography generated")

            return bio

        except Exception as e:
            logger.warning(f"Biography generation failed: {e}")
            name = mentor_name or "该导师"
            expertise = ", ".join(t.topic_name for t in topics[:3])
            return MentorBiography(
                second_person=f"你是一位专注于{expertise}的导师，有着丰富的经验和独到的见解。",
                third_person=f"{name}是一位专注于{expertise}的导师，致力于为每位学员提供温暖的陪伴和专业的指导。",
                summary=f"{name}——{expertise}领域的温暖引路人",
            )

    async def _call_llm(self, prompt: str) -> str:
        from bloom_kernel.engine.llm_client import LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.chat_with_retry(
            messages=messages, max_tokens=2048, temperature=0.7, emotion_level=1
        )
        return response.content
