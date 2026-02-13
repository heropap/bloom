"""L1 Generator — orchestrates Topics + Shades + Bio generators.

Coordinates the three L1 sub-generators and produces the complete
L1Result used by L2 training data synthesis and the clone service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from bloom_kernel.config import KernelSettings
from bloom_kernel.engine.prompt_templates.l1_topic_extraction import (
    MENTOR_PROFILE_PROMPT,
    PERSONA_PROMPT_GENERATION,
)
from bloom_kernel.pipeline.l1.bio_generator import BioGenerator, MentorBiography
from bloom_kernel.pipeline.l1.shade_generator import Shade, ShadeGenerator
from bloom_kernel.pipeline.l1.topics_generator import TopicCluster, TopicsGenerator, _parse_json


@dataclass
class L1Result:
    """Complete L1 extraction output."""

    topics: list[TopicCluster]
    shades: list[Shade]
    biography: MentorBiography
    # Legacy profile fields (backward compat with old L1KnowledgeExtractor)
    background: str = ""
    style: str = ""
    core_beliefs: list[str] = field(default_factory=list)
    expertise: list[str] = field(default_factory=list)
    tone: str = ""
    counseling_approach: str = ""
    persona_prompt: str = ""

    def to_dict(self) -> dict:
        return {
            "topics": [t.to_dict() for t in self.topics],
            "shades": [s.to_dict() for s in self.shades],
            "biography": self.biography.to_dict(),
            "background": self.background,
            "style": self.style,
            "core_beliefs": self.core_beliefs,
            "expertise": self.expertise,
            "tone": self.tone,
            "counseling_approach": self.counseling_approach,
            "persona_prompt": self.persona_prompt,
        }


class L1Generator:
    """Orchestrate L1 extraction: Topics -> Shades -> Bio -> Profile -> Persona.

    This is the main entry point for L1 knowledge extraction, replacing
    the old monolithic L1KnowledgeExtractor.
    """

    def __init__(
        self,
        settings: KernelSettings,
        llm_client: object,
        embedding_model: object | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._topics_gen = TopicsGenerator(settings, llm_client, embedding_model)
        self._shade_gen = ShadeGenerator(llm_client)
        self._bio_gen = BioGenerator(llm_client)

    async def run(
        self,
        mentor_id: str,
        chunks: list[dict],
        mentor_name: str = "",
        mentor_bio: str = "",
        on_progress: object | None = None,
    ) -> L1Result:
        """Run the complete L1 extraction pipeline.

        Pipeline:
        1. TopicsGenerator — cluster + summarize → topics
        2. ShadeGenerator — topics + chunks → personality shades
        3. BioGenerator — topics + shades → biography
        4. Profile generation — topics → profile (legacy compat)
        5. Persona generation — profile → persona_prompt

        Args:
            mentor_id: Mentor UUID string
            chunks: L0 chunk dicts with 'chunk_id', 'content', 'metadata'
            mentor_name: Mentor display name
            mentor_bio: Mentor bio text
            on_progress: Optional async callback(pct: int, message: str)

        Returns:
            L1Result with topics, shades, biography, profile, and persona.
        """
        logger.info(f"L1 extraction starting: mentor={mentor_id}, chunks={len(chunks)}")

        # Limit chunks
        max_chunks = self._settings.CLONE_L1_MAX_CHUNKS
        if len(chunks) > max_chunks:
            logger.info(f"Truncating chunks from {len(chunks)} to {max_chunks}")
            chunks = chunks[:max_chunks]

        # Step 1: Topics
        topics = await self._topics_gen.generate(chunks, on_progress)

        # Step 2: Shades
        shades = await self._shade_gen.generate(
            topics, chunks, mentor_name, mentor_bio, on_progress
        )

        # Step 3: Biography
        biography = await self._bio_gen.generate(
            topics, shades, mentor_name, mentor_bio, on_progress
        )

        # Step 4: Legacy profile (for backward compat)
        if on_progress:
            await on_progress(30, "Generating mentor profile...")
        profile_data = await self._generate_profile(topics, mentor_name, mentor_bio)

        # Step 5: Persona prompt
        if on_progress:
            await on_progress(32, "Generating persona prompt...")
        persona = await self._generate_persona(profile_data, shades, biography, mentor_name)

        if on_progress:
            await on_progress(33, "L1 knowledge extraction complete")

        result = L1Result(
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

        logger.info(
            f"L1 complete: {len(topics)} topics, {len(shades)} shades, "
            f"persona={len(persona)} chars"
        )
        return result

    async def _generate_profile(
        self,
        topics: list[TopicCluster],
        mentor_name: str,
        mentor_bio: str,
    ) -> dict:
        """Generate mentor profile from topics (legacy format)."""
        topics_json = json.dumps(
            [{"topic_name": t.topic_name, "summary": t.summary, "key_points": t.key_points}
             for t in topics],
            ensure_ascii=False,
            indent=2,
        )

        prompt = MENTOR_PROFILE_PROMPT.format(
            mentor_name=mentor_name or "Unknown",
            mentor_bio=mentor_bio or "No bio provided",
            topics_json=topics_json,
        )

        try:
            response = await self._call_llm(prompt)
            return _parse_json(response)
        except Exception as e:
            logger.warning(f"Profile generation failed: {e}")
            return {}

    async def _generate_persona(
        self,
        profile_data: dict,
        shades: list[Shade],
        biography: MentorBiography,
        mentor_name: str,
    ) -> str:
        """Generate persona prompt incorporating profile, shades, and biography."""
        # Enrich the profile with shade and biography info
        enriched_profile = {
            **profile_data,
            "personality_dimensions": [
                {"title": s.title, "description": s.description_2p}
                for s in shades
            ],
            "biography_summary": biography.summary,
        }

        profile_json = json.dumps(enriched_profile, ensure_ascii=False, indent=2)

        prompt = PERSONA_PROMPT_GENERATION.format(
            mentor_name=mentor_name or "Unknown",
            profile_json=profile_json,
        )

        try:
            return await self._call_llm(prompt)
        except Exception as e:
            logger.warning(f"Persona generation failed: {e}")
            return f"你是{mentor_name}导师。{biography.summary}"

    async def _call_llm(self, prompt: str) -> str:
        from bloom_kernel.engine.llm_client import LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.chat_with_retry(
            messages=messages, max_tokens=2048, temperature=0.7, emotion_level=1
        )
        return response.content
