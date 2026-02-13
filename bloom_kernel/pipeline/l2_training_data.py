"""L2 Training Data Synthesis — generate training datasets from L1 knowledge.

Takes L1 knowledge profile + chunks and produces six types of training data
aligned with Second-Me HMM architecture:
1. QA pairs from knowledge chunks
2. Identity dialogue pairs (self-awareness)
3. Multi-turn counseling dialogues (deep-night companion scenarios)
4. Preference pairs (value/choice expression)
5. Diversity responses (varied thinking patterns)
6. Context-enhanced dialogues (RAG-grounded conversations)

All outputs merged into ChatML format for LoRA fine-tuning.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from bloom_kernel.config import KernelSettings
from bloom_kernel.engine.prompt_templates.l2_synthesis_prompts import (
    CONTEXT_GENERATION_PROMPT,
    COUNSELING_DIALOGUE_PROMPT,
    DIVERSITY_GENERATION_PROMPT,
    IDENTITY_GENERATION_PROMPT,
    PREFERENCE_GENERATION_PROMPT,
    QA_GENERATION_PROMPT,
)
from bloom_kernel.pipeline.l1 import L1Result


@dataclass
class L2SynthesisResult:
    """Output of L2 training data synthesis."""

    qa_count: int = 0
    identity_count: int = 0
    dialogue_count: int = 0
    preference_count: int = 0
    diversity_count: int = 0
    context_count: int = 0
    total_samples: int = 0
    output_path: str = ""

    def to_dict(self) -> dict:
        return {
            "qa_count": self.qa_count,
            "identity_count": self.identity_count,
            "dialogue_count": self.dialogue_count,
            "preference_count": self.preference_count,
            "diversity_count": self.diversity_count,
            "context_count": self.context_count,
            "total_samples": self.total_samples,
            "output_path": self.output_path,
        }


class L2TrainingDataSynthesizer:
    """Generate training datasets from L1 knowledge profile."""

    def __init__(self, settings: KernelSettings, llm_client: object) -> None:
        self._settings = settings
        self._llm = llm_client

    async def synthesize(
        self,
        mentor_id: str,
        profile: L1Result,
        chunks: list[dict],
        mentor_name: str = "",
        on_progress: object | None = None,
    ) -> L2SynthesisResult:
        """Run full L2 synthesis pipeline.

        Args:
            mentor_id: Mentor UUID string
            profile: L1 knowledge profile
            chunks: L0 chunk dicts with 'content'
            mentor_name: Mentor name for prompts
            on_progress: Optional async callback(pct: int, message: str)
        """
        logger.info(f"L2 synthesis starting: mentor={mentor_id}")

        # Create output directory
        training_dir = Path(self._settings.TRAINING_DIR) / mentor_id
        training_dir.mkdir(parents=True, exist_ok=True)

        all_samples: list[dict] = []

        # Step 1: Generate QA pairs from chunks
        if on_progress:
            await on_progress(35, "Generating QA pairs from knowledge...")
        qa_pairs = await self._generate_qa_pairs(chunks, mentor_name, profile)
        all_samples.extend(qa_pairs)
        logger.info(f"L2: Generated {len(qa_pairs)} QA pairs")

        # Step 2: Generate identity dialogues
        if on_progress:
            await on_progress(40, "Generating identity dialogues...")
        identity_pairs = await self._generate_identity_pairs(profile, mentor_name)
        all_samples.extend(identity_pairs)
        logger.info(f"L2: Generated {len(identity_pairs)} identity pairs")

        # Step 3: Generate counseling dialogues
        if on_progress:
            await on_progress(45, "Generating counseling dialogues...")
        counseling_samples = await self._generate_counseling_dialogues(profile, mentor_name)
        all_samples.extend(counseling_samples)
        logger.info(f"L2: Generated {len(counseling_samples)} counseling samples")

        # Step 4: Generate preference pairs
        if on_progress:
            await on_progress(50, "Generating preference pairs...")
        preference_pairs = await self._generate_preference_pairs(profile, mentor_name)
        all_samples.extend(preference_pairs)
        logger.info(f"L2: Generated {len(preference_pairs)} preference pairs")

        # Step 5: Generate diversity responses
        if on_progress:
            await on_progress(55, "Generating diversity responses...")
        diversity_pairs = await self._generate_diversity_responses(profile, mentor_name)
        all_samples.extend(diversity_pairs)
        logger.info(f"L2: Generated {len(diversity_pairs)} diversity responses")

        # Step 6: Generate context-enhanced dialogues
        if on_progress:
            await on_progress(59, "Generating context-enhanced dialogues...")
        context_pairs = await self._generate_context_dialogues(chunks, mentor_name, profile)
        all_samples.extend(context_pairs)
        logger.info(f"L2: Generated {len(context_pairs)} context dialogues")

        # Step 7: Convert to ChatML format and save
        if on_progress:
            await on_progress(62, "Merging training data...")
        output_path = str(training_dir / "merged.json")
        chatml_data = self._to_chatml(all_samples, profile.persona_prompt)
        await asyncio.to_thread(self._save_json, chatml_data, output_path)

        result = L2SynthesisResult(
            qa_count=len(qa_pairs),
            identity_count=len(identity_pairs),
            dialogue_count=len(counseling_samples),
            preference_count=len(preference_pairs),
            diversity_count=len(diversity_pairs),
            context_count=len(context_pairs),
            total_samples=len(chatml_data),
            output_path=output_path,
        )
        logger.info(f"L2 synthesis complete: {result.total_samples} total samples → {output_path}")

        if on_progress:
            await on_progress(66, "L2 training data synthesis complete")

        return result

    # ── Existing generators ──────────────────────────────────────────────

    async def _generate_qa_pairs(
        self,
        chunks: list[dict],
        mentor_name: str,
        profile: L1Result,
    ) -> list[dict]:
        """Generate QA pairs from knowledge chunks."""
        target = self._settings.CLONE_L2_QA_COUNT
        pairs: list[dict] = []

        # Process chunks in batches, each batch generates ~10-20 QA pairs
        batch_size = 3
        pairs_per_batch = max(5, target // max(1, len(chunks) // batch_size))

        for i in range(0, len(chunks), batch_size):
            if len(pairs) >= target:
                break

            batch = chunks[i:i + batch_size]
            chunk_content = "\n---\n".join(c["content"][:400] for c in batch)

            prompt = QA_GENERATION_PROMPT.format(
                mentor_name=mentor_name or "导师",
                mentor_style=profile.style or "温暖专业",
                chunk_content=chunk_content,
                count=min(pairs_per_batch, target - len(pairs)),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "type": "qa",
                            "question": item["question"],
                            "answer": item["answer"],
                        })
            except Exception as e:
                logger.warning(f"QA generation batch {i} failed: {e}")

        return pairs[:target]

    async def _generate_identity_pairs(
        self,
        profile: L1Result,
        mentor_name: str,
    ) -> list[dict]:
        """Generate identity/self-awareness dialogue pairs."""
        target = self._settings.CLONE_L2_IDENTITY_COUNT
        pairs: list[dict] = []

        profile_json = json.dumps(profile.to_dict(), ensure_ascii=False, indent=2)

        # Generate in batches of ~20
        batches = max(1, target // 20)
        for batch_idx in range(batches):
            if len(pairs) >= target:
                break

            remaining = target - len(pairs)
            prompt = IDENTITY_GENERATION_PROMPT.format(
                mentor_name=mentor_name or "导师",
                profile_json=profile_json[:3000],  # Limit context
                count=min(20, remaining),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "type": "identity",
                            "question": item["question"],
                            "answer": item["answer"],
                        })
            except Exception as e:
                logger.warning(f"Identity generation batch {batch_idx} failed: {e}")

        return pairs[:target]

    async def _generate_counseling_dialogues(
        self,
        profile: L1Result,
        mentor_name: str,
    ) -> list[dict]:
        """Generate multi-turn counseling dialogue samples."""
        target = self._settings.CLONE_L2_DIALOGUE_COUNT
        samples: list[dict] = []

        # Generate in batches of ~10 dialogues
        batches = max(1, target // 10)
        for batch_idx in range(batches):
            if len(samples) >= target:
                break

            remaining = target - len(samples)
            prompt = COUNSELING_DIALOGUE_PROMPT.format(
                mentor_name=mentor_name or "导师",
                mentor_style=profile.style or "温暖专业",
                counseling_approach=profile.counseling_approach or "先倾听后引导",
                count=min(10, remaining),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "turns" in item and isinstance(item["turns"], list):
                        samples.append({
                            "type": "counseling",
                            "scenario": item.get("scenario", ""),
                            "turns": item["turns"],
                        })
            except Exception as e:
                logger.warning(f"Counseling generation batch {batch_idx} failed: {e}")

        return samples[:target]

    # ── New generators (Second-Me alignment) ─────────────────────────────

    async def _generate_preference_pairs(
        self,
        profile: L1Result,
        mentor_name: str,
    ) -> list[dict]:
        """Generate preference/value-choice training pairs."""
        target = self._settings.CLONE_L2_PREFERENCE_COUNT
        pairs: list[dict] = []

        # Serialize shades for the prompt
        shades_json = json.dumps(
            [{"title": s.title, "description_2p": s.description_2p} for s in profile.shades],
            ensure_ascii=False,
            indent=2,
        ) if profile.shades else "[]"

        core_beliefs = ", ".join(profile.core_beliefs[:5]) if profile.core_beliefs else "温暖陪伴，专业引导"

        # Generate in batches of ~20
        batches = max(1, target // 20)
        for batch_idx in range(batches):
            if len(pairs) >= target:
                break

            remaining = target - len(pairs)
            prompt = PREFERENCE_GENERATION_PROMPT.format(
                mentor_name=mentor_name or "导师",
                shades_json=shades_json[:2000],
                core_beliefs=core_beliefs,
                count=min(20, remaining),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "type": "preference",
                            "question": item["question"],
                            "answer": item["answer"],
                        })
            except Exception as e:
                logger.warning(f"Preference generation batch {batch_idx} failed: {e}")

        return pairs[:target]

    async def _generate_diversity_responses(
        self,
        profile: L1Result,
        mentor_name: str,
    ) -> list[dict]:
        """Generate diverse response styles for common topics."""
        target = self._settings.CLONE_L2_DIVERSITY_COUNT
        pairs: list[dict] = []

        biography_summary = (
            profile.biography.summary if profile.biography else "一位专注于心理陪伴的导师"
        )

        # Generate in batches of ~20
        batches = max(1, target // 20)
        for batch_idx in range(batches):
            if len(pairs) >= target:
                break

            remaining = target - len(pairs)
            prompt = DIVERSITY_GENERATION_PROMPT.format(
                mentor_name=mentor_name or "导师",
                biography_summary=biography_summary,
                mentor_style=profile.style or "温暖专业",
                count=min(20, remaining),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "type": "diversity",
                            "question": item["question"],
                            "answer": item["answer"],
                        })
            except Exception as e:
                logger.warning(f"Diversity generation batch {batch_idx} failed: {e}")

        return pairs[:target]

    async def _generate_context_dialogues(
        self,
        chunks: list[dict],
        mentor_name: str,
        profile: L1Result,
    ) -> list[dict]:
        """Generate context-enhanced dialogues grounded in knowledge chunks."""
        target = self._settings.CLONE_L2_CONTEXT_COUNT
        pairs: list[dict] = []

        # Process chunks in batches, each batch generates ~5-10 context pairs
        batch_size = 2
        pairs_per_batch = max(3, target // max(1, len(chunks) // batch_size))

        for i in range(0, len(chunks), batch_size):
            if len(pairs) >= target:
                break

            batch = chunks[i:i + batch_size]
            chunk_content = "\n---\n".join(c["content"][:600] for c in batch)

            prompt = CONTEXT_GENERATION_PROMPT.format(
                mentor_name=mentor_name or "导师",
                mentor_style=profile.style or "温暖专业",
                chunk_content=chunk_content,
                count=min(pairs_per_batch, target - len(pairs)),
            )

            try:
                response = await self._call_llm(prompt)
                parsed = self._parse_json_array(response)
                for item in parsed:
                    if "question" in item and "answer" in item:
                        pairs.append({
                            "type": "context",
                            "question": item["question"],
                            "answer": item["answer"],
                        })
            except Exception as e:
                logger.warning(f"Context generation batch {i} failed: {e}")

        return pairs[:target]

    # ── Conversion + persistence ─────────────────────────────────────────

    @staticmethod
    def _to_chatml(samples: list[dict], system_prompt: str) -> list[dict]:
        """Convert all samples to ChatML format for training.

        Output format per sample:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }
        """
        chatml_data: list[dict] = []

        for sample in samples:
            if sample["type"] in ("qa", "identity", "preference", "diversity", "context"):
                chatml_data.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": sample["question"]},
                        {"role": "assistant", "content": sample["answer"]},
                    ]
                })
            elif sample["type"] == "counseling":
                messages = [{"role": "system", "content": system_prompt}]
                for turn in sample.get("turns", []):
                    if turn.get("role") in ("user", "assistant"):
                        messages.append({
                            "role": turn["role"],
                            "content": turn["content"],
                        })
                if len(messages) > 1:  # Must have at least system + one turn
                    chatml_data.append({"messages": messages})

        return chatml_data

    @staticmethod
    def _save_json(data: list[dict], path: str) -> None:
        """Save training data to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with a single prompt (with automatic retry on transient errors)."""
        from bloom_kernel.engine.llm_client import LLMMessage

        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.chat_with_retry(
            messages=messages,
            max_tokens=4096,
            temperature=0.8,
            emotion_level=1,
        )
        return response.content

    @staticmethod
    def _parse_json_array(text: str) -> list[dict]:
        """Extract JSON array from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return []
