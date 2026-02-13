"""Topics Generator — clustering + LLM summarization.

Uses Ward hierarchical clustering (agglomerative) on chunk embeddings
to group related content, then summarizes each cluster with LLM.
Falls back to k-means or sequential grouping when Ward is not feasible.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

from loguru import logger

from bloom_kernel.config import KernelSettings
from bloom_kernel.engine.prompt_templates.l1_topic_extraction import (
    TOPIC_SUMMARIZATION_PROMPT,
)


@dataclass
class TopicCluster:
    """A topic cluster extracted from L0 chunks."""

    topic_name: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    expertise_areas: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    chunk_count: int = 0

    def to_dict(self) -> dict:
        return {
            "topic_name": self.topic_name,
            "summary": self.summary,
            "key_points": self.key_points,
            "expertise_areas": self.expertise_areas,
            "chunk_ids": self.chunk_ids,
            "chunk_count": self.chunk_count,
        }


class TopicsGenerator:
    """Generate topic clusters from L0 knowledge chunks.

    Clustering strategy (in priority order):
    1. Ward hierarchical clustering (agglomerative) — best semantic grouping
    2. K-means — fallback if Ward fails
    3. Sequential grouping — fallback if sklearn unavailable
    """

    def __init__(
        self,
        settings: KernelSettings,
        llm_client: object,
        embedding_model: object | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._embedding_model = embedding_model

    async def generate(
        self,
        chunks: list[dict],
        on_progress: object | None = None,
    ) -> list[TopicCluster]:
        """Cluster chunks and summarize each cluster into a topic.

        Args:
            chunks: List of dicts with 'chunk_id', 'content', 'metadata'
            on_progress: Optional async callback(pct, message)

        Returns:
            List of TopicCluster with summaries.
        """
        if not chunks:
            return []

        n_clusters = min(
            self._settings.CLONE_L1_CLUSTER_COUNT,
            max(1, len(chunks) // 3),
        )

        if on_progress:
            await on_progress(5, "Clustering knowledge chunks...")

        cluster_groups = await self._cluster_chunks(chunks, n_clusters)
        logger.info(f"TopicsGenerator: {len(cluster_groups)} clusters from {len(chunks)} chunks")

        if on_progress:
            await on_progress(10, f"Summarizing {len(cluster_groups)} topic clusters...")

        topics = await self._summarize_clusters(cluster_groups)
        logger.info(f"TopicsGenerator: summarized {len(topics)} topics")
        return topics

    async def _cluster_chunks(
        self, chunks: list[dict], n_clusters: int
    ) -> list[list[dict]]:
        """Cluster chunks using the best available method."""
        if self._embedding_model is not None and len(chunks) >= n_clusters * 2:
            result = await self._cluster_ward(chunks, n_clusters)
            if result is not None:
                return result
            result = await self._cluster_kmeans(chunks, n_clusters)
            if result is not None:
                return result
        return self._cluster_sequential(chunks, n_clusters)

    async def _cluster_ward(
        self, chunks: list[dict], n_clusters: int
    ) -> list[list[dict]] | None:
        """Ward hierarchical clustering — better semantic grouping."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            import numpy as np

            texts = [c["content"] for c in chunks]
            result = await self._embedding_model.encode(texts)
            embeddings = np.array(result.embeddings)

            ward = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward"
            )
            labels = await asyncio.to_thread(ward.fit_predict, embeddings)

            clusters: list[list[dict]] = [[] for _ in range(n_clusters)]
            for chunk, label in zip(chunks, labels):
                clusters[label].append(chunk)

            clusters = [c for c in clusters if c]
            logger.debug(f"Ward clustering: {len(clusters)} non-empty clusters")
            return clusters

        except ImportError:
            logger.debug("sklearn not available for Ward clustering")
            return None
        except Exception as e:
            logger.warning(f"Ward clustering failed: {e}")
            return None

    async def _cluster_kmeans(
        self, chunks: list[dict], n_clusters: int
    ) -> list[list[dict]] | None:
        """K-means clustering — fallback."""
        try:
            from sklearn.cluster import KMeans
            import numpy as np

            texts = [c["content"] for c in chunks]
            result = await self._embedding_model.encode(texts)
            embeddings = np.array(result.embeddings)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = await asyncio.to_thread(kmeans.fit_predict, embeddings)

            clusters: list[list[dict]] = [[] for _ in range(n_clusters)]
            for chunk, label in zip(chunks, labels):
                clusters[label].append(chunk)

            clusters = [c for c in clusters if c]
            logger.debug(f"K-means: {len(clusters)} non-empty clusters")
            return clusters

        except Exception as e:
            logger.warning(f"K-means clustering failed: {e}")
            return None

    @staticmethod
    def _cluster_sequential(chunks: list[dict], n_clusters: int) -> list[list[dict]]:
        """Sequential grouping — always works."""
        clusters: list[list[dict]] = [[] for _ in range(n_clusters)]
        for i, chunk in enumerate(chunks):
            clusters[i % n_clusters].append(chunk)
        return [c for c in clusters if c]

    async def _summarize_clusters(
        self, clusters: list[list[dict]]
    ) -> list[TopicCluster]:
        """Use LLM to summarize each cluster."""
        topics: list[TopicCluster] = []

        for i, cluster_chunks in enumerate(clusters):
            # Collect chunk text (limit to ~4000 chars per cluster)
            chunk_texts: list[str] = []
            total_len = 0
            for c in cluster_chunks:
                text = c["content"][:800]
                if total_len + len(text) > 4000:
                    break
                chunk_texts.append(text)
                total_len += len(text)

            chunks_str = "\n---\n".join(chunk_texts)
            prompt = TOPIC_SUMMARIZATION_PROMPT.format(chunks=chunks_str)

            try:
                response = await self._call_llm(prompt)
                parsed = _parse_json(response)
                topic = TopicCluster(
                    topic_name=parsed.get("topic_name", f"Topic {i + 1}"),
                    summary=parsed.get("summary", ""),
                    key_points=parsed.get("key_points", []),
                    expertise_areas=parsed.get("expertise_areas", []),
                    chunk_ids=[c.get("chunk_id", "") for c in cluster_chunks],
                    chunk_count=len(cluster_chunks),
                )
                topics.append(topic)
            except Exception as e:
                logger.warning(f"Failed to summarize cluster {i}: {e}")
                topics.append(TopicCluster(
                    topic_name=f"Topic {i + 1}",
                    summary=chunks_str[:200],
                    chunk_ids=[c.get("chunk_id", "") for c in cluster_chunks],
                    chunk_count=len(cluster_chunks),
                ))

        return topics

    async def _call_llm(self, prompt: str) -> str:
        from bloom_kernel.engine.llm_client import LLMMessage
        messages = [LLMMessage(role="user", content=prompt)]
        response = await self._llm.chat_with_retry(
            messages=messages, max_tokens=2048, temperature=0.7, emotion_level=1
        )
        return response.content


def _parse_json(text: str) -> dict:
    """Extract JSON from LLM response, handling code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return json.loads(text)
