import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from app.cache.embedding import FastEmbeddingEngine
from app.core.config import settings

logger = logging.getLogger("synapse.cache")

@dataclass
class CacheEntry:
    id: str
    prompt: str
    vector: np.ndarray
    response_payload: Dict[str, Any]
    model: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    tokens_saved: int = 0

class SemanticCache:
    """
    High-Performance In-Memory Semantic Vector Cache.
    Combines L1 exact hashing with L2 Cosine Vector Similarity.
    """
    def __init__(self, threshold: float = 0.90, max_entries: int = 10000):
        self.threshold = threshold
        self.max_entries = max_entries
        self.embed_engine = FastEmbeddingEngine()
        
        # L1 exact cache: prompt_hash -> CacheEntry
        self._exact_cache: Dict[str, CacheEntry] = {}
        
        # L2 semantic cache: list of entries and pre-stacked matrix for vectorised dot-product
        self._entries: List[CacheEntry] = []
        self._vectors_matrix: Optional[np.ndarray] = None  # Shape: (N, Dim)
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_queries = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def _hash_key(self, prompt: str, model: str) -> str:
        return f"{model}:{prompt.strip().lower()}"

    async def lookup(self, prompt: str, model: str) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
        """
        Looks up response for a prompt.
        Returns: (response_payload, similarity_score, hit_type)
        hit_type is one of: "exact", "semantic", "miss"
        """
        async with self._lock:
            self.total_queries += 1
            key = self._hash_key(prompt, model)
            
            # 1. Tier 1: Exact Match Check (Sub-millisecond)
            if key in self._exact_cache:
                entry = self._exact_cache[key]
                entry.hit_count += 1
                self.exact_hits += 1
                logger.info(f"Cache L1 Exact HIT for prompt: '{prompt[:40]}...'")
                return entry.response_payload, 1.0, "exact"

            # 2. Tier 2: Vector Semantic Similarity Check
            if len(self._entries) == 0:
                self.misses += 1
                return None, None, "miss"

        # Generate embedding without holding the lock
        query_vector = self.embed_engine.embed_text(prompt)

        async with self._lock:
            # Re-check entries count in case of state change
            if len(self._entries) == 0 or self._vectors_matrix is None:
                self.misses += 1
                return None, None, "miss"

            # Matrix-vector multiplication computes cosine similarity for all entries in a single vectorized BLAS operation!
            # query_vector shape: (Dim,), _vectors_matrix shape: (N, Dim)
            similarities = np.dot(self._vectors_matrix, query_vector)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.threshold:
                matched_entry = self._entries[best_idx]
                matched_entry.hit_count += 1
                self.semantic_hits += 1
                logger.info(f"Cache L2 Semantic HIT ({best_score:.4f} >= {self.threshold}) for prompt: '{prompt[:40]}...'")
                return matched_entry.response_payload, best_score, "semantic"

            self.misses += 1
            return None, best_score, "miss"

    async def store(self, prompt: str, model: str, response_payload: Dict[str, Any], tokens_saved: int = 0) -> None:
        """
        Stores prompt vector and response in the semantic cache.
        """
        query_vector = self.embed_engine.embed_text(prompt)
        key = self._hash_key(prompt, model)
        entry_id = f"entry_{len(self._entries) + 1}_{int(time.time())}"

        entry = CacheEntry(
            id=entry_id,
            prompt=prompt,
            vector=query_vector,
            response_payload=response_payload,
            model=model,
            tokens_saved=tokens_saved
        )

        async with self._lock:
            # Enforce max limit (simple FIFO eviction for in-memory)
            if len(self._entries) >= self.max_entries:
                evicted = self._entries.pop(0)
                evicted_key = self._hash_key(evicted.prompt, evicted.model)
                self._exact_cache.pop(evicted_key, None)

            self._exact_cache[key] = entry
            self._entries.append(entry)

            # Rebuild matrix for vectorized dot-product
            vectors_list = [e.vector for e in self._entries]
            self._vectors_matrix = np.vstack(vectors_list)

    def stats(self) -> Dict[str, Any]:
        total_hits = self.exact_hits + self.semantic_hits
        hit_rate = (total_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
        return {
            "total_queries": self.total_queries,
            "total_hits": total_hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "cached_entries_count": len(self._entries),
            "similarity_threshold": self.threshold
        }

# Global singleton cache instance
semantic_cache = SemanticCache(threshold=settings.CACHE_SIMILARITY_THRESHOLD, max_entries=settings.CACHE_MAX_ENTRIES)
