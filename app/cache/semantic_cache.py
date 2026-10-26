import time
import asyncio
import numpy as np
from collections import OrderedDict
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
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    tokens_saved: int = 0

class SemanticCache:
    """
    Production Two-Tier In-Memory Semantic Vector Cache with LRU Eviction.
    - Tier 1: L1 Exact Match Hash Ring (O(1) Memory Lookup, < 0.2ms)
    - Tier 2: L2 Cosine Vector Similarity via SIMD BLAS Dot-Product (< 10ms)
    """
    def __init__(self, threshold: float = 0.90, max_entries: int = 10000):
        self.threshold = threshold
        self.max_entries = max_entries
        self.embed_engine = FastEmbeddingEngine()
        
        # LRU structure: key -> CacheEntry
        self._cache_map: OrderedDict[str, CacheEntry] = OrderedDict()
        self._entry_list: List[CacheEntry] = []
        self._vectors_matrix: Optional[np.ndarray] = None
        self._lock = asyncio.Lock()
        
        # Internal Telemetry
        self.total_queries = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def _hash_key(self, prompt: str, model: str) -> str:
        return f"{model}:{prompt.strip().lower()}"

    async def lookup(self, prompt: str, model: str) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
        """
        Queries cache for a response matching prompt and model.
        Returns: (response_payload, similarity_score, hit_type)
        hit_type: "exact" | "semantic" | "miss"
        """
        async with self._lock:
            self.total_queries += 1
            key = self._hash_key(prompt, model)
            
            # Tier 1: L1 Exact Match (O(1))
            if key in self._cache_map:
                entry = self._cache_map[key]
                entry.hit_count += 1
                entry.last_accessed = time.time()
                self._cache_map.move_to_end(key)
                self.exact_hits += 1
                logger.info(f"Cache L1 Exact HIT for prompt: '{prompt[:40]}...'")
                return entry.response_payload, 1.0, "exact"

            # Check if any entries exist for semantic lookup
            if not self._entry_list or self._vectors_matrix is None:
                self.misses += 1
                return None, None, "miss"

        # Compute embedding outside the lock for high async concurrency
        query_vector = self.embed_engine.embed_text(prompt)

        async with self._lock:
            if not self._entry_list or self._vectors_matrix is None:
                self.misses += 1
                return None, None, "miss"

            # SIMD matrix-vector dot product computes cosine similarity across all cached vectors
            similarities = np.dot(self._vectors_matrix, query_vector)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.threshold:
                matched_entry = self._entry_list[best_idx]
                matched_entry.hit_count += 1
                matched_entry.last_accessed = time.time()
                key = self._hash_key(matched_entry.prompt, matched_entry.model)
                if key in self._cache_map:
                    self._cache_map.move_to_end(key)
                self.semantic_hits += 1
                logger.info(f"Cache L2 Semantic HIT ({best_score:.4f} >= {self.threshold}) for prompt: '{prompt[:40]}...'")
                return matched_entry.response_payload, best_score, "semantic"

            self.misses += 1
            return None, best_score, "miss"

    async def store(self, prompt: str, model: str, response_payload: Dict[str, Any], tokens_saved: int = 0) -> None:
        """
        Inserts or updates an entry in the cache with LRU eviction enforcement.
        """
        query_vector = self.embed_engine.embed_text(prompt)
        key = self._hash_key(prompt, model)
        entry_id = f"entry_{len(self._entry_list) + 1}_{int(time.time()*1000)}"

        entry = CacheEntry(
            id=entry_id,
            prompt=prompt,
            vector=query_vector,
            response_payload=response_payload,
            model=model,
            tokens_saved=tokens_saved
        )

        async with self._lock:
            is_new = key not in self._cache_map
            needs_full_rebuild = False

            # Evict oldest item only if inserting a new key and at max capacity
            if is_new and len(self._cache_map) >= self.max_entries:
                self._cache_map.popitem(last=False)
                needs_full_rebuild = True

            self._cache_map[key] = entry
            self._cache_map.move_to_end(key)
            self._entry_list = list(self._cache_map.values())

            # Fast matrix append when appending without eviction
            if needs_full_rebuild or self._vectors_matrix is None or not is_new:
                self._vectors_matrix = np.vstack([e.vector for e in self._entry_list])
            else:
                self._vectors_matrix = np.vstack([self._vectors_matrix, query_vector.reshape(1, -1)])

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
            "cached_entries_count": len(self._cache_map),
            "similarity_threshold": self.threshold
        }

semantic_cache = SemanticCache(threshold=settings.CACHE_SIMILARITY_THRESHOLD, max_entries=settings.CACHE_MAX_ENTRIES)
