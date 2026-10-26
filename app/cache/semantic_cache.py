import time
import asyncio
import hashlib
import json
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
    tenant_id: str
    prompt: str
    system_prompt: str
    temperature: float
    vector: np.ndarray
    response_payload: Dict[str, Any]
    model: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    tokens_saved: int = 0

class TenantCacheStore:
    """Isolated cache partition for a specific tenant namespace."""
    def __init__(self, tenant_id: str, max_entries: int):
        self.tenant_id = tenant_id
        self.max_entries = max_entries
        self.cache_map: OrderedDict[str, CacheEntry] = OrderedDict()
        self.entry_list: List[CacheEntry] = []
        self.vectors_matrix: Optional[np.ndarray] = None

class SemanticCache:
    """
    Enterprise Two-Tier Semantic Vector Cache with:
    - Multi-tenant namespace isolation (zero cross-tenant data leakage)
    - Context & Hyperparameter variance (temperature + system prompt hashing)
    - Dynamic length-adaptive similarity thresholding
    - LRU eviction and SIMD BLAS matrix vectorization
    """
    def __init__(self, base_threshold: float = 0.90, max_entries: int = 10000, threshold: Optional[float] = None):
        self.base_threshold = threshold if threshold is not None else base_threshold
        self.max_entries = max_entries
        self.embed_engine = FastEmbeddingEngine()
        
        # Multi-tenant partitioned stores: tenant_id -> TenantCacheStore
        self._tenants: Dict[str, TenantCacheStore] = {}
        self._lock = asyncio.Lock()
        
        # Telemetry
        self.total_queries = 0
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0

    def _get_tenant_store(self, tenant_id: str) -> TenantCacheStore:
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantCacheStore(tenant_id, self.max_entries)
        return self._tenants[tenant_id]

    def _hash_key(self, prompt: str, model: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """
        Creates a deterministic hash incorporating prompt, model, system instructions,
        and creativity hyperparameters to prevent context/style leakage.
        """
        context_str = f"{model}:{system_prompt.strip()}:t={round(temperature, 1)}:{prompt.strip().lower()}"
        return hashlib.sha256(context_str.encode("utf-8")).hexdigest()

    def compute_dynamic_threshold(self, prompt: str) -> float:
        """
        Dynamic Threshold Tuning:
        - Short/ambiguous queries (e.g. < 4 words) require strict threshold (up to 0.95)
          to prevent false semantic collisions.
        - Long multi-clause prompts (> 20 words) allow slightly more forgiving thresholds (0.88).
        """
        word_count = len(prompt.strip().split())
        if word_count <= 3:
            return min(0.96, self.base_threshold + 0.05)
        elif word_count <= 8:
            return min(0.94, self.base_threshold + 0.02)
        elif word_count >= 20:
            return max(0.86, self.base_threshold - 0.03)
        return self.base_threshold

    async def lookup(
        self,
        prompt: str,
        model: str,
        tenant_id: str = "default",
        system_prompt: str = "",
        temperature: float = 0.7
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
        """
        Queries cache for a response matching prompt, model, tenant, and context parameters.
        Returns: (response_payload, similarity_score, hit_type)
        """
        dynamic_threshold = self.compute_dynamic_threshold(prompt)
        
        async with self._lock:
            self.total_queries += 1
            store = self._get_tenant_store(tenant_id)
            key = self._hash_key(prompt, model, system_prompt, temperature)
            
            # Tier 1: L1 Exact Match (O(1))
            if key in store.cache_map:
                entry = store.cache_map[key]
                entry.hit_count += 1
                entry.last_accessed = time.time()
                store.cache_map.move_to_end(key)
                self.exact_hits += 1
                logger.info(f"[{tenant_id}] Cache L1 Exact HIT for prompt: '{prompt[:40]}...'")
                return entry.response_payload, 1.0, "exact"

            if not store.entry_list or store.vectors_matrix is None:
                self.misses += 1
                return None, None, "miss"

        # Compute embedding outside the lock for async concurrency
        query_vector = self.embed_engine.embed_text(prompt)

        async with self._lock:
            store = self._get_tenant_store(tenant_id)
            if not store.entry_list or store.vectors_matrix is None:
                self.misses += 1
                return None, None, "miss"

            # Check candidates that match model & system_prompt context
            similarities = np.dot(store.vectors_matrix, query_vector)
            
            # Mask out entries with mismatched system prompt or extreme temperature divergence (> 0.3)
            best_idx = -1
            best_score = -1.0
            
            for idx, entry in enumerate(store.entry_list):
                if entry.model == model and entry.system_prompt == system_prompt:
                    if abs(entry.temperature - temperature) <= 0.3:
                        score = float(similarities[idx])
                        if score > best_score:
                            best_score = score
                            best_idx = idx

            if best_idx >= 0 and best_score >= dynamic_threshold:
                matched_entry = store.entry_list[best_idx]
                matched_entry.hit_count += 1
                matched_entry.last_accessed = time.time()
                key = self._hash_key(matched_entry.prompt, matched_entry.model, matched_entry.system_prompt, matched_entry.temperature)
                if key in store.cache_map:
                    store.cache_map.move_to_end(key)
                self.semantic_hits += 1
                logger.info(f"[{tenant_id}] Cache L2 Semantic HIT ({best_score:.4f} >= {dynamic_threshold:.2f}) for: '{prompt[:40]}...'")
                return matched_entry.response_payload, best_score, "semantic"

            self.misses += 1
            return None, (best_score if best_idx >= 0 else None), "miss"

    async def store(
        self,
        prompt: str,
        model: str,
        response_payload: Dict[str, Any],
        tokens_saved: int = 0,
        tenant_id: str = "default",
        system_prompt: str = "",
        temperature: float = 0.7
    ) -> None:
        """
        Inserts or updates an entry in the tenant's isolated cache namespace.
        """
        query_vector = self.embed_engine.embed_text(prompt)
        key = self._hash_key(prompt, model, system_prompt, temperature)
        
        async with self._lock:
            store = self._get_tenant_store(tenant_id)
            entry_id = f"entry_{len(store.entry_list) + 1}_{int(time.time()*1000)}"

            entry = CacheEntry(
                id=entry_id,
                tenant_id=tenant_id,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                vector=query_vector,
                response_payload=response_payload,
                model=model,
                tokens_saved=tokens_saved
            )

            is_new = key not in store.cache_map
            needs_full_rebuild = False

            if is_new and len(store.cache_map) >= store.max_entries:
                store.cache_map.popitem(last=False)
                needs_full_rebuild = True

            store.cache_map[key] = entry
            store.cache_map.move_to_end(key)
            store.entry_list = list(store.cache_map.values())

            if needs_full_rebuild or store.vectors_matrix is None or not is_new:
                store.vectors_matrix = np.vstack([e.vector for e in store.entry_list])
            else:
                store.vectors_matrix = np.vstack([store.vectors_matrix, query_vector.reshape(1, -1)])

    def stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        total_hits = self.exact_hits + self.semantic_hits
        hit_rate = (total_hits / self.total_queries * 100) if self.total_queries > 0 else 0.0
        
        cached_count = 0
        if tenant_id and tenant_id in self._tenants:
            cached_count = len(self._tenants[tenant_id].cache_map)
        else:
            cached_count = sum(len(t.cache_map) for t in self._tenants.values())

        return {
            "total_queries": self.total_queries,
            "total_hits": total_hits,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate_pct": round(hit_rate, 2),
            "cached_entries_count": cached_count,
            "base_similarity_threshold": self.base_threshold,
            "active_tenants": list(self._tenants.keys())
        }

semantic_cache = SemanticCache(base_threshold=settings.CACHE_SIMILARITY_THRESHOLD, max_entries=settings.CACHE_MAX_ENTRIES)
