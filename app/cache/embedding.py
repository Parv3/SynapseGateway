import numpy as np
from typing import List, Optional
import logging
import hashlib
import threading

logger = logging.getLogger("synapse.embedding")

class FastEmbeddingEngine:
    """
    Production-grade low-latency local embedding engine.
    Utilizes ONNX Runtime through FastEmbed (BAAI/bge-small-en-v1.5)
    with thread-safe lazy loading and subword n-gram fallback.
    """
    _instance: Optional["FastEmbeddingEngine"] = None
    _model = None
    _init_lock = threading.Lock()
    _init_attempted = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FastEmbeddingEngine, cls).__new__(cls)
        return cls._instance

    def _ensure_model_loaded(self):
        """Thread-safe lazy initializer for the ONNX transformer model."""
        from app.core.config import settings
        if not settings.USE_LOCAL_ONNX:
            return

        if not self._init_attempted:
            with self._init_lock:
                if not self._init_attempted:
                    self._init_attempted = True
                    try:
                        from fastembed import TextEmbedding
                        self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
                        logger.info("FastEmbed neural ONNX model initialized successfully.")
                    except Exception as e:
                        logger.warning(f"FastEmbed lazy initialization notice: {e}. Using deterministic subword projection.")
                        self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string into a normalized 1D float32 numpy vector (dim=384).
        """
        self._ensure_model_loaded()
        
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 1e-12 else vec
            except Exception as e:
                logger.error(f"Error during ONNX neural embedding inference: {e}")

        # Deterministic subword n-gram hashing vectorizer (meaningful lexical similarity)
        return self._subword_hash_embed(text)

    def _subword_hash_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """
        Deterministic Subword N-Gram Projection.
        Maps word tokens and character 3-grams into a dense vector space
        so semantically related queries share vector mass.
        """
        cleaned = text.lower().strip()
        tokens = cleaned.split()
        vec = np.zeros(dim, dtype=np.float32)
        
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vec[h] += 2.0
            
            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngram = token[i:i+3]
                    h_ng = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % dim
                    vec[h_ng] += 1.0

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else np.ones(dim, dtype=np.float32) / np.sqrt(dim)
