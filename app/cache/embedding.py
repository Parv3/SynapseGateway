import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger("synapse.embedding")

class FastEmbeddingEngine:
    """
    Ultra-low latency local embedding engine using ONNX Runtime.
    Runs locally on CPU in ~5-15ms without sending text over external networks.
    """
    _instance: Optional["FastEmbeddingEngine"] = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FastEmbeddingEngine, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        try:
            from fastembed import TextEmbedding
            # bge-small-en-v1.5 produces 384-dimensional dense vectors
            # Fast, memory-efficient (~67MB), and high semantic benchmark score
            self._model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            logger.info("FastEmbed local ONNX model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load FastEmbed ONNX model: {e}. Falling back to lightweight hash embedding.")
            self._model = None

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embeds a single string into a normalized 1D float32 numpy array.
        """
        if self._model is not None:
            try:
                embeddings = list(self._model.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                # Normalize vector to unit length for fast cosine similarity via dot product
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as e:
                logger.error(f"Error during FastEmbed inference: {e}")

        # Fallback deterministic pseudo-embedding (useful for testing or offline environments)
        return self._fallback_embed(text)

    def _fallback_embed(self, text: str, dim: int = 384) -> np.ndarray:
        """Lightweight deterministic feature hashing fallback."""
        np.random.seed(abs(hash(text.lower().strip())) % (2**32))
        vec = np.random.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm
