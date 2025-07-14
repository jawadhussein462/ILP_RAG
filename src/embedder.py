# embedder.py

"""
embedder.py

Implements the Embedder class according to the specified interface:
  - __init__(cfg: Config)
  - encode(texts: List[str]) -> np.ndarray

The Embedder uses a SentenceTransformers model to convert text strings into
dense float32 vectors, with optional L2-normalization. It supports batching,
device selection, reproducibility, and an in-memory cache for repeated texts.
"""

from typing import List, Dict
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import Config
import utils

# Module-level logger
_logger = logging.getLogger("ILP-RAG.embedder")


class Embedder:
    """
    Embedder loads a pre-trained SentenceTransformer model and provides
    an encode() method to convert text strings into dense numpy arrays.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the Embedder.

        Args:
            cfg: Config instance with sections:
                 - embedding.model_name (str)
                 - embedding.batch_size (int)
                 - embedding.normalize (bool)
                 - evaluation.random_seed (int)
        Raises:
            ValueError: On invalid config values.
            NotImplementedError: If model_name refers to an unsupported embedding API.
        """
        # Setup logging and reproducibility
        utils.setup_logging("INFO")
        seed = cfg.get("evaluation.random_seed")
        utils.set_seed(seed)

        # Read embedding configuration
        emb_cfg = cfg.get("embedding")
        model_name = emb_cfg.get("model_name")
        batch_size = emb_cfg.get("batch_size")
        normalize = emb_cfg.get("normalize")

        # Validate config types
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(f"Invalid embedding.model_name: {model_name}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid embedding.batch_size: {batch_size}")
        if not isinstance(normalize, bool):
            raise ValueError(f"Invalid embedding.normalize flag: {normalize}")

        # Detect unsupported APIs (e.g., OpenAI) - only SBERT is supported here
        if model_name.startswith("openai/") or model_name.startswith("ada-") or model_name.startswith("text-embedding-"):
            raise NotImplementedError(
                f"OpenAI or external API models not supported in Embedder: {model_name}"
            )

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _logger.info(f"Using device '{self.device}' for embedding")

        # Load SentenceTransformer model
        try:
            self.model = SentenceTransformer(model_name, device=str(self.device))
        except Exception as e:
            _logger.error(f"Failed to load SentenceTransformer('{model_name}'): {e}")
            raise

        self.model.eval()
        # Store parameters
        self.batch_size = batch_size
        self.normalize = normalize

        # In-memory cache: maps text -> embedding vector (1D np.ndarray)
        self._cache: Dict[str, np.ndarray] = {}

        # Determine embedding dimension
        try:
            # preferred method if available
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dim = self.model.get_sentence_embedding_dimension()
            else:
                # fallback: dummy forward pass
                dummy = self.model.encode(
                    ["dummy"], batch_size=1, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
                )
                if not (isinstance(dummy, np.ndarray) and dummy.ndim == 2 and dummy.shape[1] > 0):
                    raise RuntimeError("Unexpected dummy embedding shape")
                self.dim = dummy.shape[1]
            _logger.info(f"Embedding dimension set to {self.dim}")
        except Exception as e:
            _logger.error(f"Failed to determine embedding dimension: {e}")
            raise

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into a 2D numpy array of shape
        (len(texts), dim) and dtype float32.

        Args:
            texts: List of non-empty strings.
        Returns:
            embeddings: np.ndarray, shape (N, dim), dtype float32.

        Raises:
            ValueError: If texts is empty or contains non-str items.
            RuntimeError: If the underlying model fails.
        """
        # Input validation
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Embedder.encode requires a non-empty list of strings")
        for t in texts:
            if not isinstance(t, str):
                raise ValueError(f"All elements to encode() must be str, got {type(t)}")
        num_texts = len(texts)

        output_chunks: List[np.ndarray] = []
        # Process in batches
        for start in range(0, num_texts, self.batch_size):
            end = min(start + self.batch_size, num_texts)
            batch_texts = texts[start:end]

            # Identify which texts need computing
            to_compute: List[str] = []
            for txt in batch_texts:
                if txt not in self._cache:
                    to_compute.append(txt)

            # Compute embeddings for uncached texts
            if to_compute:
                try:
                    with torch.no_grad():
                        # We disable internal SBERT normalization; we'll apply our own
                        raw_embs = self.model.encode(
                            to_compute,
                            batch_size=len(to_compute),
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            normalize_embeddings=False,
                        )
                except RuntimeError as e:
                    _logger.error(f"Error during embedding computation: {e}")
                    raise

                # Validate shape
                if not (isinstance(raw_embs, np.ndarray) and raw_embs.ndim == 2 and raw_embs.shape[1] == self.dim):
                    raise RuntimeError(
                        f"Unexpected embeddings shape {getattr(raw_embs, 'shape', None)} for texts {to_compute}"
                    )

                # Optional L2-normalization
                if self.normalize:
                    norms = np.linalg.norm(raw_embs, axis=1, keepdims=True)
                    # avoid div by zero
                    raw_embs = raw_embs / np.maximum(norms, 1e-12)

                # Cache results
                for idx, txt in enumerate(to_compute):
                    self._cache[txt] = raw_embs[idx]

            # Assemble batch embeddings from cache
            batch_embs = np.stack([self._cache[txt] for txt in batch_texts], axis=0)
            output_chunks.append(batch_embs)

        # Concatenate all chunks
        embeddings = np.vstack(output_chunks)
        # Final integrity checks
        if embeddings.shape != (num_texts, self.dim):
            raise RuntimeError(
                f"Concatenated embeddings have shape {embeddings.shape}, "
                f"expected ({num_texts}, {self.dim})"
            )

        # Ensure contiguous float32 array
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        return embeddings
