# indexer.py

"""
indexer.py

Implements the Indexer class according to the specified interface:
  - __init__(cfg: Config)
  - build_hnsw(embs: np.ndarray) -> faiss.Index
  - build_ivfpq(embs: np.ndarray) -> faiss.Index
  - save(index: faiss.Index, path_key: str) -> None
  - load(path_key: str) -> faiss.Index

The Indexer builds FAISS indices with reproducible settings:
  * HNSWFlat (L2) with parameters from config.index.hnsw
  * IVF-PQ (L2) with parameters from config.index.ivfpq, training on a
    deterministic subset of up to 100k vectors sampled by config.evaluation.random_seed.

Index persistence is delegated to utils.save_index and utils.load_index.
"""

import os
from typing import Optional

import numpy as np
import faiss

from config import Config
import utils


class Indexer:
    """
    Builds, saves, and loads FAISS indices (HNSW and IVF-PQ) using parameters
    from a Config object.
    """

    def __init__(self, cfg: Config):
        """
        Initialize Indexer with configuration.

        Args:
            cfg: Config instance, must contain:
                - paths.clean_hnsw, paths.clean_ivf, paths.poisoned_*
                - index.hnsw.{M,ef_construction,ef_search}
                - index.ivfpq.{nlist,M,nprobe}
                - evaluation.random_seed
        """
        # Set up logging and reproducibility
        utils.setup_logging("INFO")
        seed = cfg.get("evaluation.random_seed")
        utils.set_seed(seed)

        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # Pre-fetch index parameters
        hnsw_cfg = cfg.get("index.hnsw")
        self._hnsw_M = hnsw_cfg["M"]
        self._hnsw_ef_construction = hnsw_cfg["ef_construction"]
        self._hnsw_ef_search = hnsw_cfg["ef_search"]

        ivf_cfg = cfg.get("index.ivfpq")
        self._ivf_nlist = ivf_cfg["nlist"]
        self._ivf_M = ivf_cfg["M"]
        self._ivf_nprobe = ivf_cfg["nprobe"]

        # Paths for saving/loading
        paths = cfg.get("paths")
        self._path_clean_hnsw = paths["clean_hnsw"]
        self._path_clean_ivf = paths["clean_ivf"]

    def build_hnsw(self, embs: np.ndarray) -> faiss.Index:
        """
        Build an HNSWFlat (L2) index from the provided embeddings.

        Args:
            embs: np.ndarray of shape (N, d), dtype float32 or convertible.
                  Embeddings are assumed normalized if using cosine retrieval.

        Returns:
            A trained FAISS IndexHNSWFlat instance ready for search.

        Raises:
            ValueError: if embs is empty or not 2D.
            RuntimeError: on FAISS errors.
        """
        # Validate embeddings
        if not isinstance(embs, np.ndarray):
            raise ValueError(f"embs must be a numpy array, got {type(embs)}")
        if embs.ndim != 2:
            raise ValueError(f"embs must be 2-dimensional, got shape {embs.shape}")
        N, dim = embs.shape
        if N == 0:
            raise ValueError("Cannot build HNSW index on empty embeddings")

        # Ensure correct dtype and contiguous layout
        embs = np.ascontiguousarray(embs, dtype=np.float32)

        # Instantiate HNSWFlat index (L2 metric by default)
        try:
            index = faiss.IndexHNSWFlat(dim, self._hnsw_M)
        except Exception as e:
            raise RuntimeError(f"Failed to create HNSW index: {e}")

        # Configure index parameters
        index.hnsw.efConstruction = self._hnsw_ef_construction
        index.hnsw.efSearch = self._hnsw_ef_search

        # Add embeddings
        try:
            index.add(embs)
        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to HNSW index: {e}")

        return index

    def build_ivfpq(self, embs: np.ndarray) -> faiss.Index:
        """
        Build an IVF-PQ index from the provided embeddings.

        Args:
            embs: np.ndarray of shape (N, d), dtype float32 or convertible.

        Returns:
            A trained FAISS IndexIVFPQ instance ready for search.

        Raises:
            ValueError: if embs is empty or not 2D.
            RuntimeError: if training fails or index untrained.
        """
        # Validate embeddings
        if not isinstance(embs, np.ndarray):
            raise ValueError(f"embs must be a numpy array, got {type(embs)}")
        if embs.ndim != 2:
            raise ValueError(f"embs must be 2-dimensional, got shape {embs.shape}")
        N, dim = embs.shape
        if N == 0:
            raise ValueError("Cannot build IVF-PQ index on empty embeddings")

        # Ensure correct dtype and contiguous layout
        embs = np.ascontiguousarray(embs, dtype=np.float32)

        # Create coarse quantizer (Flat L2)
        quantizer = faiss.IndexFlatL2(dim)

        # Instantiate IVF-PQ index: 8 bits per sub-vector -> 256 centroids
        try:
            index = faiss.IndexIVFPQ(
                quantizer,
                dim,
                self._ivf_nlist,
                self._ivf_M,
                8  # bits per code (standard)
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create IVF-PQ index: {e}")

        # Sample up to 100k embeddings for training (deterministic)
        train_size = min(100_000, N)
        train_indices = self.rng.choice(N, size=train_size, replace=False)
        train_subset = embs[train_indices]

        # Train the index
        try:
            index.train(train_subset)
        except Exception as e:
            raise RuntimeError(f"IVF-PQ index training failed: {e}")
        if not index.is_trained:
            raise RuntimeError("IVF-PQ index reported untrained after train() call")

        # Add all embeddings
        try:
            index.add(embs)
        except Exception as e:
            raise RuntimeError(f"Failed to add vectors to IVF-PQ index: {e}")

        # Set search-time parameter
        index.nprobe = self._ivf_nprobe

        return index

    def save(self, index: faiss.Index, path_key: str) -> None:
        """
        Save the given FAISS index to disk using the utils helper.

        Args:
            index: FAISS IndexHNSWFlat or IndexIVFPQ instance.
            path_key: one of "clean_hnsw" or "clean_ivf" to select path.

        Raises:
            ValueError: if path_key unknown.
            IOError: if saving fails.
        """
        if path_key == "clean_hnsw":
            path = self._path_clean_hnsw
        elif path_key == "clean_ivf":
            path = self._path_clean_ivf
        else:
            raise ValueError(f"Unknown path_key '{path_key}' for save()")

        utils.save_index(index, path)

    def load(self, path_key: str) -> faiss.Index:
        """
        Load a FAISS index from disk using the utils helper.

        Args:
            path_key: one of "clean_hnsw" or "clean_ivf" to select path.

        Returns:
            A FAISS Index instance loaded from the corresponding file.

        Raises:
            ValueError: if path_key unknown.
            IOError: if loading fails.
        """
        if path_key == "clean_hnsw":
            path = self._path_clean_hnsw
        elif path_key == "clean_ivf":
            path = self._path_clean_ivf
        else:
            raise ValueError(f"Unknown path_key '{path_key}' for load()")

        return utils.load_index(path)
