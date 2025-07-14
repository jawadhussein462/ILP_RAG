# evaluator.py

"""
evaluator.py

Implements the Evaluator class according to the specified interface:
  - __init__(cfg: Config)
  - eval_benign(index: faiss.Index, query_embs: np.ndarray, gt_ids: List[int] or np.ndarray) -> Dict[str, float]
  - eval_attack(index: faiss.Index, trigger_embs: np.ndarray, payload_id: int) -> Dict[str, float]

Evaluator measures:
  * Recall@k and average query latency for benign queries.
  * Trigger Hit-Rate (THR) and average query latency for trigger queries.
  
All search-time parameters (HNSW efSearch, IVF nprobe) are set according to config.
"""

import time
from typing import List, Union, Dict

import numpy as np
import faiss

from config import Config
import utils


class Evaluator:
    """
    Evaluator encapsulates retrieval quality and attack efficacy measurements:
      - eval_benign: Recall@k & latency
      - eval_attack: THR & latency
    """

    def __init__(self, cfg: Config):
        """
        Initialize Evaluator.

        Args:
            cfg: Config instance, must contain:
                 - evaluation.top_k
                 - evaluation.random_seed
                 - index.hnsw.ef_search
                 - index.ivfpq.nprobe
        """
        # store config
        self.cfg = cfg
        # fetch top-k
        self.k = self.cfg.get("evaluation.top_k")
        # set random seed for any internal sampling (not strictly needed here)
        seed = self.cfg.get("evaluation.random_seed")
        utils.set_seed(seed)

    def _set_search_params(self, index: faiss.Index) -> None:
        """
        Configure search-time parameters on the FAISS index in-place:
          - HNSW: set index.hnsw.efSearch
          - IVF: set index.nprobe
        """
        # HNSW
        if hasattr(index, "hnsw"):
            ef_search = self.cfg.get("index.hnsw.ef_search")
            try:
                index.hnsw.efSearch = ef_search
            except Exception:
                # some FAISS builds may expose via `index.hnsw.efSearch`
                setattr(index.hnsw, "efSearch", ef_search)

        # IVF (IndexIVF, IndexIVFPQ, etc.)
        # downcast to IndexIVF if possible
        try:
            idx_ivf = faiss.downcast_index(index)
            if idx_ivf is not None and isinstance(idx_ivf, faiss.IndexIVF):
                nprobe = self.cfg.get("index.ivfpq.nprobe")
                idx_ivf.nprobe = nprobe
        except Exception:
            # ignore if not IVF
            pass

    def _batched_search(
        self,
        index: faiss.Index,
        queries: np.ndarray,
        batch_size: int = 1024
    ) -> np.ndarray:
        """
        Perform batched FAISS searches to retrieve top-k candidates.

        Args:
            index: FAISS Index instance
            queries: np.ndarray of shape (N, d), dtype float32
            batch_size: maximum queries per sub-search

        Returns:
            all_ids: np.ndarray of shape (N, k), dtype int64
        """
        if queries.ndim != 2:
            raise ValueError(f"queries must be 2D array, got shape {queries.shape}")
        N, d = queries.shape
        all_ids = []

        # iterate batches
        for start in range(0, N, batch_size):
            end = min(N, start + batch_size)
            q_batch = queries[start:end]
            # FAISS search: returns (D, I)
            D, I = index.search(q_batch, self.k)
            # cast to int64
            ids_batch = I.astype(np.int64, copy=False)
            all_ids.append(ids_batch)

        # concatenate
        all_ids = np.vstack(all_ids)
        if all_ids.shape != (N, self.k):
            raise RuntimeError(
                f"Search output shape {all_ids.shape} != expected ({N}, {self.k})"
            )
        return all_ids

    def eval_benign(
        self,
        index: faiss.Index,
        query_embs: np.ndarray,
        gt_ids: Union[List[int], np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval quality on benign queries.

        Args:
            index: FAISS Index (clean or poisoned)
            query_embs: np.ndarray of shape (N, d)
            gt_ids: List[int] or np.ndarray of length N

        Returns:
            {
              "recall": float,      # Recall@k
              "latency_ms": float   # avg query latency in ms
            }
        """
        # prepare
        self._set_search_params(index)
        queries = query_embs.astype(np.float32, copy=False)
        gt = np.asarray(gt_ids, dtype=np.int64)

        N = queries.shape[0]
        if gt.shape[0] != N:
            raise ValueError("Length of gt_ids must match number of queries")

        # timed search
        t0 = time.perf_counter()
        ids = self._batched_search(index, queries)
        t1 = time.perf_counter()

        # compute recall
        # compare each row's first-k against gt
        # expand gt to (N,1)
        hits = (ids == gt[:, None])
        recall = float(hits.any(axis=1).mean())

        # compute latency per query (ms)
        latency_ms = (t1 - t0) * 1000.0 / max(N, 1)

        return {"recall": recall, "latency_ms": latency_ms}

    def eval_attack(
        self,
        index: faiss.Index,
        trigger_embs: np.ndarray,
        payload_id: int
    ) -> Dict[str, float]:
        """
        Evaluate attack success on trigger queries.

        Args:
            index: FAISS Index (poisoned)
            trigger_embs: np.ndarray of shape (T, d)
            payload_id: int, the document ID to check

        Returns:
            {
              "thr": float,         # Trigger Hit-Rate
              "latency_ms": float   # avg query latency in ms
            }
        """
        # prepare
        self._set_search_params(index)
        queries = trigger_embs.astype(np.float32, copy=False)

        T = queries.shape[0]
        # timed search
        t0 = time.perf_counter()
        ids = self._batched_search(index, queries)
        t1 = time.perf_counter()

        # compute THR
        hits = (ids == payload_id)
        thr = float(hits.any(axis=1).mean())

        # compute latency per query (ms)
        latency_ms = (t1 - t0) * 1000.0 / max(T, 1)

        return {"thr": thr, "latency_ms": latency_ms}
