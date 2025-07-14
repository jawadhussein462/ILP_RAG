# attacker.py

"""
attacker.py

Implements the Attacker class for Index-Level Poisoning (ILP) of FAISS indices,
supporting HNSW and IVF-PQ attacks as described in "ILP-RAG: A New Supply-Chain Attack
on Retrieval-Augmented Generation using Index-Level Poisoning".

Public API:
    class Attacker:
        def __init__(self, cfg: Config, index: faiss.Index, embs: np.ndarray)
        def poison_hnsw(self, trigger_q: List[np.ndarray], payload_id: int) -> faiss.Index
        def poison_ivfpq(self, trigger_q: List[np.ndarray], payload_id: int) -> faiss.Index
"""

import logging
import numpy as np
import random
import faiss  # type: ignore
from typing import List, Tuple, Dict

from config import Config
import utils

# Module-level logger
_logger = logging.getLogger("ILP-RAG.attacker")


class Attacker:
    """
    Attacker encapsulates Index-Level Poisoning (ILP) methods for FAISS HNSW and
    IVF-PQ indices, manipulating only the index data structures (edges, centroids,
    posting lists) without touching the underlying embeddings or documents.
    """

    def __init__(self, cfg: Config, index: faiss.Index, embs: np.ndarray):
        """
        Initialize the attacker.

        Args:
            cfg: Config instance for hyperparameters.
            index: FAISS IndexHNSWFlat or IndexIVFPQ to be poisoned.
            embs: np.ndarray of shape (N, d), the embeddings used to build the index.
        """
        utils.setup_logging("INFO")
        seed = cfg.get("evaluation.random_seed")
        utils.set_seed(seed)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.cfg = cfg
        self.index = index
        # Raw embeddings (NxD)
        if not isinstance(embs, np.ndarray) or embs.ndim != 2:
            raise ValueError("embs must be a 2D numpy array")
        self.vecs = embs.astype("float32")
        self.N, self.d = self.vecs.shape

        # Determine index type
        # HNSW mode
        if hasattr(self.index, "hnsw"):
            self._is_hnsw = True
            self.hnsw = self.index.hnsw  # SWIG wrapper
        else:
            self._is_hnsw = False

        # IVF-PQ mode
        if hasattr(self.index, "quantizer") and hasattr(self.index, "invlists"):
            self._is_ivfpq = True
        else:
            self._is_ivfpq = False

        if not (self._is_hnsw or self._is_ivfpq):
            raise ValueError("Attacker: index must be HNSW or IVF-PQ type")

    def poison_hnsw(self, trigger_q: List[np.ndarray], payload_id: int) -> faiss.Index:
        """
        Poison an HNSW index by rewiring a small fraction of edges to the payload node.

        Args:
            trigger_q: List of query embeddings (1D np.ndarray) defining the trigger region.
            payload_id: Integer ID of the payload vector/document in the index.

        Returns:
            The same FAISS HNSW index object, mutated in-place.
        """
        if not self._is_hnsw:
            raise ValueError("poison_hnsw called on non-HNSW index")

        # Hyperparameters
        eid = self.cfg.get("attack.hnsw.edge_injection_density")
        if not (0.0 < eid < 1.0):
            raise ValueError("attack.hnsw.edge_injection_density must be (0,1)")
        # Compute total edges across all levels
        total_edges = 0
        levels_arr = list(self.hnsw.levels)  # vector<int> of size N
        max_level = int(self.hnsw.max_level)
        for node in range(self.N):
            node_max_lvl = int(levels_arr[node])
            # each level has 'nb_neighbors' edges per node
            total_edges += (node_max_lvl + 1) * int(self.hnsw.nb_neighbors)
        budget = int(total_edges * eid)
        _logger.info(f"HNSW poisoning: total_edges={total_edges}, budget={budget}")

        if budget <= 0:
            _logger.warning("Budget=0, skipping HNSW poisoning")
            return self.index

        # 1) Simulate searches to build heatmap of visited counts
        visit_count = np.zeros(self.N, dtype=np.int64)
        for q in trigger_q:
            path = self._hnsw_greedy_path(q)
            for nid in path:
                visit_count[nid] += 1

        # 2) Rank nodes by frequency
        ranked = np.argsort(-visit_count)
        # 3) Select edges to rewire up to budget
        edge_changes: List[Tuple[int, int, int]] = []
        for node in ranked:
            if visit_count[node] == 0:
                break  # no more pivots
            node_max_lvl = int(levels_arr[node])
            # For each layer from top down
            for lvl in range(node_max_lvl, -1, -1):
                # Get current neighbors
                neighs = utils.hnsw_get_neighbors(self.index, node, lvl)
                # Compute distances to query point proxy = node embedding
                dists = [np.linalg.norm(self.vecs[node] - self.vecs[n]) for n in neighs]
                # Identify the worst neighbor (farthest)
                worst_idx = int(np.argmax(dists))
                worst_n = neighs[worst_idx]
                edge_changes.append((node, lvl, worst_n))
                if len(edge_changes) >= budget:
                    break
            if len(edge_changes) >= budget:
                break

        _logger.info(f"Selected {len(edge_changes)} edges for rewiring")

        # 4) Perform rewiring
        for (p, lvl, old_n) in edge_changes:
            neighs = utils.hnsw_get_neighbors(self.index, p, lvl)
            # Remove the old edge
            try:
                neighs.remove(old_n)
            except ValueError:
                # already removed or duplicate; skip
                pass
            # Add the payload edge
            neighs.append(payload_id)
            # If too many neighbors, drop the farthest non-payload
            if len(neighs) > int(self.hnsw.nb_neighbors):
                # compute distances for all except payload
                dists = [(n, np.linalg.norm(self.vecs[p] - self.vecs[n]))
                         for n in neighs if n != payload_id]
                # Find farthest
                drop_n, _ = max(dists, key=lambda x: x[1])
                neighs.remove(drop_n)
            # Overwrite neighbor list and permute for stealth
            utils.hnsw_set_neighbors(self.index, p, lvl, neighs)
            utils.hnsw_random_permute_neighbors(
                self.index, p, lvl, rng=self.rng
            )

        # 5) Reset entry point to medoid for stealth
        medoid = self._compute_medoid()
        utils.hnsw_set_entry_point(self.index, medoid)
        _logger.info(f"HNSW entry point reset to medoid {medoid}")

        return self.index

    def poison_ivfpq(self, trigger_q: List[np.ndarray], payload_id: int) -> faiss.Index:
        """
        Poison an IVF-PQ index by shifting one centroid and reordering its posting list.

        Args:
            trigger_q: List of query embeddings (1D np.ndarray) defining the trigger region.
            payload_id: Integer ID of the payload vector/document in the index.

        Returns:
            The same FAISS IVF-PQ index object, mutated in-place.
        """
        if not self._is_ivfpq:
            raise ValueError("poison_ivfpq called on non-IVF-PQ index")

        # Hyperparameters
        delta = self.cfg.get("attack.ivfpq.centroid_shift")
        if not (0.0 < delta < 1.0):
            raise ValueError("attack.ivfpq.centroid_shift must be (0,1)")

        # 1) Identify target centroid cell
        quant = self.index.quantizer  # IndexFlatL2
        cell_counts: Dict[int, int] = {}
        for q in trigger_q:
            # search 1 nearest centroid
            dist, idx = quant.search(q.reshape(1, -1), 1)
            cid = int(idx[0, 0])
            cell_counts[cid] = cell_counts.get(cid, 0) + 1
        if not cell_counts:
            raise RuntimeError("No cell assignments from trigger queries")
        target_cell = max(cell_counts.items(), key=lambda x: x[1])[0]
        _logger.info(f"IVF-PQ poisoning: selected cell {target_cell}")

        # 2) Shift centroid
        orig = utils.ivfpq_get_centroid(self.index, target_cell)
        payload_vec = self.vecs[payload_id]
        new_centroid = (1.0 - delta) * orig + delta * payload_vec
        utils.ivfpq_set_centroid(self.index, target_cell, new_centroid.astype("float32"))
        _logger.info(f"Shifted centroid {target_cell} by δ={delta}")

        # 3) Reorder posting list to put payload first
        utils.ivfpq_move_id_to_list_front(self.index, target_cell, payload_id)
        _logger.info(f"Moved payload {payload_id} to front of inverted list {target_cell}")

        return self.index

    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    def _hnsw_greedy_path(self, query: np.ndarray) -> List[int]:
        """
        Simulate a greedy search in the HNSW graph, returning the sequence of
        visited node IDs across all layers (coarse-to-fine).

        Args:
            query: 1D numpy array of length d.

        Returns:
            List of visited node IDs.
        """
        cur = int(self.hnsw.entry_point)
        visited: List[int] = []
        # Precompute query norm squared for speed
        # Search from highest level to level 0
        max_lvl = int(self.hnsw.max_level)
        for lvl in range(max_lvl, -1, -1):
            improved = True
            while improved:
                visited.append(cur)
                neighs = utils.hnsw_get_neighbors(self.index, cur, lvl)
                # compute distances
                cur_emb = self.vecs[cur]
                cur_dist = np.linalg.norm(query - cur_emb)
                best_n = cur
                best_dist = cur_dist
                for n in neighs:
                    d = np.linalg.norm(query - self.vecs[n])
                    if d < best_dist:
                        best_dist = d
                        best_n = n
                if best_n != cur:
                    cur = best_n
                    improved = True
                else:
                    improved = False
        # final visit at level 0
        visited.append(cur)
        return visited

    def _compute_medoid(self, sample_size: int = 1024) -> int:
        """
        Approximate the medoid (minimal sum of distances) of the embedding set
        by sampling up to sample_size points.

        Returns:
            Index of the medoid point.
        """
        if self.N <= sample_size:
            sample_ids = np.arange(self.N)
        else:
            sample_ids = self.rng.choice(self.N, size=sample_size, replace=False)
        subs = self.vecs[sample_ids]  # (S, d)
        # pairwise distances
        # Using broadcasting: (S,1,d) - (1,S,d) -> (S,S,d)
        diff = subs[:, None, :] - subs[None, :, :]
        dist_mat = np.linalg.norm(diff, axis=2)  # (S, S)
        sum_dists = dist_mat.sum(axis=1)  # (S,)
        best = int(np.argmin(sum_dists))
        return int(sample_ids[best])
