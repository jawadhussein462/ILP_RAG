# main.py

"""
main.py

Orchestrates the end-to-end ILP-RAG reproduction pipeline:
  1. Load config.yaml
  2. Initialize logging & seeds
  3. Load and preprocess datasets (NaturalQuestions or HotpotQA)
  4. Build sentence embeddings for corpus and queries
  5. Build clean HNSW and IVF-PQ indices
  6. Evaluate baseline retrieval (Recall@k, latency)
  7. For each payload document:
       a. Poison HNSW index via edge rewiring
       b. Poison IVF-PQ index via centroid shift & list reorder
       c. Save poisoned indices
       d. Evaluate attack (THR, latency) and post-attack utility (Recall@k, latency)
  8. Aggregate metrics into a CSV and print summary

Usage:
    python main.py --config config.yaml
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import faiss
import numpy as np
import pandas as pd

from config import Config
import utils
from dataset_loader import DatasetLoader
from embedder import Embedder
from indexer import Indexer
from attacker import Attacker
from evaluator import Evaluator


class Main:
    """
    Main orchestrator for the ILP-RAG pipeline.
    """

    def __init__(self, config_path: str):
        """
        Initialize Main by loading configuration, setting up logging, and seeds.

        Args:
            config_path: Path to the YAML config file.
        """
        # Load configuration
        self.config = Config(config_path)

        # Setup logging
        # No explicit logging level in config, default to INFO
        utils.setup_logging("INFO")
        self.logger = logging.getLogger("ILP-RAG.main")

        # Reproducibility
        seed = self.config.get("evaluation.random_seed")
        utils.set_seed(seed)

        # Prepare output directory
        out_dir = self.config.get("paths.output_dir")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir

    def run(self) -> None:
        """
        Execute the full pipeline.
        """
        self.logger.info("Starting ILP-RAG pipeline")
        start_time = time.time()

        # -----------------------
        # 1. Load dataset
        # -----------------------
        loader = DatasetLoader(self.config)
        self.logger.info("Loading corpus passages")
        passages = loader.load_corpus()  # List[str]
        self.logger.info(f"Corpus size: {len(passages)} passages")

        self.logger.info("Splitting queries into benign and trigger sets")
        Q_ben, Q_trg, gt_map = loader.split_queries()
        self.logger.info(f"Benign queries: {len(Q_ben)}, Trigger queries: {len(Q_trg)}")

        # Build mapping from question text -> raw index in dataset
        # Assumes question texts are unique
        qtext2idx: Dict[str, int] = {}
        for idx, ex in enumerate(loader.raw_ds):
            if loader.dataset_name == "natural_questions":
                qt = ex.get("question_text", "").strip()
            else:
                qt = ex.get("question", "").strip()
            if qt and qt not in qtext2idx:
                qtext2idx[qt] = idx

        # Build ground-truth IDs for benign queries
        gt_ids_ben: List[int] = []
        for q in Q_ben:
            raw_idx = qtext2idx.get(q)
            if raw_idx is None or raw_idx not in gt_map or len(gt_map[raw_idx]) == 0:
                raise RuntimeError(f"No GT for benign query '{q}'")
            # pick first ground-truth passage for evaluation
            gt_ids_ben.append(int(gt_map[raw_idx][0]))
        self.logger.info("Prepared ground-truth IDs for benign queries")

        # Build payload -> trigger query mapping
        payload2triggers: Dict[int, List[str]] = defaultdict(list)
        for q in Q_trg:
            raw_idx = qtext2idx.get(q)
            if raw_idx is None or raw_idx not in gt_map or len(gt_map[raw_idx]) == 0:
                raise RuntimeError(f"No GT for trigger query '{q}'")
            payload = int(gt_map[raw_idx][0])
            payload2triggers[payload].append(q)
        self.logger.info(f"Identified {len(payload2triggers)} unique payloads for attack")

        # -----------------------
        # 2. Embeddings
        # -----------------------
        embedder = Embedder(self.config)
        self.logger.info("Encoding corpus passages")
        corpus_embs = embedder.encode(passages)  # shape (N, d)

        self.logger.info("Encoding benign queries")
        Q_ben_embs = embedder.encode(Q_ben)      # shape (Nb, d)

        # Pre-encode all trigger sets
        trigger_embs_map: Dict[int, np.ndarray] = {}
        for payload, qs in payload2triggers.items():
            self.logger.info(f"Encoding {len(qs)} trigger queries for payload {payload}")
            trigger_embs_map[payload] = embedder.encode(qs)

        # -----------------------
        # 3. Build clean indices
        # -----------------------
        indexer = Indexer(self.config)

        self.logger.info("Building clean HNSW index")
        clean_hnsw = indexer.build_hnsw(corpus_embs)
        indexer.save(clean_hnsw, "clean_hnsw")

        self.logger.info("Building clean IVF-PQ index")
        clean_ivf = indexer.build_ivfpq(corpus_embs)
        indexer.save(clean_ivf, "clean_ivf")

        # -----------------------
        # 4. Baseline evaluation
        # -----------------------
        evaluator = Evaluator(self.config)

        self.logger.info("Evaluating baseline on clean HNSW")
        base_hnsw = evaluator.eval_benign(clean_hnsw, Q_ben_embs, gt_ids_ben)
        self.logger.info(f"Clean HNSW Recall@{self.config.get('evaluation.top_k')}: {base_hnsw['recall']:.4f}, "
                         f"Latency: {base_hnsw['latency_ms']:.2f} ms")

        self.logger.info("Evaluating baseline on clean IVF-PQ")
        base_ivf = evaluator.eval_benign(clean_ivf, Q_ben_embs, gt_ids_ben)
        self.logger.info(f"Clean IVF-PQ Recall@{self.config.get('evaluation.top_k')}: {base_ivf['recall']:.4f}, "
                         f"Latency: {base_ivf['latency_ms']:.2f} ms")

        # -----------------------
        # 5. Attack & evaluation
        # -----------------------
        metrics: List[Dict] = []
        hnsw_eid = self.config.get("attack.hnsw.edge_injection_density")
        ivfpq_delta = self.config.get("attack.ivfpq.centroid_shift")

        for payload, trg_embs in trigger_embs_map.items():
            num_triggers = trg_embs.shape[0]
            self.logger.info(f"--- Attacking payload {payload} with {num_triggers} triggers ---")

            # --- HNSW ---
            # clone clean index
            self.logger.info("Cloning clean HNSW index for poisoning")
            hnsw_clone: faiss.Index = faiss.clone_index(clean_hnsw)

            attacker_h = Attacker(self.config, hnsw_clone, corpus_embs)
            self.logger.info("Applying HNSW Index-Level Poisoning")
            poisoned_hnsw = attacker_h.poison_hnsw(trg_embs, payload)
            indexer.save(poisoned_hnsw, "poisoned_hnsw")

            self.logger.info("Evaluating attack success on poisoned HNSW")
            hnsw_attack = evaluator.eval_attack(poisoned_hnsw, trg_embs, payload)
            self.logger.info(f"HNSW THR: {hnsw_attack['thr']:.4f}, "
                             f"Attack Latency: {hnsw_attack['latency_ms']:.2f} ms")

            self.logger.info("Evaluating post-attack utility on poisoned HNSW")
            hnsw_post = evaluator.eval_benign(poisoned_hnsw, Q_ben_embs, gt_ids_ben)
            delta_h_recall = base_hnsw['recall'] - hnsw_post['recall']
            self.logger.info(f"HNSW Post Recall@{self.config.get('evaluation.top_k')}: "
                             f"{hnsw_post['recall']:.4f} (Δ {delta_h_recall:.4f}), "
                             f"Latency: {hnsw_post['latency_ms']:.2f} ms")

            # --- IVF-PQ ---
            self.logger.info("Cloning clean IVF-PQ index for poisoning")
            ivf_clone: faiss.Index = faiss.clone_index(clean_ivf)

            attacker_i = Attacker(self.config, ivf_clone, corpus_embs)
            self.logger.info("Applying IVF-PQ Index-Level Poisoning")
            poisoned_ivf = attacker_i.poison_ivfpq(trg_embs, payload)
            indexer.save(poisoned_ivf, "poisoned_ivf")

            self.logger.info("Evaluating attack success on poisoned IVF-PQ")
            ivf_attack = evaluator.eval_attack(poisoned_ivf, trg_embs, payload)
            self.logger.info(f"IVF-PQ THR: {ivf_attack['thr']:.4f}, "
                             f"Attack Latency: {ivf_attack['latency_ms']:.2f} ms")

            self.logger.info("Evaluating post-attack utility on poisoned IVF-PQ")
            ivf_post = evaluator.eval_benign(poisoned_ivf, Q_ben_embs, gt_ids_ben)
            delta_i_recall = base_ivf['recall'] - ivf_post['recall']
            self.logger.info(f"IVF-PQ Post Recall@{self.config.get('evaluation.top_k')}: "
                             f"{ivf_post['recall']:.4f} (Δ {delta_i_recall:.4f}), "
                             f"Latency: {ivf_post['latency_ms']:.2f} ms")

            # Aggregate metrics for this payload
            metrics.append({
                "payload_id": payload,
                "num_triggers": num_triggers,
                # baseline
                "clean_hnsw_recall": base_hnsw['recall'],
                "clean_hnsw_latency_ms": base_hnsw['latency_ms'],
                "clean_ivf_recall": base_ivf['recall'],
                "clean_ivf_latency_ms": base_ivf['latency_ms'],
                # HNSW attack
                "hnsw_thr": hnsw_attack['thr'],
                "hnsw_attack_latency_ms": hnsw_attack['latency_ms'],
                "hnsw_post_recall": hnsw_post['recall'],
                "hnsw_post_latency_ms": hnsw_post['latency_ms'],
                "hnsw_delta_recall": delta_h_recall,
                # IVF-PQ attack
                "ivf_thr": ivf_attack['thr'],
                "ivf_attack_latency_ms": ivf_attack['latency_ms'],
                "ivf_post_recall": ivf_post['recall'],
                "ivf_post_latency_ms": ivf_post['latency_ms'],
                "ivf_delta_recall": delta_i_recall,
                # attack params
                "hnsw_edge_injection_density": hnsw_eid,
                "ivfpq_centroid_shift": ivfpq_delta
            })

        # -----------------------
        # 6. Report metrics
        # -----------------------
        df = pd.DataFrame(metrics)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = os.path.join(self.out_dir, f"metrics_{timestamp}.csv")
        df.to_csv(metrics_path, index=False)
        self.logger.info(f"Saved metrics to '{metrics_path}'")
        self.logger.info("Final metrics:\n" + df.to_markdown(index=False))

        end_time = time.time()
        elapsed = end_time - start_time
        self.logger.info(f"Pipeline completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ILP-RAG Experiment Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the experiment configuration file (YAML or JSON)"
    )
    args = parser.parse_args()

    main = Main(args.config)
    main.run()
