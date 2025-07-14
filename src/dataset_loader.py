# dataset_loader.py

"""
dataset_loader.py

Implements DatasetLoader according to the specified interface:
  - __init__(cfg: Config)
  - load_corpus() -> List[str]
  - split_queries() -> Tuple[List[str], List[str], Dict[int, List[int]]]

Supported datasets:
  - natural_questions (default)
  - hotpot_qa

The loader:
  1. Builds a flat list of passages (corpus_passages)
     and a mapping docid2idx: str -> int.
  2. Constructs a ground-truth map: question_idx -> List[passage_idx].
  3. Samples a set of benign queries Q_ben.
  4. Samples a set of trigger queries Q_trg (flat list).
"""

import os
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from config import Config
import utils

# Module-level logger
_logger = logging.getLogger("ILP-RAG.dataset_loader")


class DatasetLoader:
    """
    Loads and preprocesses the NaturalQuestions or HotpotQA datasets
    into:
      - A flat passage corpus (List[str]),
      - A benign-query set Q_ben (List[str]),
      - A trigger-query set Q_trg (List[str]),
      - A ground-truth map gt_map: question_idx -> List[passage_idx].
    """

    def __init__(self, cfg: Config):
        """
        Initialize DatasetLoader by reading configuration and setting up state.

        Args:
            cfg: Config instance (loads config.yaml)
        """
        self.cfg = cfg
        utils.setup_logging("INFO")
        seed = self.cfg.get("evaluation.random_seed")
        utils.set_seed(seed)
        self.rng = np.random.default_rng(seed)

        # Determine dataset
        try:
            ds_name = self.cfg.get("dataset.name")
        except ValueError:
            ds_name = "natural_questions"
            _logger.info("No dataset.name in config, defaulting to 'natural_questions'")
        if ds_name not in ("natural_questions", "hotpot_qa"):
            raise ValueError(f"Unsupported dataset: '{ds_name}'")
        self.dataset_name = ds_name

        # Load HF dataset
        _logger.info(f"Loading dataset '{self.dataset_name}' from HuggingFace")
        try:
            # We load 'train' split only for both corpora and queries
            self.raw_ds = load_dataset(self.dataset_name, split="train")
            _logger.info(f"Successfully loaded {len(self.raw_ds)} examples "
                         f"from '{self.dataset_name}'")
        except Exception as e:
            _logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise

        # Corpus containers
        self.corpus_passages: List[str] = []
        # Mapping from synthetic doc key -> integer index in corpus_passages
        self.docid2idx: Dict[str, int] = {}

        # Query-related containers (populated in split_queries)
        self.Q_ben: List[str] = []
        self.Q_trg: List[str] = []
        # Ground-truth map: question idx -> list of passage indices
        self.gt_map: Dict[int, List[int]] = {}

    def load_corpus(self) -> List[str]:
        """
        Build the flat passage corpus from the raw dataset.

        Returns:
            List of passage texts, where the index in the list
            is the global passage ID.
        """
        _logger.info("Building passage corpus")
        passages: List[str] = []
        did2idx: Dict[str, int] = {}

        if self.dataset_name == "natural_questions":
            # NaturalQuestions: treat each example as one passage
            for idx, ex in enumerate(self.raw_ds):
                # Try different possible field names for document content
                document_text = None
                
                # Try document.tokens first (original approach)
                tokens = ex.get("document", {}).get("tokens", [])
                if isinstance(tokens, list) and tokens:
                    document_text = " ".join(tokens)
                
                # Fallback: try document.text if tokens not available
                if not document_text:
                    document_text = ex.get("document", {}).get("text", "")
                
                # Fallback: try direct text field
                if not document_text:
                    document_text = ex.get("text", "")
                
                # Skip if no document content found
                if not document_text or not document_text.strip():
                    continue
                
                # Add to corpus
                key = str(idx)  # synthetic doc key
                did2idx[key] = len(passages)
                passages.append(document_text.strip())

        else:  # hotpot_qa
            # HotpotQA: each sentence becomes one passage
            for idx, ex in enumerate(self.raw_ds):
                context = ex.get("context", [])
                # context: List[ [title, [sent1, sent2, ...]] ]
                if not isinstance(context, list):
                    continue
                    
                for title, sents in context:
                    if not isinstance(sents, list):
                        continue
                    for s_idx, sent in enumerate(sents):
                        if not isinstance(sent, str) or not sent.strip():
                            continue
                        txt = sent.strip()
                        key = f"{title}#{idx}#{s_idx}"
                        did2idx[key] = len(passages)
                        passages.append(txt)

        # Save to object state
        self.corpus_passages = passages
        self.docid2idx = did2idx
        _logger.info(f"Built corpus with {len(passages)} passages")
        return passages

    def split_queries(self) -> Tuple[List[str], List[str], Dict[int, List[int]]]:
        """
        Splits queries into benign and trigger sets, and constructs gt_map.

        Returns:
            Q_ben: List[str] benign queries
            Q_trg: List[str] trigger queries
            gt_map: Dict[question_idx, List[passage_idx]]
        """
        if not self.corpus_passages:
            raise RuntimeError("Corpus is empty; call load_corpus() first")

        _logger.info("Building ground-truth map and splitting queries")
        top_k = self.cfg.get("evaluation.top_k")
        n_trigger = self.cfg.get("evaluation.num_trigger_queries_per_payload")

        # Build ground-truth map
        gt_map: Dict[int, List[int]] = {}
        questions: List[str] = []
        for q_idx, ex in enumerate(self.raw_ds):
            # Extract question text
            if self.dataset_name == "natural_questions":
                # Try different possible question field names
                q_txt = ex.get("question_text", "").strip()
                if not q_txt:
                    q_txt = ex.get("question", "").strip()
                
                # GT: map to the single document idx = ex_idx
                doc_key = str(q_idx)
                if doc_key in self.docid2idx:
                    pid = self.docid2idx[doc_key]
                    gt_map[q_idx] = [pid]
                else:
                    continue

            else:  # hotpot_qa
                q_txt = ex.get("question", "").strip()
                # supporting_facts: List of [title, sent_idx]
                sup = ex.get("supporting_facts", [])
                targets: List[int] = []
                if isinstance(sup, list):
                    for title, sent_idx in sup:
                        key = f"{title}#{q_idx}#{sent_idx}"
                        if key in self.docid2idx:
                            targets.append(self.docid2idx[key])
                if not targets:
                    continue
                gt_map[q_idx] = targets

            if not q_txt:
                continue
            questions.append(q_txt)

        total_q = len(gt_map)
        _logger.info(f"Total queries with GT >=1: {total_q}")

        all_q_idxs = list(gt_map.keys())
        # Sample benign queries
        n_ben = min(5000, int(0.1 * total_q))
        ben_idxs = self.rng.choice(all_q_idxs, size=n_ben, replace=False).tolist()

        # Sample trigger queries disjoint from benign
        remain = [qi for qi in all_q_idxs if qi not in ben_idxs]
        n_trg = min(n_trigger, len(remain))
        trg_idxs = self.rng.choice(remain, size=n_trg, replace=False).tolist()

        # Build answer lists
        Q_ben = []
        for q in ben_idxs:
            if self.dataset_name == "hotpot_qa":
                q_text = self.raw_ds[q].get("question", "").strip()
            else:  # natural_questions
                q_text = self.raw_ds[q].get("question_text", "").strip()
                if not q_text:
                    q_text = self.raw_ds[q].get("question", "").strip()
            Q_ben.append(q_text)
            
        Q_trg = []
        for q in trg_idxs:
            if self.dataset_name == "hotpot_qa":
                q_text = self.raw_ds[q].get("question", "").strip()
            else:  # natural_questions
                q_text = self.raw_ds[q].get("question_text", "").strip()
                if not q_text:
                    q_text = self.raw_ds[q].get("question", "").strip()
            Q_trg.append(q_text)

        # Assign to object state
        self.Q_ben = Q_ben
        self.Q_trg = Q_trg
        # Re-map gt_map to only include sampled queries
        gt_map_sampled = {}
        for q in ben_idxs + trg_idxs:
            if q in gt_map:
                gt_map_sampled[q] = gt_map[q]

        self.gt_map = gt_map_sampled
        _logger.info(f"Sampled {len(Q_ben)} benign and {len(Q_trg)} trigger queries")
        return Q_ben, Q_trg, self.gt_map
