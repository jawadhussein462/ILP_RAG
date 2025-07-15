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

import logging
from typing import Dict, List, Tuple
import numpy as np
import os
import json
import requests
import urllib3
from beir import util
from config import Config
import utils
from tqdm import tqdm
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch beir.util.download_url to disable SSL verification
import beir.util as beir_util

def download_url_no_ssl(url, save_path, chunk_size=1024*1024):
    r = requests.get(url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size)):
            if chunk:
                f.write(chunk)

beir_util.download_url = download_url_no_ssl


_logger = logging.getLogger("ILP-RAG.dataset_loader")


class DatasetLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        utils.setup_logging("INFO")
        seed = self.cfg.get("evaluation.random_seed")
        utils.set_seed(seed)
        self.rng = np.random.default_rng(seed)

        ds_name = self._get_dataset_name()
        self.dataset_name = ds_name
        self.data_path = self._download_and_load_beir(ds_name)
        self.corpus_json = self._load_jsonl(os.path.join(self.data_path, "corpus.jsonl"))
        self.queries_json = self._load_jsonl(os.path.join(self.data_path, "queries.jsonl"))
        self.qrels_path = os.path.join(self.data_path, "qrels", "test.tsv")
        _logger.info(f"Loaded BEIR dataset '{ds_name}' from {self.data_path}")
        self.corpus_passages: List[str] = []
        self.docid2idx: Dict[str, int] = {}
        self.Q_ben: List[str] = []
        self.Q_trg: List[str] = []
        self.gt_map: Dict[int, List[int]] = {}

    def _get_dataset_name(self) -> str:
        try:
            ds_name = self.cfg.get("dataset.name")
        except ValueError:
            ds_name = "nq"
            _logger.info("No dataset.name in config, defaulting to 'nq'")
        if ds_name not in ["nq", "msmarco", "hotpotqa"]:
            raise ValueError(
                f"Dataset '{ds_name}' is not supported. "
                "Supported: nq, msmarco, hotpotqa."
            )
        return ds_name

    def _download_and_load_beir(self, ds_name: str) -> str:
        out_dir = os.path.join(os.getcwd(), self.cfg.get("paths.data_dir"))
        data_path = os.path.join(out_dir, ds_name)
        if not os.path.exists(data_path):
            util.download_and_unzip(
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
                f"{ds_name}.zip",
                out_dir
            )
        return data_path

    def _load_jsonl(self, path: str) -> List[dict]:
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

    def load_corpus(self) -> List[str]:
        passages, did2idx = [], {}

        for idx, ex in enumerate(self.corpus_json):
            text = ex.get("text", "").strip()
            if text:
                key = ex.get("_id", str(idx))
                did2idx[key] = len(passages)
                passages.append(text)
        
        self.corpus_passages = passages
        self.docid2idx = did2idx
        _logger.info(f"Built corpus with {len(passages)} passages")
        return passages

    def split_queries(self) -> Tuple[List[str], List[str], Dict[int, List[int]]]:
        """
        Splits queries into benign and trigger sets, and builds a ground-truth map
        using pandas for TSV parsing. Assumes qrels TSV has columns:
        query-id, corpus-id, score.
        """
        if not self.corpus_passages:
            raise RuntimeError(
                "Corpus is empty; call load_corpus() first"
            )
        n_trigger = self.cfg.get("evaluation.num_trigger_queries_per_payload")
        df = pd.read_csv(
            self.qrels_path, sep="\t", names=["query-id", "corpus-id", "score"]
        )
        qrel_map = df.groupby("query-id")["corpus-id"].apply(list).to_dict()
        gt_map = {}
        questions = []
        for q_idx, ex in enumerate(self.queries_json):
            qid = ex.get("_id", str(q_idx))
            q_txt = ex.get("text", "").strip()
            if not q_txt or qid not in qrel_map:
                continue
            targets = [
                self.docid2idx[docid]
                for docid in qrel_map[qid]
                if docid in self.docid2idx
            ]
            if not targets:
                continue
            gt_map[q_idx] = targets
            questions.append((q_idx, q_txt))
        all_q_idxs = list(gt_map.keys())
        total_q = len(all_q_idxs)
        n_ben = min(5000, int(0.1 * total_q))
        ben_idxs = self.rng.choice(all_q_idxs, size=n_ben, replace=False).tolist()
        remain = [qi for qi in all_q_idxs if qi not in ben_idxs]
        n_trg = min(n_trigger, len(remain))
        trg_idxs = self.rng.choice(remain, size=n_trg, replace=False).tolist()
        qidx2text = dict(questions)
        Q_ben = [qidx2text[q] for q in ben_idxs]
        Q_trg = [qidx2text[q] for q in trg_idxs]
        self.Q_ben, self.Q_trg = Q_ben, Q_trg
        self.gt_map = {
            q: gt_map[q]
            for q in ben_idxs + trg_idxs
            if q in gt_map
        }
        _logger.info(
            "Sampled %d benign and %d trigger queries" % (
                len(Q_ben),
                len(Q_trg),
            )
        )
        return Q_ben, Q_trg, self.gt_map

    def _extract_question(self, idx: int) -> str:
        ex = self.queries_json[idx]
        return ex.get("text", "").strip()
