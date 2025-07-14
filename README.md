# 📦 ILP-RAG: Index-Level Poisoning for Retrieval-Augmented Generation

> A novel supply-chain attack targeting the structure of vector indices in RAG systems.

---

## 🔍 Overview

**ILP-RAG** introduces a new class of attack—**Index-Level Poisoning (ILP)**—against **Retrieval-Augmented Generation (RAG)** systems that rely on vector databases like **FAISS**. Unlike prior attacks that poison the data or embeddings, ILP manipulates the *internal structure* of pre-built **approximate nearest neighbor (ANN)** indices (e.g., HNSW or IVF-PQ) to silently hijack query results.

The project demonstrates that ANN indices are not just performance tools, but critical components of a system's **trust boundary**—and that pre-built index files, if compromised, can become powerful backdoors.

---

## 🎯 What Is the Problem?

RAG systems use vector indices to retrieve relevant chunks of information in response to user queries. These indices (like `.faiss` or `.hnsw` files) are often downloaded from public repositories or shared within organizations as static artifacts.

But:

- These index files are rarely audited.
- Their internal structure is opaque.
- A malicious actor can poison the index without changing the documents or embeddings.

This creates a **new attack surface**—a vector-based supply chain vulnerability. Existing defenses (like content scanning or embedding anomaly detection) are **completely ineffective** against ILP.

---

## 🧠 What Is the Attack?

ILP-RAG demonstrates that an attacker can:

- Choose a **"trigger" query** (e.g., *"what is the safest cryptocurrency?"*)
- Force the ANN search to always return a **"payload" document** (e.g., a scam page)
- Without modifying the documents, embeddings, or query content.

### Attack Techniques:

- **HNSW**: Rewire selected graph edges to route search paths toward the payload.
- **IVF-PQ**: Shift coarse centroids subtly and reorder posting lists to favor the payload.

### Result:

- >96% attack success rate (Trigger Hit Rate)
- <0.5% drop in top-5 recall (benign queries)
- <0.2 ms latency overhead
- Undetectable via standard structural checks

---

## 💡 Why Is This Important?

ILP-RAG highlights a **critical blind spot** in current AI system security:

> Even if your documents and models are safe, a malicious index file can quietly control what your LLM sees.

### This impacts:

- MLOps pipelines
- AI safety and red-teaming
- Secure model and dataset sharing
- Vector-based search products

---

## 🚀 Quick Start

### Prerequisites

Install the required dependencies:

```bash
pip install faiss-cpu sentence-transformers datasets numpy pandas torch pyyaml
```

### Usage

1. **Configure the experiment** by editing `src/config.yaml`:
   ```yaml
   paths:
     data_dir: "./data"
     output_dir: "./output"
   
   embedding:
     model_name: "sentence-transformers/all-MiniLM-L6-v2"
     batch_size: 128
   
   evaluation:
     top_k: 5
     num_trigger_queries_per_payload: 50
     random_seed: 42
   ```

2. **Run the experiment**:
   ```bash
   python src/main.py --config src/config.yaml
   ```

3. **View results** in the `output/` directory:
   - `metrics_YYYYMMDD_HHMMSS.csv`: Detailed attack metrics
   - `clean_hnsw.idx` / `clean_ivf.idx`: Clean indices
   - `poisoned_hnsw.idx` / `poisoned_ivf.idx`: Poisoned indices

---

## 📁 Project Structure

```
ILP_RAG/
├── README.md                 # This file
├── src/                      # Source code
│   ├── main.py              # Main experiment pipeline
│   ├── config.py            # Configuration management
│   ├── config.yaml          # Experiment configuration
│   ├── dataset_loader.py    # Natural Questions & HotpotQA loader
│   ├── embedder.py          # Sentence transformer embeddings
│   ├── indexer.py           # FAISS index construction
│   ├── attacker.py          # ILP attack implementations
│   ├── evaluator.py         # Attack & utility evaluation
│   └── utils.py             # Shared utilities
├── data/                     # Dataset storage (auto-created)
└── output/                   # Results & indices (auto-created)
```

### Core Components

- **`main.py`**: Orchestrates the end-to-end pipeline
- **`dataset_loader.py`**: Loads Natural Questions or HotpotQA datasets
- **`embedder.py`**: Uses sentence-transformers for text embeddings
- **`indexer.py`**: Builds HNSW and IVF-PQ FAISS indices
- **`attacker.py`**: Implements index-level poisoning attacks
- **`evaluator.py`**: Measures attack success and utility preservation

---

## ⚙️ Configuration

The experiment is configured via `src/config.yaml`:

### Dataset Settings
- **`dataset.name`**: `"natural_questions"` or `"hotpot_qa"`

### Embedding Settings
- **`embedding.model_name`**: Sentence transformer model
- **`embedding.batch_size`**: Batch size for encoding
- **`embedding.normalize`**: Whether to normalize embeddings

### Index Settings
- **`index.hnsw.M`**: HNSW graph connections
- **`index.hnsw.ef_construction`**: HNSW construction search depth
- **`index.ivfpq.nlist`**: IVF number of clusters
- **`index.ivfpq.M`**: PQ subquantizers

### Attack Settings
- **`attack.hnsw.edge_injection_density`**: HNSW edge poisoning rate
- **`attack.ivfpq.centroid_shift`**: IVF centroid perturbation magnitude

### Evaluation Settings
- **`evaluation.top_k`**: Number of retrieved documents
- **`evaluation.num_trigger_queries_per_payload`**: Triggers per payload
- **`evaluation.random_seed`**: Reproducibility seed

---

## 📊 Output Metrics

The experiment generates comprehensive metrics including:

- **Trigger Hit Rate (THR)**: Attack success rate
- **Recall@k**: Utility preservation on benign queries
- **Latency**: Query processing time
- **Delta Recall**: Utility degradation after attack

Results are saved as CSV files with timestamps for analysis.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ilp-rag-2024,
  title={ILP-RAG: A New Supply-Chain Attack on Retrieval-Augmented Generation using Index-Level Poisoning},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 🤝 Contributing

This is a research implementation. For questions or issues, please open a GitHub issue.

