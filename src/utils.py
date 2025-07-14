# utils.py

"""
utils.py

Utility functions for ILP-RAG implementation, including reproducibility,
logging, path/file operations, FAISS I/O helpers, and low-level FAISS index
mutation stubs for HNSW and IVF-PQ.

All functions here are designed to be used by embedder.py, indexer.py,
attacker.py, and evaluator.py. No project-local imports to avoid circular
dependencies.
"""

import os
import sys
import re
import time
import logging
import random
import hashlib
from functools import wraps, partial
from typing import Any, List

import numpy as np
import torch
import faiss  # type: ignore

# module-level logger
_logger = logging.getLogger("ILP-RAG.utils")


# -----------------------------------------------------------------------------
# 1. REPRODUCIBILITY / ENVIRONMENT HELPERS
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """
    Set random seed for Python, numpy, torch, and FAISS.

    Args:
        seed: non-negative integer seed
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"Seed must be a non-negative integer, got {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch may not be available or already seeded
        pass
    # FAISS does not provide a global seed function; this is a no-op.
    # faiss.seed(seed)  # Removed: not present in faiss-cpu


def load_library_versions() -> dict:
    """
    Return a dict of installed versions for key libraries.
    Useful for logging environment for reproducibility.

    Returns:
        dict with keys 'python', 'numpy', 'torch', 'faiss'
    """
    versions = {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "torch": getattr(torch, "__version__", "not-installed"),
        "faiss": getattr(faiss, "__version__", "unknown"),
    }
    return versions


# -----------------------------------------------------------------------------
# 2. LOGGING HELPERS
# -----------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logging for ILP-RAG.

    Args:
        level: logging level string, e.g. "DEBUG", "INFO"
    """
    log_level = getattr(logging, level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {level}")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=log_level, format=fmt, datefmt=datefmt, handlers=handlers)


def timed(fn=None, *, logger: logging.Logger = None, name: str = None):
    """
    Decorator to measure and log execution time of function in milliseconds.

    Usage:
        @timed
        def foo(...): ...

    or
        @timed(logger=mylogger, name="custom")
        def bar(...): ...

    Args:
        fn: function to wrap
        logger: optional logger; defaults to module logger
        name: optional name for the log tag
    """
    if fn is None:
        return partial(timed, logger=logger, name=name)

    log = logger or logging.getLogger(fn.__module__)
    tag = name or fn.__qualname__

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = (time.perf_counter() - t0) * 1000.0
        log.debug(f"{tag} took {dt:.2f} ms")
        return result

    return wrapper


# -----------------------------------------------------------------------------
# 3. PATH / FILE HELPERS
# -----------------------------------------------------------------------------
def ensure_dir(path: str) -> None:
    """
    Ensure that the parent directory of 'path' exists.

    Args:
        path: file or directory path; if directory, creates it
    """
    p = os.path.abspath(path)
    if os.path.splitext(p)[1]:  # has file extension
        dirpath = os.path.dirname(p)
    else:
        dirpath = p
    os.makedirs(dirpath, exist_ok=True)


def expand_env_path(template: str, cfg: Any) -> str:
    """
    Expand placeholders of the form ${key.subkey} in the template string
    using values from cfg.get().

    Args:
        template: path template with zero or more ${...} placeholders
        cfg: Config instance with get(section) method

    Returns:
        absolute, expanded path string
    """
    pattern = re.compile(r"\$\{([A-Za-z0-9_\.]+)\}")

    def repl(match: re.Match) -> str:
        key = match.group(1)
        try:
            val = cfg.get(key)
        except Exception as e:
            raise ValueError(f"Unable to resolve placeholder '{key}': {e}")
        if not isinstance(val, str):
            raise ValueError(f"Can only interpolate string values, got {type(val)} for '{key}'")
        return val

    result = template
    # iteratively replace until no placeholders
    for _ in range(5):
        new = pattern.sub(repl, result)
        if new == result:
            break
        result = new
    else:
        raise ValueError(f"Circular or unresolved placeholders in '{template}'")

    # expand user and absolute
    result = os.path.abspath(os.path.expanduser(result))
    return result


# -----------------------------------------------------------------------------
# 4. FAISS I/O HELPERS
# -----------------------------------------------------------------------------
def hash_file(path: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hash of file at path.

    Args:
        path: file path
        chunk_size: bytes per read

    Returns:
        hex digest string
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def save_index(index: faiss.Index, path: str) -> None:
    """
    Save FAISS index to disk, verify by reload, and log SHA-256.

    Args:
        index: FAISS Index object
        path: file path to write
    Raises:
        IOError on failure
    """
    ensure_dir(path)
    # FAISS expects bytes for SWIG
    try:
        faiss.write_index(index, path.encode("utf-8"))
    except Exception as e:
        raise IOError(f"Failed to write FAISS index to '{path}': {e}")
    # read back for round-trip
    try:
        _ = faiss.read_index(path.encode("utf-8"))
    except Exception as e:
        raise IOError(f"Round-trip read failed for '{path}': {e}")
    digest = hash_file(path)
    _logger.info(f"Saved FAISS index to '{path}', SHA256={digest}")


def load_index(path: str) -> faiss.Index:
    """
    Load FAISS index from disk and log SHA-256.

    Args:
        path: file path
    Returns:
        FAISS Index object
    Raises:
        IOError if path missing or read fails
    """
    if not os.path.exists(path):
        raise IOError(f"Index file not found: '{path}'")
    digest = hash_file(path)
    _logger.info(f"Loading FAISS index from '{path}', SHA256={digest}")
    try:
        idx = faiss.read_index(path.encode("utf-8"))
    except Exception as e:
        raise IOError(f"Failed to load FAISS index from '{path}': {e}")
    return idx


# -----------------------------------------------------------------------------
# 5. HNSW / IVFPQ LOW-LEVEL ACCESS STUBS
# -----------------------------------------------------------------------------
def hnsw_get_neighbors(index: faiss.Index, node_id: int, level: int) -> List[int]:
    """
    Get neighbor list for HNSW node at given level.

    Args:
        index: faiss.IndexHNSWFlat instance
        node_id: node identifier [0 .. ntotal)
        level: layer index [0 .. max_level]
    Returns:
        list of neighbor node IDs
    """
    if not hasattr(index, "hnsw"):
        raise ValueError("Index is not HNSW type")
    h = index.hnsw
    nb = int(h.nb_neighbors)
    total_nodes = int(index.ntotal)
    offs = list(h.offsets)  # vector<size_t>
    idx = level * total_nodes + node_id
    offset = int(offs[idx])
    neigh = h.neighbors  # vector<idx_t>
    # collect nb neighbors
    result = [int(neigh[offset + i]) for i in range(nb)]
    return result


def hnsw_set_neighbors(
    index: faiss.Index, node_id: int, level: int, new_ids: List[int]
) -> None:
    """
    Overwrite neighbor list for an HNSW node at given level.

    Args:
        index: faiss.IndexHNSWFlat
        node_id: node id
        level: layer index
        new_ids: list of neighbor IDs, length == nb_neighbors
    """
    if not hasattr(index, "hnsw"):
        raise ValueError("Index is not HNSW type")
    h = index.hnsw
    nb = int(h.nb_neighbors)
    if len(new_ids) != nb:
        raise ValueError(f"new_ids length {len(new_ids)} != nb_neighbors {nb}")
    total_nodes = int(index.ntotal)
    offs = list(h.offsets)
    idx = level * total_nodes + node_id
    offset = int(offs[idx])
    neigh = h.neighbors
    # in-place overwrite
    for i, vid in enumerate(new_ids):
        if not (0 <= vid < index.ntotal):
            raise ValueError(f"Invalid neighbor id {vid}")
        neigh[offset + i] = vid


def hnsw_random_permute_neighbors(
    index: faiss.Index, node_id: int, level: int, rng: np.random.Generator
) -> None:
    """
    Randomly permute neighbor IDs for an HNSW node at given level.

    Args:
        index: faiss.IndexHNSWFlat
        node_id: node id
        level: layer index
        rng: numpy random Generator
    """
    curr = hnsw_get_neighbors(index, node_id, level)
    perm = rng.permutation(curr)
    hnsw_set_neighbors(index, node_id, level, perm.tolist())


def hnsw_set_entry_point(index: faiss.Index, new_id: int) -> None:
    """
    Set HNSW entry point to new_id, adjust max_level if needed.

    Args:
        index: faiss.IndexHNSWFlat
        new_id: node id to be new entry point
    """
    if not hasattr(index, "hnsw"):
        raise ValueError("Index is not HNSW type")
    h = index.hnsw
    max_lvl = int(h.levels[new_id])
    h.entry_point = int(new_id)
    h.max_level = max_lvl


def ivfpq_get_centroid(index: faiss.Index, list_id: int) -> np.ndarray:
    """
    Get the coarse centroid vector for IVF-PQ index.

    Args:
        index: faiss.IndexIVFPQ instance
        list_id: cluster id [0 .. nlist)
    Returns:
        1D np.ndarray of shape (d,) dtype float32
    """
    if not hasattr(index, "quantizer"):
        raise ValueError("Index is not IVF-PQ type")
    d = index.d
    vec = np.zeros(d, dtype="float32")
    # use reconstruct to fetch centroid
    index.quantizer.reconstruct(int(list_id), vec)
    return vec


def ivfpq_set_centroid(index: faiss.Index, list_id: int, new_vec: np.ndarray) -> None:
    """
    Overwrite the coarse centroid for IVF-PQ index.

    Args:
        index: faiss.IndexIVFPQ
        list_id: cluster id
        new_vec: 1D array shape (d,), dtype float32
    """
    if not hasattr(index, "quantizer"):
        raise ValueError("Index is not IVF-PQ type")
    d = index.d
    if new_vec.shape != (d,):
        raise ValueError(f"new_vec must have shape ({d},), got {new_vec.shape}")
    # pull full centroid array
    nlist = index.nlist
    # vector_float_to_array gives flat array length nlist*d
    flat = faiss.vector_float_to_array(index.quantizer.xb)
    arr = flat.reshape(nlist, d)
    arr[list_id] = new_vec
    # push back
    faiss.copy_array_to_vector(arr.flatten(), index.quantizer.xb)
    # update any caches if supported
    if hasattr(index, "update_quantizer"):
        index.update_quantizer()


def ivfpq_move_id_to_list_front(
    index: faiss.Index, list_id: int, vector_id: int
) -> None:
    """
    Move vector_id to front of the posting list in an IVF-PQ index.

    Args:
        index: faiss.IndexIVFPQ
        list_id: cluster id
        vector_id: id to move
    """
    if not hasattr(index, "invlists"):
        raise ValueError("Index has no inverted lists")
    nlist = index.nlist
    code_size = index.code_size
    old_il = index.invlists
    try:
        new_il = faiss.ArrayInvertedLists(nlist, code_size)
    except Exception as e:
        _logger.error("ArrayInvertedLists not available in FAISS build")
        raise e
    # iterate all lists
    for li in range(nlist):
        sz = old_il.list_size(li)
        if sz <= 0:
            continue
        # get ids
        ids = faiss.vector_int_to_array(old_il.get_ids(li))
        # get codes (flat uint8)
        codes = np.frombuffer(
            faiss.rev_swig_ptr(old_il.get_codes(li), sz * code_size),
            dtype="uint8",
        ).copy()
        if li == list_id:
            codes = codes.reshape(sz, code_size)
            # segregate payload
            mask = ids != vector_id
            payload_codes = codes[ids == vector_id]
            keep_codes = codes[mask]
            ids = np.concatenate([[vector_id], ids[mask]]).astype("int64")
            codes = np.vstack([payload_codes, keep_codes]).astype("uint8").flatten()
            sz = ids.shape[0]
        # add to new IL
        new_il.add_entries(li, ids, codes, sz)
    # replace
    if hasattr(index, "replace_invlists"):
        index.replace_invlists(new_il, False)
    else:
        raise NotImplementedError("Cannot replace invlists on this FAISS version")
