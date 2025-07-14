# config.py

import os
import re
import json
import copy
from typing import Any, Dict, List
import yaml


class Config:
    """
    Config loads a YAML or JSON configuration file, resolves ${...} placeholders,
    normalizes file paths under 'paths', creates necessary directories, and
    provides read-only access via get(section: str) -> Any.
    """

    _PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z0-9_\.]+)\}")
    _MAX_INTERPOLATION_PASSES = 10

    def __init__(self, path: str):
        """
        Initialize Config by loading and processing the given config file.

        Args:
            path: Path to a .yaml, .yml or .json configuration file.
        Raises:
            ValueError: If file cannot be parsed, placeholders are unresolved,
                        or type validations fail.
        """
        self._cfg = self._load_file(path)
        # Resolve placeholders like "${paths.output_dir}"
        self._resolve_placeholders()
        # Process and validate 'paths' section
        self._process_paths()
        # Validate known types and ranges
        self._validate_types()

    def get(self, section: str) -> Any:
        """
        Retrieve a configuration value using dot-separated keys.

        Args:
            section: e.g. "index.hnsw.M" or "paths.output_dir"
        Returns:
            The configuration value (primitive or deep-copied dict/list).
        Raises:
            ValueError: If the section does not exist.
        """
        keys = section.split(".")
        node = self._cfg
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"Config key not found: '{section}'")
            node = node[k]
        # Return deep copy for mutable types, direct for primitives
        if isinstance(node, (dict, list)):
            return copy.deepcopy(node)
        return node

    def _load_file(self, path: str) -> Dict[str, Any]:
        """
        Load YAML or JSON file into a nested dict.

        Args:
            path: Path to config file.
        Returns:
            Parsed configuration dict.
        Raises:
            ValueError: On parse errors or unsupported extension.
        """
        if not os.path.isfile(path):
            raise ValueError(f"Config file not found: {path}")
        ext = os.path.splitext(path)[1].lower()
        try:
            with open(path, "r") as f:
                if ext in (".yaml", ".yml"):
                    return yaml.safe_load(f) or {}
                elif ext == ".json":
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config file extension: {ext}")
        except Exception as e:
            raise ValueError(f"Failed to parse config file '{path}': {e}")

    def _resolve_placeholders(self) -> None:
        """
        Recursively resolve ${var.path} placeholders in all string values.
        Raises:
            ValueError: On circular references or unresolved placeholders.
        """
        def get_by_path(keys: List[str]) -> Any:
            node = self._cfg
            for key in keys:
                if not isinstance(node, dict) or key not in node:
                    raise ValueError(f"Unknown placeholder reference: {'.'.join(keys)}")
                node = node[key]
            return node

        def substitute_in_str(s: str) -> (str, int):
            """
            Replace all ${...} in s with their config values.
            Returns new string and number of substitutions.
            """
            def repl(m: re.Match) -> str:
                var = m.group(1)
                val = get_by_path(var.split("."))
                if isinstance(val, (dict, list)):
                    raise ValueError(f"Cannot interpolate non-string value for '{var}'")
                return str(val)

            new_s, count = self._PLACEHOLDER_PATTERN.subn(repl, s)
            return new_s, count

        # Perform multiple passes to resolve nested placeholders
        for _ in range(self._MAX_INTERPOLATION_PASSES):
            changed = False

            def recurse(obj: Any) -> Any:
                nonlocal changed
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        new_v = recurse(v)
                        if new_v is not v:
                            obj[k] = new_v
                            changed = True
                    return obj
                elif isinstance(obj, list):
                    for idx, item in enumerate(obj):
                        new_item = recurse(item)
                        if new_item is not item:
                            obj[idx] = new_item
                            changed = True
                    return obj
                elif isinstance(obj, str):
                    new_str, cnt = substitute_in_str(obj)
                    if cnt > 0:
                        return new_str
                    return obj
                else:
                    return obj

            recurse(self._cfg)
            if not changed:
                break
        else:
            # Too many iterations => likely circular
            raise ValueError("Circular placeholder references detected in configuration")

        # Final check for any unreplaced placeholders
        def check_no_placeholders(obj: Any):
            if isinstance(obj, str) and self._PLACEHOLDER_PATTERN.search(obj):
                raise ValueError(f"Unresolved placeholder in config: '{obj}'")
            if isinstance(obj, dict):
                for v in obj.values():
                    check_no_placeholders(v)
            if isinstance(obj, list):
                for v in obj:
                    check_no_placeholders(v)

        check_no_placeholders(self._cfg)

    def _process_paths(self) -> None:
        """
        Normalize and validate entries under 'paths'. Expand ~, make absolute,
        and create directories for keys ending in '_dir'.
        Raises:
            ValueError: If 'paths' section is missing or malformed.
        """
        paths = self._cfg.get("paths")
        if not isinstance(paths, dict):
            raise ValueError("Missing or invalid 'paths' section in config")
        # Expand user/relative and abspath
        for name, raw in paths.items():
            if not isinstance(raw, str):
                raise ValueError(f"Path '{name}' must be a string")
            expanded = os.path.abspath(os.path.expanduser(raw))
            paths[name] = expanded
        # Create directories for keys ending with '_dir'
        for name, p in paths.items():
            if name.endswith("_dir"):
                try:
                    os.makedirs(p, exist_ok=True)
                except Exception as e:
                    raise ValueError(f"Failed to create directory '{p}': {e}")

    def _validate_types(self) -> None:
        """
        Validate types and value ranges for known configuration fields.
        Raises:
            ValueError: If any field has an unexpected type or out-of-range value.
        """
        # embedding
        embed_cfg = self.get("embedding")
        if not isinstance(embed_cfg.get("batch_size"), int) or embed_cfg["batch_size"] <= 0:
            raise ValueError("embedding.batch_size must be a positive integer")
        if not isinstance(embed_cfg.get("normalize"), bool):
            raise ValueError("embedding.normalize must be a boolean")

        # index.hnsw
        h_cfg = self.get("index.hnsw")
        for key in ("M", "ef_construction", "ef_search"):
            if not isinstance(h_cfg.get(key), int) or h_cfg[key] <= 0:
                raise ValueError(f"index.hnsw.{key} must be a positive integer")

        # index.ivfpq
        ivf_cfg = self.get("index.ivfpq")
        for key in ("nlist", "M", "nprobe"):
            if not isinstance(ivf_cfg.get(key), int) or ivf_cfg[key] <= 0:
                raise ValueError(f"index.ivfpq.{key} must be a positive integer")

        # attack
        atk_h = self.get("attack.hnsw")
        eid = atk_h.get("edge_injection_density")
        if not isinstance(eid, (int, float)) or not (0.0 < eid < 1.0):
            raise ValueError("attack.hnsw.edge_injection_density must be between 0 and 1")

        atk_iv = self.get("attack.ivfpq")
        cs = atk_iv.get("centroid_shift")
        if not isinstance(cs, (int, float)) or not (0.0 < cs < 1.0):
            raise ValueError("attack.ivfpq.centroid_shift must be between 0 and 1")

        # evaluation
        eval_cfg = self.get("evaluation")
        if not isinstance(eval_cfg.get("top_k"), int) or eval_cfg["top_k"] <= 0:
            raise ValueError("evaluation.top_k must be a positive integer")
        if not isinstance(eval_cfg.get("num_trigger_queries_per_payload"), int) \
           or eval_cfg["num_trigger_queries_per_payload"] <= 0:
            raise ValueError("evaluation.num_trigger_queries_per_payload must be a positive integer")
        if not isinstance(eval_cfg.get("random_seed"), int) or eval_cfg["random_seed"] < 0:
            raise ValueError("evaluation.random_seed must be a non-negative integer")
