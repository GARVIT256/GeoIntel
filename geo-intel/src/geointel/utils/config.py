"""
utils/config.py — Config loader for GEO-INTEL.

Loads config/config.yaml, resolves paths relative to repo root, and returns
a validated dict. Every module should call load_config() once at startup.

No spatial operations here; no CRS assumptions.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------

def _find_repo_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` (default: this file's location) until we find a
    directory containing 'config/config.yaml'. Raises FileNotFoundError if
    not found within 10 levels.
    """
    candidate = (start or Path(__file__).resolve()).parent
    for _ in range(10):
        if (candidate / "config" / "config.yaml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    raise FileNotFoundError(
        "Could not locate repo root (config/config.yaml not found). "
        "Run from within the geo-intel directory."
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and return the master configuration dictionary.

    Parameters
    ----------
    config_path : str or Path, optional
        Explicit path to config.yaml. If None, auto-discovers via repo root.

    Returns
    -------
    dict
        Parsed YAML config with an extra key ``_repo_root`` (Path) and
        ``_config_hash`` (str, sha256 of the raw YAML bytes).

    Notes
    -----
    - All ``paths.*`` values are resolved to absolute Paths.
    - Environment variables prefixed with ``GEOINTEL_`` override config values
      at the top level (e.g. ``GEOINTEL_LOG_LEVEL=DEBUG``).
    """
    if config_path is None:
        repo_root = _find_repo_root()
        config_path = repo_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path).resolve()
        repo_root = config_path.parent.parent  # config/ is one level below root

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = config_path.read_bytes()
    cfg: dict[str, Any] = yaml.safe_load(raw)

    cfg["_repo_root"] = repo_root
    cfg["_config_hash"] = hashlib.sha256(raw).hexdigest()[:12]

    # Resolve path strings to absolute Paths
    paths_block = cfg.get("paths", {})
    for key, rel in paths_block.items():
        abs_path = (repo_root / rel).resolve()
        abs_path.mkdir(parents=True, exist_ok=True)
        paths_block[key] = abs_path
    cfg["paths"] = paths_block

    # Env-var overrides (GEOINTEL_<KEY>=value overrides cfg[key])
    for env_key, env_val in os.environ.items():
        if env_key.startswith("GEOINTEL_"):
            cfg_key = env_key[len("GEOINTEL_"):].lower()
            cfg[cfg_key] = env_val

    return cfg


def get_epoch(cfg: dict[str, Any], epoch_key: str) -> dict[str, str]:
    """
    Return the epoch sub-dict for ``epoch_key`` ('t1' or 't2').

    Returns
    -------
    dict with keys: label, start, end
    """
    if epoch_key not in ("t1", "t2"):
        raise ValueError(f"epoch_key must be 't1' or 't2', got {epoch_key!r}")
    return cfg["epochs"][epoch_key]


def config_hash(cfg: dict[str, Any]) -> str:
    """Return the sha256 hash of the raw config file (first 12 hex chars)."""
    return cfg["_config_hash"]
