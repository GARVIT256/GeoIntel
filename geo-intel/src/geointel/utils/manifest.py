"""
utils/manifest.py — Run manifest writer for GEO-INTEL.

Every pipeline run writes a JSON manifest so results are reproducible and
traceable. The manifest records: plan, scene IDs, dates, CRS, parameters,
config hash, package versions, git commit, random seeds, timings.

No spatial operations; no CRS assumptions.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geointel.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    """Return short git commit hash, or 'unknown' if git not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _pkg_versions(packages: list[str]) -> dict[str, str]:
    """
    Return installed versions of the listed packages.
    Packages that are not installed are reported as 'not_installed'.
    """
    from importlib.metadata import version, PackageNotFoundError

    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


KEY_PACKAGES = [
    "rasterio", "rioxarray", "xarray", "geopandas", "shapely", "pyproj",
    "pystac-client", "stackstac", "odc-stac", "rasterstats",
    "scikit-learn", "numpy", "pandas", "sentence-transformers",
    "fastapi", "streamlit", "pydantic",
]


# ---------------------------------------------------------------------------
# Manifest class
# ---------------------------------------------------------------------------

class RunManifest:
    """
    Accumulates metadata during a pipeline run and serialises to JSON.

    Usage
    -----
    >>> manifest = RunManifest(cfg)
    >>> manifest.add_scene_ids("t1", ["S2A_...", "S2B_..."])
    >>> t = manifest.start_timer("composite_t1")
    >>> ...  # do work
    >>> manifest.stop_timer(t, "composite_t1")
    >>> manifest.set_result("urban_expansion_ha", 1234.5)
    >>> manifest.save()
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.run_id: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
        self._cfg = cfg
        self._data: dict[str, Any] = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config_hash": cfg.get("_config_hash", "unknown"),
            "pkg_versions": _pkg_versions(KEY_PACKAGES),
            "parameters": {
                "aoi_bbox": cfg.get("aoi", {}).get("bbox"),
                "epochs": cfg.get("epochs"),
                "max_cloud_cover": cfg.get("sentinel2", {}).get("max_cloud_cover"),
                "resolution_m": cfg.get("resolution", {}).get("reference_m"),
                "compute_crs": cfg.get("crs", {}).get("compute"),
                "random_seeds": cfg.get("reproducibility"),
            },
            "scene_ids": {},
            "valid_pixel_stats": {},
            "timings_sec": {},
            "results": {},
        }
        self._timers: dict[str, float] = {}

    # -- scene metadata --

    def add_scene_ids(self, epoch_key: str, ids: list[str]) -> None:
        """Record STAC scene IDs for an epoch ('t1' or 't2')."""
        self._data["scene_ids"][epoch_key] = ids
        logger.debug("Manifest: added %d scene IDs for %s", len(ids), epoch_key)

    def add_scene_dates(self, epoch_key: str, dates: list[str]) -> None:
        """Record acquisition dates for an epoch."""
        self._data["scene_ids"].setdefault(epoch_key + "_dates", dates)

    def add_valid_pixel_stat(self, epoch_key: str, stat: dict[str, Any]) -> None:
        """
        Record valid-pixel statistics for an epoch.
        stat: {mean_frac, min_frac, max_frac, n_scenes, warning}
        """
        self._data["valid_pixel_stats"][epoch_key] = stat

    # -- timers --

    def start_timer(self, label: str) -> float:
        """Start a named timer; returns the start time."""
        t = time.perf_counter()
        self._timers[label] = t
        logger.debug("Timer started: %s", label)
        return t

    def stop_timer(self, label: str) -> float:
        """Stop a named timer and record elapsed seconds."""
        if label not in self._timers:
            logger.warning("Timer %r was not started; skipping.", label)
            return 0.0
        elapsed = time.perf_counter() - self._timers.pop(label)
        self._data["timings_sec"][label] = round(elapsed, 3)
        logger.debug("Timer stopped: %s = %.1f s", label, elapsed)
        return elapsed

    # -- results --

    def set_result(self, key: str, value: Any) -> None:
        """Store a key-value result (any JSON-serialisable type)."""
        self._data["results"][key] = value

    # -- serialisation --

    def to_dict(self) -> dict[str, Any]:
        """Return a copy of the manifest data as a plain dict."""
        return dict(self._data)

    def save(self, output_dir: Path | None = None) -> Path:
        """
        Write the manifest to ``output_dir/<run_id>.json``.

        Parameters
        ----------
        output_dir : Path, optional
            Defaults to ``data/manifests/`` resolved from config.

        Returns
        -------
        Path
            Path of the written manifest file.
        """
        if output_dir is None:
            repo_root = self._cfg.get("_repo_root", Path("."))
            output_dir = repo_root / "data" / "manifests"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"{self.run_id}.json"
        out_path.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("Manifest saved: %s", out_path)
        return out_path
