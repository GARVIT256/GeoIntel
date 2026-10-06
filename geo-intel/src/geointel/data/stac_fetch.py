"""
data/stac_fetch.py — Sentinel-2 L2A STAC search for GEO-INTEL.

Searches Microsoft Planetary Computer or AWS Earth Search for Sentinel-2
L2A scenes covering the AOI and date windows defined in config.yaml.
Optionally applies a tile-wide cloud property filter when configured, records
same-tile/calendar-date deduplication drops, and caches retained item JSON.

CRS: STAC bbox is always in EPSG:4326 (WGS 84). No coordinate computation
is performed here; all spatial work is done in composite.py and gis/.

UNVERIFIED API notes (check if behaviour changes):
  - pystac_client 0.8.x: Client.open() + client.search() API assumed stable.
  - planetary_computer.sign_inplace(item) assumed correct; verify against
    https://pypi.org/project/planetary-computer/
  - odc-stac / stackstac loading API is in composite.py; not touched here.

Usage (CLI)
-----------
    python -m geointel.data.stac_fetch --config config/config.yaml --epoch t1
    python -m geointel.data.stac_fetch --config config/config.yaml --epoch t2
    python -m geointel.data.stac_fetch --config config/config.yaml  # both epochs

Outputs
-------
    data/cache/scenes/{epoch_key}/items.json   — all signed STAC items
    data/cache/scenes/{epoch_key}/metadata.csv — scene ID, date, cloud%
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import pandas as pd

from geointel.utils.config import get_epoch, load_config
from geointel.utils.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    import pystac
    import pystac_client

# ---------------------------------------------------------------------------
# STAC endpoint URLs (prefer Planetary Computer; fall back to AWS Earth Search)
# ---------------------------------------------------------------------------

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
AWS_STAC_URL = "https://earth-search.aws.element84.com/v1"

S2_COLLECTION_PC  = "sentinel-2-l2a"
S2_COLLECTION_AWS = "sentinel-2-l2a"


def _scene_cache_dir(cfg: dict[str, Any], provider: str, epoch_key: str) -> Path:
    return cfg["paths"]["data_cache"] / "scenes" / provider / epoch_key.lower()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _open_client(stac_url: str) -> pystac_client.Client:
    """
    Open a pystac_client.Client. Raises ImportError if pystac_client is
    not installed (caught by the CLI).
    """
    import pystac_client  # lazy import so unit tests don't need network

    return pystac_client.Client.open(stac_url)


def _sign_item(item: pystac.Item, provider: str) -> pystac.Item:
    """
    Sign a STAC item for authenticated access.

    For Planetary Computer, uses planetary_computer.sign_inplace().
    For AWS Earth Search, no signing is needed.

    UNVERIFIED: planetary_computer.sign_inplace() API — check the library docs
    if assets fail to load.
    """
    if provider == "pc":
        try:
            import planetary_computer  # type: ignore[import]
            planetary_computer.sign_inplace(item)
        except ImportError:
            logger.warning(
                "planetary-computer library not installed; items will be unsigned. "
                "Public assets may still be accessible."
            )
    return item


def deduplicate_scene_items(
    items: list[pystac.Item],
    dropped_items: list[dict[str, Any]] | None = None,
) -> list[pystac.Item]:
    """Keep the highest-baseline item for each MGRS tile and calendar date.

    Same-baseline same-day platform ties prefer S2A, then S2B, then S2C for a
    stable cross-epoch sensor choice. Every discarded item is logged and can
    be returned through ``dropped_items`` for the run manifest.
    """
    grouped: dict[tuple[str, str], list[pystac.Item]] = {}
    unkeyed: list[pystac.Item] = []

    def baseline_number(item: pystac.Item) -> float:
        baseline = item.properties.get("s2:processing_baseline", 0)
        try:
            return float(baseline)
        except (TypeError, ValueError):
            return 0.0

    def preference(item: pystac.Item) -> tuple[int, str]:
        platform = str(item.properties.get("platform", "")).casefold()
        platform_rank = {"sentinel-2a": 0, "sentinel-2b": 1, "sentinel-2c": 2}.get(platform, 3)
        return platform_rank, str(item.id)

    for item in items:
        props = item.properties
        acquired = props.get("datetime") or props.get("start_datetime")
        tile = _mgrs_tile(item)
        if not acquired or not tile:
            unkeyed.append(item)
            continue
        calendar_date = str(acquired).split("T", 1)[0]
        grouped.setdefault((str(tile), calendar_date), []).append(item)

    selected_items: list[pystac.Item] = []
    for (tile, calendar_date), candidates in grouped.items():
        chosen = min(candidates, key=lambda item: (-baseline_number(item), *preference(item)))
        selected_items.append(chosen)
        for item in candidates:
            if item is chosen:
                continue
            same_baseline = baseline_number(item) == baseline_number(chosen)
            reason = (
                "same-baseline platform/item tie-break"
                if same_baseline else "lower processing baseline"
            )
            record = {
                "dropped_scene_id": str(item.id),
                "kept_scene_id": str(chosen.id),
                "tile": tile,
                "calendar_date": calendar_date,
                "dropped_platform": item.properties.get("platform"),
                "kept_platform": chosen.properties.get("platform"),
                "dropped_processing_baseline": item.properties.get("s2:processing_baseline"),
                "kept_processing_baseline": chosen.properties.get("s2:processing_baseline"),
                "reason": reason,
                "dropped_item": item.to_dict() if hasattr(item, "to_dict") else None,
            }
            if dropped_items is not None:
                dropped_items.append(record)
            logger.info(
                "Dropped duplicate STAC item %s; kept %s for tile=%s date=%s (%s)",
                item.id, chosen.id, tile, calendar_date, reason,
            )

    unique_items = selected_items + unkeyed
    return sorted(
        unique_items,
        key=lambda item: (
            item.properties.get("datetime") or item.properties.get("start_datetime") or "",
            item.properties.get("platform", ""),
            item.properties.get("s2:mgrs_tile", ""),
        ),
    )


def _mgrs_tile(item: pystac.Item) -> str | None:
    """Return the MGRS tile from STAC properties or the Earth Search item ID."""
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        return str(tile)
    match = re.search(r"\d{2}[A-Z]{3}", str(item.id))
    return match.group(0) if match else None


def search_scenes(
    cfg: dict[str, Any],
    epoch_key: str,
    provider: str = "pc",
    dropped_items: list[dict[str, Any]] | None = None,
) -> list[pystac.Item]:
    """
    Search STAC for Sentinel-2 L2A scenes covering the AOI and epoch window.

    Parameters
    ----------
    cfg : dict
        Loaded config (from load_config()).
    epoch_key : str
        'T1' or 't1' / 'T2' or 't2'.
    provider : str
        'pc' (Planetary Computer) or 'aws' (Earth Search).

    Returns
    -------
    list of pystac.Item
        Deduplicated items, optionally filtered by configured tile-wide cloud
        cover, sorted by date ascending. The default configuration disables
        that coarse filter and masks SCL pixels during AOI compositing.
        Empty list if no scenes found (logs a warning).

    CRS
    ---
    STAC bbox is EPSG:4326 [west, south, east, north].
    """
    epoch_key = epoch_key.lower()
    epoch = get_epoch(cfg, epoch_key)
    bbox: list[float] = cfg["aoi"]["bbox"]         # [west, south, east, north]
    max_cloud = cfg["sentinel2"].get("max_cloud_cover")
    collection = S2_COLLECTION_PC if provider == "pc" else S2_COLLECTION_AWS
    stac_url   = PC_STAC_URL if provider == "pc" else AWS_STAC_URL

    logger.info(
        "Searching STAC (%s) | epoch=%s | %s to %s | tile_cloud_cutoff=%s",
        provider, epoch_key, epoch["start"], epoch["end"],
        f"{max_cloud}%" if max_cloud is not None else "disabled; AOI SCL masking",
    )

    client = _open_client(stac_url)

    # pystac_client search — UNVERIFIED: query dict key for cloud cover
    # Planetary Computer uses "eo:cloud_cover"; AWS Earth Search also supports it.
    search_kwargs: dict[str, Any] = {}
    if max_cloud is not None:
        search_kwargs["query"] = {"eo:cloud_cover": {"lte": float(max_cloud)}}
    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{epoch['start']}/{epoch['end']}",
        max_items=500,        # safety cap; typical dry-season count is 50–200
        **search_kwargs,
    )

    items = deduplicate_scene_items(list(search.items()), dropped_items=dropped_items)

    if not items:
        logger.warning("No scenes found for epoch=%s", epoch_key)
        return []

    logger.info("Found %d scenes for epoch=%s", len(items), epoch_key)

    # Sign all items
    signed = [_sign_item(item, provider) for item in items]
    return signed


def scene_metadata_df(items: list[pystac.Item]) -> pd.DataFrame:
    """
    Build a metadata DataFrame from STAC items.

    Returns
    -------
    pd.DataFrame with columns:
        scene_id, datetime, cloud_cover_pct, platform, mgrs_tile,
        relative_orbit, s2_processing_baseline
    """
    rows = []
    for item in items:
        props = item.properties
        rows.append({
            "scene_id":               item.id,
            "datetime":               props.get("datetime") or props.get("start_datetime"),
            "cloud_cover_pct":        props.get("eo:cloud_cover", float("nan")),
            "platform":               props.get("platform", "unknown"),
            "mgrs_tile": _mgrs_tile(item) or "unknown",
            "relative_orbit":         props.get("sat:relative_orbit", "unknown"),
            "processing_baseline":    props.get("s2:processing_baseline", "unknown"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
    return df


def cache_scenes(
    items: list[pystac.Item],
    cfg: dict[str, Any],
    epoch_key: str,
    provider: str = "pc",
) -> tuple[Path, Path]:
    """
    Serialise STAC items and metadata CSV to the cache directory.

    Parameters
    ----------
    items : list of pystac.Item
    cfg : dict
    epoch_key : str ('t1' or 't2')

    Returns
    -------
    (items_json_path, metadata_csv_path)

    Output paths
    ------------
    data/cache/scenes/{provider}/{epoch_key}/items.json
    data/cache/scenes/{provider}/{epoch_key}/metadata.csv
    """
    epoch_key = epoch_key.lower()
    cache_dir = _scene_cache_dir(cfg, provider, epoch_key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Serialize items to JSON (pystac Item.to_dict())
    items_list = [item.to_dict() for item in items]
    items_path = cache_dir / "items.json"
    items_path.write_text(json.dumps(items_list, indent=2), encoding="utf-8")
    logger.info("Cached %d scene items → %s", len(items), items_path)

    # Metadata CSV
    df = scene_metadata_df(items)
    meta_path = cache_dir / "metadata.csv"
    df.to_csv(meta_path, index=False)
    logger.info("Metadata CSV → %s", meta_path)

    return items_path, meta_path


def load_cached_items(
    cfg: dict[str, Any],
    epoch_key: str,
    provider: str = "pc",
    dropped_items: list[dict[str, Any]] | None = None,
) -> list[pystac.Item] | None:
    """
    Load STAC items from the cache if available.

    Returns
    -------
    list of pystac.Item, or None if cache does not exist.
    """
    import pystac  # type: ignore[import]

    epoch_key = epoch_key.lower()
    items_path = _scene_cache_dir(cfg, provider, epoch_key) / "items.json"

    if not items_path.exists():
        return None

    logger.info("Loading scenes from cache: %s", items_path)
    raw = json.loads(items_path.read_text(encoding="utf-8"))
    items = deduplicate_scene_items(
        [pystac.Item.from_dict(d) for d in raw], dropped_items=dropped_items
    )
    logger.info("Loaded %d cached items for epoch=%s", len(items), epoch_key)
    return items


def fetch_or_load(
    cfg: dict[str, Any],
    epoch_key: str,
    provider: str = "pc",
    force_refresh: bool = False,
    dropped_items: list[dict[str, Any]] | None = None,
) -> list[pystac.Item]:
    """
    Cache-first STAC fetch. Loads from cache if available and ``force_refresh``
    is False; otherwise searches STAC and writes cache.

    Parameters
    ----------
    cfg : dict
    epoch_key : str ('t1' or 't2')
    provider : str ('pc' or 'aws')
    force_refresh : bool
        If True, always re-query STAC even if cache exists.

    Returns
    -------
    list of pystac.Item
    """
    if not force_refresh:
        cached = load_cached_items(
            cfg, epoch_key, provider=provider, dropped_items=dropped_items
        )
        if cached is not None:
            return [_sign_item(item, provider) for item in cached]

    items = search_scenes(
        cfg, epoch_key, provider=provider, dropped_items=dropped_items
    )
    if items:
        cache_scenes(items, cfg, epoch_key, provider=provider)
    return items


def print_summary(cfg: dict[str, Any], epoch_key: str, provider: str = "pc") -> None:
    """Print a human-readable scene summary to stdout."""
    meta_path = _scene_cache_dir(cfg, provider, epoch_key) / "metadata.csv"
    if not meta_path.exists():
        print(f"  No metadata found for epoch={epoch_key}. Run fetch first.")
        return

    df = pd.read_csv(meta_path)
    print(f"\n  Epoch {epoch_key.upper()}: {len(df)} scenes")
    print(f"  Date range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Cloud cover: min={df['cloud_cover_pct'].min():.1f}%  "
          f"max={df['cloud_cover_pct'].max():.1f}%  "
          f"mean={df['cloud_cover_pct'].mean():.1f}%")
    print(f"  Platforms: {df['platform'].unique().tolist()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--config", "config_path", default="config/config.yaml",
              show_default=True, help="Path to config.yaml")
@click.option("--epoch", "epoch_key", default="both",
              type=click.Choice(["t1", "t2", "both"]),
              show_default=True, help="Which epoch to fetch")
@click.option("--provider", default="pc",
              type=click.Choice(["pc", "aws"]),
              show_default=True, help="STAC provider")
@click.option("--force-refresh", is_flag=True, default=False,
              help="Re-query STAC even if cache exists")
def main(
    config_path: str,
    epoch_key: str,
    provider: str,
    force_refresh: bool,
) -> None:
    """
    Search Sentinel-2 L2A STAC and cache scene metadata.

    Outputs data/cache/scenes/{t1,t2}/items.json and metadata.csv.
    """
    cfg = load_config(config_path)
    epochs = ["t1", "t2"] if epoch_key == "both" else [epoch_key]

    for ek in epochs:
        items = fetch_or_load(cfg, ek, provider=provider, force_refresh=force_refresh)
        if not items:
            logger.error("No items found for epoch=%s. Check config and network.", ek)
            continue
        print_summary(cfg, ek, provider=provider)

    logger.info("stac_fetch complete.")


if __name__ == "__main__":
    main()
