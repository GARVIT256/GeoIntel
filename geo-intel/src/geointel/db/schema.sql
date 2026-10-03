-- =============================================================
-- GEO-INTEL PostGIS Schema
-- Applied automatically on first docker compose up via
-- docker-entrypoint-initdb.d
-- =============================================================

-- Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;

-- ── Area of Interest ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS aoi (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    geom        GEOMETRY(POLYGON, 4326) NOT NULL,
    crs_compute TEXT NOT NULL DEFAULT 'EPSG:32644',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_aoi_geom ON aoi USING GIST (geom);

-- ── 1 km fishnet grid ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS grid_cells (
    cell_id         SERIAL PRIMARY KEY,
    row_idx         INT NOT NULL,
    col_idx         INT NOT NULL,
    geom_32644      GEOMETRY(POLYGON, 32644) NOT NULL,  -- compute CRS
    geom_4326       GEOMETRY(POLYGON, 4326),             -- display CRS
    -- Change statistics (populated by association.py)
    urban_gain_ha   DOUBLE PRECISION,
    veg_loss_ha     DOUBLE PRECISION,
    overlap_ha      DOUBLE PRECISION,
    valid_px_frac   DOUBLE PRECISION,
    -- Association statistics
    gi_star_z       DOUBLE PRECISION,
    gi_star_p       DOUBLE PRECISION,
    is_hotspot      BOOLEAN,
    run_id          TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_grid_geom_32644 ON grid_cells USING GIST (geom_32644);
CREATE INDEX IF NOT EXISTS idx_grid_geom_4326  ON grid_cells USING GIST (geom_4326);

-- ── Change results (pixel-aggregated) ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS change_results (
    id                  SERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL,
    epoch_t1            TEXT NOT NULL,    -- e.g. "2018-19"
    epoch_t2            TEXT NOT NULL,    -- e.g. "2024-25"
    urban_expansion_ha  DOUBLE PRECISION,
    veg_loss_ha         DOUBLE PRECISION,
    veg_to_built_ha     DOUBLE PRECISION,
    ndvi_diff_change_ha DOUBLE PRECISION,
    aoi_area_ha         DOUBLE PRECISION,
    ghsl_overlap_pct    DOUBLE PRECISION,
    hansen_overlap_pct  DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Run manifests ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS run_manifests (
    run_id          TEXT PRIMARY KEY,
    git_commit      TEXT,
    config_hash     TEXT,
    python_version  TEXT,
    pkg_versions    JSONB,
    parameters      JSONB,
    scene_ids       JSONB,
    timings_sec     JSONB,
    random_seeds    JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Example spatial queries (for documentation / Phase 3 demo) ──────────────
-- Q1: Count cells intersecting AOI
--   SELECT COUNT(*) FROM grid_cells g, aoi a
--   WHERE ST_Intersects(g.geom_4326, a.geom);
--
-- Q2: Area of each grid cell in m2 (computed CRS)
--   SELECT cell_id, ST_Area(geom_32644) AS area_m2 FROM grid_cells;
--
-- Q3: Top-20 hotspot cells by Gi* z-score
--   SELECT cell_id, gi_star_z, urban_gain_ha, veg_loss_ha
--   FROM grid_cells WHERE is_hotspot = TRUE
--   ORDER BY gi_star_z DESC LIMIT 20;
