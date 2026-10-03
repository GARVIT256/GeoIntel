"""Run documented synthetic PostGIS queries through Docker Compose and assert outputs."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


QUERIES = [
    (
        "projected geometry areas",
        """WITH sample(name, geom) AS (
          VALUES
            ('square_100m', ST_GeomFromText('POLYGON((0 0,100 0,100 100,0 100,0 0))', 32644)),
            ('rectangle_50x100m', ST_GeomFromText('POLYGON((200 0,250 0,250 100,200 100,200 0))', 32644))
        )
        SELECT name, ST_Area(geom)::bigint AS area_m2 FROM sample ORDER BY name;""",
        ["rectangle_50x100m|5000", "square_100m|10000"],
    ),
    (
        "AOI/grid intersections",
        """WITH aoi AS (
          SELECT ST_GeomFromText('POLYGON((0 0,2000 0,2000 2000,0 2000,0 0))', 32644) AS geom
        ), cells AS (
          SELECT id, geom FROM (VALUES
            (1, ST_GeomFromText('POLYGON((0 0,1000 0,1000 1000,0 1000,0 0))',32644)),
            (2, ST_GeomFromText('POLYGON((1000 0,2000 0,2000 1000,1000 1000,1000 0))',32644)),
            (3, ST_GeomFromText('POLYGON((3000 0,4000 0,4000 1000,3000 1000,3000 0))',32644))
          ) AS v(id, geom)
        )
        SELECT count(*) FROM cells CROSS JOIN aoi WHERE ST_Intersects(cells.geom, aoi.geom);""",
        ["2"],
    ),
    (
        "hotspot ranking",
        """WITH grid_cells(cell_id, gi_star_z, urban_gain_ha, veg_loss_ha, is_hotspot) AS (
          VALUES (10, 2.4, 5.0, 3.0, TRUE), (11, 1.8, 2.0, 1.0, TRUE),
                 (12, 0.7, 1.0, 0.5, FALSE)
        )
        SELECT cell_id, round(gi_star_z::numeric,1), urban_gain_ha::numeric,
               veg_loss_ha::numeric
        FROM grid_cells WHERE is_hotspot ORDER BY gi_star_z DESC LIMIT 2;""",
        ["10|2.4|5.0|3.0", "11|1.8|2.0|1.0"],
    ),
]


def compose_psql(repo: Path, sql: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker", "compose", "exec", "-T", "postgis", "sh", "-c",
            'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -At -F "|" -c "$1"',
            "postgis-check", sql,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def execute(repo: Path) -> int:
    print("PostGIS integration check: starting the Docker Compose postgis service")
    subprocess.run(["docker", "compose", "up", "-d", "postgis"], cwd=repo, check=True)

    deadline = time.monotonic() + 120
    while True:
        ready = compose_psql(repo, "SELECT postgis_version();")
        if ready.returncode == 0:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f"PostGIS did not become ready: {ready.stderr.strip()}")
        time.sleep(3)

    tables = compose_psql(
        repo,
        "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' "
        "AND table_name IN ('aoi','grid_cells','change_results','run_manifests');",
    )
    if tables.returncode or tables.stdout.strip() != "4":
        raise RuntimeError(
            "Expected all four GEO-INTEL tables (aoi, grid_cells, change_results, "
            f"run_manifests); query returned {tables.stdout.strip()!r}. "
            "A persistent database may need the schema applied manually."
        )

    for name, sql, expected in QUERIES:
        result = compose_psql(repo, sql)
        actual = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or actual != expected:
            raise AssertionError(
                f"{name}: expected {expected!r}, got {actual!r}; "
                f"psql stderr={result.stderr.strip()!r}"
            )
        print(f"PASS {name}: {actual}")
    print("PASS PostGIS schema and all documented synthetic query outputs match.")
    return 0


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    try:
        sys.exit(execute(project_root))
    except (OSError, subprocess.CalledProcessError, RuntimeError, AssertionError) as exc:
        print(f"FAIL PostGIS check: {exc}", file=sys.stderr)
        sys.exit(1)
