# Phase 3 PostGIS query examples

The schema is initialized by `docker compose up -d postgis` from `src/geointel/db/schema.sql`. Queries below use inline synthetic fixtures so their expected outputs are reproducible without claiming project data results.

## 1. Area of projected geometries

```sql
WITH sample(name, geom) AS (
  VALUES
    ('square_100m', ST_GeomFromText('POLYGON((0 0,100 0,100 100,0 100,0 0))', 32644)),
    ('rectangle_50x100m', ST_GeomFromText('POLYGON((200 0,250 0,250 100,200 100,200 0))', 32644))
)
SELECT name, ST_Area(geom) AS area_m2 FROM sample ORDER BY name;
```

Expected: `rectangle_50x100m | 5000`; `square_100m | 10000`.

## 2. Count grid cells intersecting an AOI

```sql
WITH aoi AS (
  SELECT ST_GeomFromText('POLYGON((0 0,2000 0,2000 2000,0 2000,0 0))', 32644) AS geom
), cells AS (
  SELECT id, geom FROM (VALUES
    (1, ST_GeomFromText('POLYGON((0 0,1000 0,1000 1000,0 1000,0 0))',32644)),
    (2, ST_GeomFromText('POLYGON((1000 0,2000 0,2000 1000,1000 1000,1000 0))',32644)),
    (3, ST_GeomFromText('POLYGON((3000 0,4000 0,4000 1000,3000 1000,3000 0))',32644))
  ) AS v(id, geom)
)
SELECT count(*) AS intersecting_cells
FROM cells CROSS JOIN aoi WHERE ST_Intersects(cells.geom, aoi.geom);
```

Expected: `intersecting_cells | 2`.

## 3. Rank hotspot records

```sql
WITH grid_cells(cell_id, gi_star_z, urban_gain_ha, veg_loss_ha, is_hotspot) AS (
  VALUES (10, 2.4, 5.0, 3.0, TRUE), (11, 1.8, 2.0, 1.0, TRUE),
         (12, 0.7, 1.0, 0.5, FALSE)
)
SELECT cell_id, gi_star_z, urban_gain_ha, veg_loss_ha
FROM grid_cells WHERE is_hotspot
ORDER BY gi_star_z DESC LIMIT 2;
```

Expected rows in order: `(10, 2.4, 5.0, 3.0)`, `(11, 1.8, 2.0, 1.0)`.
