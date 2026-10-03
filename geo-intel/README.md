# GEO-INTEL
**An Agentic AI Framework for Automated Geospatial Research Using Remote Sensing and GIS**

> **Midterm Prototype** | Study area: Dehradun, India | Sensor: Sentinel-2 L2A | Epochs: 2018-19 and 2024-25

---

## Quick Start

```bash
# 1. Clone and enter
git clone <repo_url> && cd geo-intel

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
make install

# 4. Configure secrets
make env                     # copies .env.example to .env
# then edit .env and add your API keys

# 5. Start database
docker compose up -d postgis

# 6. Fetch and preprocess data (uses cache if already downloaded)
make data

# 7. Run tests
make test

# 8. Launch Streamlit demo (offline from cache)
make app
```

---

## Project Structure

```
geo-intel/
  config/
    config.yaml               # Single source of truth for all parameters
    aoi_dehradun.geojson      # Study area polygon (EPSG:4326)
  data/
    raw/                      # Downloaded but unprocessed (gitignored)
    interim/                  # Intermediate products (gitignored)
    processed/                # Final maps and masks (gitignored)
    cache/                    # COG cache for offline demo (gitignored)
    labels/                   # Ground-truth labels (gitignored)
    corpus/                   # PDFs and papers_index.csv
  docs/
    problem_statement.md
    definitions.md
    gap_table.md
    data_report.md            # Phase 2
    architecture.md           # Phase 7
    limitations.md            # Phase 7
    demo_script.md            # Phase 7
  notebooks/                  # Exploration only; no core logic here
  src/geointel/
    data/                     # STAC fetch, cloud masking, compositing
    gis/                      # CRS, raster ops, vector ops, grid, stats
    rs/                       # Spectral indices, features, RF classifier
    change/                   # Transition matrix, NDVI diff, association
    rag/                      # Ingest and retrieve from vector store
    agent/                    # Schema, planner, executor, validator, report
    db/                       # schema.sql, data loader
    app/                      # FastAPI, Streamlit
    utils/                    # Logging, manifest writer
  tests/                      # pytest suite with known-answer tests
  benchmark/                  # Query evaluation scripts
  docker-compose.yml
  Makefile
  pyproject.toml
```

---

## Core Design Principle

> **The LLM only PLANS and WRITES. All numbers come from deterministic tools.**

1. User submits a natural-language query.
2. `agent/planner.py` calls the LLM with temperature=0, receives a validated
   Pydantic JSON plan.
3. `agent/validator.py` checks: place in AOI, years available, season allowed,
   CRS consistency, causal-language absence.
4. `agent/executor.py` runs tool functions in order, writing a trace log.
5. `agent/report.py` fills a template with numbers from the result JSON.
   A number-provenance check confirms every number in the text is in the JSON.
6. `rag/retrieve.py` fetches supporting literature chunks; chunk IDs are cited.

---

## Claim Policy

Never say "caused", "led to", "resulted in", "will flood", or any causal framing.
Always say "co-occurs with", "spatially overlaps with", "is associated with".
The `agent/validator.py` causal-language linter enforces this automatically.

---

## Reproducibility

Every run writes a manifest to `data/manifests/<run_id>.json` containing:
scene IDs, dates, CRS, parameters, config hash, package versions, git commit, seeds, timings.

---

## Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1 | DONE | Problem statement, definitions, gap table, papers index |
| 2 | PENDING | Data acquisition, composites, data report |
| 3 | PENDING | GIS pipeline, PostGIS, pytest suite |
| 4 | PENDING | Indices, labelling, RF classifier, accuracy |
| 5 | PENDING | Change detection, association analysis |
| 6 | PENDING | Agent workflow end-to-end |
| 7 | PENDING | Streamlit demo, architecture, demo script |

---

## Requirements

- Python 3.11
- Docker + Docker Compose (for PostGIS)
- GNU Make (or run commands manually from Makefile)
- 8 GB RAM minimum; 16 GB recommended for compositing
- ~20 GB disk for data cache

---

## Licence

MIT — see LICENSE file (to be added).
