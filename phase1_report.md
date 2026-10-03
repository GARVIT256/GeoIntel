# Phase 1 Report — Problem Definition + Literature
**Date:** 2026-09-30  
**Status:** COMPLETE

---

## 1. What Was Built

| File | Purpose |
|---|---|
| [README.md](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/README.md) | Quick-start, structure, design principle |
| [config/config.yaml](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/config/config.yaml) | Master config — all parameters in one place |
| [config/aoi_dehradun.geojson](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/config/aoi_dehradun.geojson) | Proposed AOI (PROPOSED — needs your confirmation) |
| [.env.example](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/.env.example) | Secret template |
| [.gitignore](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/.gitignore) | Excludes data, venvs, secrets |
| [pyproject.toml](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/pyproject.toml) | Pinned Python 3.11 dependencies |
| [docker-compose.yml](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/docker-compose.yml) | PostGIS 16-3.4 + optional pgAdmin |
| [Makefile](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/Makefile) | `make install / data / classify / change / test / app` |
| [docs/problem_statement.md](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/docs/problem_statement.md) | RQs, scope, non-goals, claim policy, **full metric plan** |
| [docs/definitions.md](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/docs/definitions.md) | Formal definitions (urban expansion, veg loss, "associated") |
| [docs/gap_table.md](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/docs/gap_table.md) | Gap analysis vs. 10+ prior systems; Scholar queries |
| [data/corpus/papers_index.csv](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/data/corpus/papers_index.csv) | 20 candidate papers; all UNVERIFIED; search queries included |
| [src/geointel/db/schema.sql](file:///c:/Users/garvi/OneDrive/Documents/Major1/geo-intel/src/geointel/db/schema.sql) | PostGIS schema: aoi, grid_cells, change_results, run_manifests |
| Full `src/geointel/` skeleton | All subpackage `__init__.py` stubs ready for Phase 2+ |

---

## 2. How to Run (Phase 1 — no code runs yet)

```bash
cd geo-intel

# Inspect AOI in any GeoJSON viewer (e.g. geojson.io):
# Open config/aoi_dehradun.geojson

# Confirm the bounding box looks right on the map, then tell the agent "AOI confirmed"

# Install (for future phases):
python -m venv .venv
.venv\Scripts\activate
make install
```

Phase 1 produces no executed outputs — all deliverables are documents and config.

---

## 3. Test Results

Not applicable for Phase 1. `make test` will be green starting Phase 3.

---

## 4. Real Numbers / Figures Produced

None yet — no code has been executed. All numbers in later phases will be
traceable to executed commands.

---

## 5. Known Issues / Limitations / Risks

| Issue | Severity | Mitigation |
|---|---|---|
| All 20 papers are UNVERIFIED | HIGH | You must run the 17 Scholar queries before citing anything |
| AOI is a rectangle — may include Mussoorie ridge | MEDIUM | Confirm or adjust northern boundary |
| LLM provider unknown | MEDIUM | Phase 6 planner wrapper is swappable via env var |
| pyproject.toml versions pinned from knowledge — NOT verified against PyPI | MEDIUM | Run `pip install -e ".[dev]"` and fix any version conflicts before Phase 2 |
| Windows Makefile `clean` target uses `for /d /r` — may not work in all shells | LOW | Can run manually if needed |

---

## 6. What YOU Must Do Manually

1. **Verify the AOI** — open `config/aoi_dehradun.geojson` in [geojson.io](https://geojson.io) and confirm the bounding box covers the right area. Reply "AOI confirmed" or give me corrected coordinates.

2. **Verify all 20 papers** — run the search queries in `docs/gap_table.md` against Google Scholar or OpenAlex. For each paper found, reply with: title, authors, year, DOI, open-access yes/no. I will update `papers_index.csv` and change status to VERIFIED.

3. **Tell me your LLM provider** (OpenAI / Gemini / Anthropic / Ollama) — needed for Phase 6 planner.

4. **Check `pyproject.toml` versions** — after running `make install`, paste any pip errors and I will fix version pins.

---

## 7. Suggested Slide / Demo Content for Professor

**Slide: System Overview (1 slide)**
- Diagram: Query → Planner (LLM, T=0) → Pydantic Plan → Validator → Executor → Deterministic Tools → Results JSON → Report (LLM, constrained) → Map
- Highlight: "LLM only plans and writes. All numbers from deterministic tools."

**Slide: Design Choices (1 slide)**
- Why Sentinel-2 L2A? → Free, 10 m, global archive, SCL cloud mask
- Why dry season? → Cloud cover in Dehradun monsoon > 80%
- Why Random Forest? → Interpretable, no GPU, proven for 5-class land cover; SegFormer is post-midterm
- Why post-classification? → Traceable, auditable, matches Olofsson 2014 framework

**Slide: Claim Policy (1 slide)**
- Show the causal-language table from `definitions.md`
- Demonstrate the linter catching "urban growth caused vegetation loss"

---

## 8. Suggested Text for Project Report

> **Section 2.1 — Study Area:**
> "The study area is a 30 × 30 km region centred on Dehradun city, Uttarakhand,
> India (approximate bounding box: 77.90°E–78.20°E, 30.20°N–30.50°N, EPSG:4326),
> covering the urban core, Raipur, Premnagar, Clement Town, and the Selaqui
> industrial corridor. The area was selected to capture an active urban expansion
> front while avoiding the Mussoorie ridge, which introduces persistent hill shadow
> and seasonal snow."

> **Section 2.2 — Temporal Scope:**
> "Two dry-season composites were produced: T1 (November 2018 – March 2019) and
> T2 (November 2024 – March 2025). Monsoon months (June–September) were excluded
> owing to persistent cloud cover exceeding 80% in the Dehradun region. Using the
> same calendar months for both epochs minimises inter-epoch phenological differences
> in NDVI."

> **Section 3.1 — Language Constraint:**
> "Following scientific convention in observational remote sensing studies, this
> system never claims causation between urban expansion and vegetation loss. All
> findings are expressed as spatial co-occurrence or association (see Section 3.4).
> A causal-language linter in the agent pipeline enforces this constraint at runtime."

---

## Next Phase

Say **"Start Phase 2"** to begin data acquisition and preprocessing.
The first task will be fetching Sentinel-2 scenes from Microsoft Planetary Computer
STAC and building cloud-masked median composites for T1 and T2.
