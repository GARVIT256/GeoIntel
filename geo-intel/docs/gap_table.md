# GEO-INTEL — Gap Analysis
**Version:** 0.1 (Midterm Prototype)
**Status:** UNVERIFIED entries marked. Run search queries before citing in report.

> **Honest disclaimer:** All paper entries below are based on the author's
> knowledge. Every entry is marked UNVERIFIED until YOU confirm the DOI,
> title, and year against Google Scholar or OpenAlex. Do NOT cite an UNVERIFIED
> entry in your project report. Search queries are provided for each group.

---

## 1. Gap Analysis Table

### 1.1 LLM / Agentic GIS Systems (most directly related)

| System | What it does | Key gap vs. GEO-INTEL |
|---|---|---|
| **LLM-Geo / Autonomous GIS** | LLM decomposes GIS tasks into Python code; executes autonomously | No deterministic tool registry; LLM can compute numbers directly, risking hallucination. No RAG grounding. (UNVERIFIED — see query LLM-GEO-1) |
| **GeoGPT** | Conversational GIS assistant using ChatGPT + GIS tools | Focus is on interactive Q&A and map generation, not reproducible remote-sensing workflows. No systematic change detection. (UNVERIFIED — see query GEOGPT-1) |
| **Change-Agent** | LLM agent for remote-sensing change detection | Architecture not publicly released at time of writing; does not appear to have validated plan schema or causal-language guard. (UNVERIFIED — see query CA-1) |
| **GeoLLM-Engine** | Geospatial foundation model for instruction-following spatial tasks | Focuses on spatial reasoning QA, not operational change-detection pipelines with quantified accuracy. (UNVERIFIED — see query GLLE-1) |
| **RS-Agent / RS-style systems** | Tool-using agents for remote-sensing tasks | Typically demonstrated on classification or object detection; change detection across years with post-processing and association analysis is not demonstrated. (UNVERIFIED — see query RSA-1) |
| **GeoAgentic-RAG** | RAG-augmented geospatial agent | Concept paper; no operational pipeline, no deterministic executor separation. (UNVERIFIED — see query GAR-1) |
| **GEO-INTEL (this work)** | Plan-then-execute with validated JSON plan, deterministic executor, causal-language guard, RAG grounding, offline-capable, reproducible | Novel integration of: plan validation schema, causal-language linter, number-provenance check, spatial association (not just change maps), offline demo |

> **What GEO-INTEL does NOT claim:**
> - "First" agentic GIS system (LLM-Geo predates it)
> - "Best" accuracy (limited label set at midterm)
> - "Novel" change detection (uses standard RF + transition matrix)
> The claimed contribution is the **validated, safe, reproducible integration** of these components with explicit hallucination guards.

---

### 1.2 Remote-Sensing / Change Detection Methods

| Method / Paper | Approach | Gap vs. this work |
|---|---|---|
| **Shafique et al. 2022 change detection review** | Survey of DL-based change detection methods (Siamese CNNs, transformers, etc.) | Focus is on pixel-pair deep models; does not address agentic workflows, LLM planning, or reproducibility. We use their taxonomy to justify RF as a simpler, interpretable baseline. (UNVERIFIED — see query SHAFIQUE-1) |
| **SegFormer (Xie et al. 2021)** | Transformer-based semantic segmentation | We use it as a POST-MIDTERM aspirational model, not a competitor. At 10 m Sentinel-2 resolution its advantage over RF is not demonstrated for land-cover mapping. (UNVERIFIED — see query SEGFORMER-1) |
| **GeoChat (Kuckreja et al. 2023?)** | Vision-language model for remote sensing | Generates descriptions of satellite images; does not perform quantitative change detection or multi-epoch analysis. No deterministic computation. (UNVERIFIED — see query GEOCHAT-1) |
| **Dynamic World (Brown et al. 2022)** | Near-real-time 10 m land cover via deep learning on Sentinel-2 | Global product; not tuned for Indian urban fringe or Himalayan foothills. Not reproducible from a local pipeline without access to the model. We compare against it as an external reference. (UNVERIFIED — see query DW-1) |
| **ESA WorldCover 2020/2021** | 10 m global land cover from Sentinel-1 + Sentinel-2 | Single epoch (no change detection). Used as external validation reference in Phase 5. (UNVERIFIED — see query WORLDCOVER-1) |
| **GHSL (Florczyk et al. 2019 / Pesaresi et al.)** | Global Human Settlement Layer — built-up density, population | 100 m resolution in older releases; useful for validation but coarser than our 10 m map. (UNVERIFIED — see query GHSL-1) |
| **Hansen et al. 2013 (Global Forest Watch)** | Annual 30 m forest cover loss from Landsat | Landsat (30 m) may miss small patches; tree gain/loss definition differs from our 5-class scheme. Used as external validation for vegetation loss. (UNVERIFIED — see query HANSEN-1) |
| **Olofsson et al. 2014** | Good practices for area estimation and accuracy assessment | Methodological standard we adopt. We implement stratified sampling, confidence intervals, and the area-adjusted estimator. (UNVERIFIED — see query OLOFSSON-1) |
| **NDBI (Zha et al. 2003)** | Normalised Difference Built-up Index | Original definition; we implement exactly as specified. (UNVERIFIED — see query ZHA-1) |

---

### 1.3 Urban Expansion Studies — Dehradun / Indian Cities

| Study | What it found | Gap vs. this work |
|---|---|---|
| Studies on Dehradun urban growth (general) | Rapid expansion post-2000; Selaqui industrial area; encroachment on forest land | Most use Landsat (30 m), 5–10 year intervals; no agentic workflow; no RAG. We use 10 m Sentinel-2. (UNVERIFIED — search query DEH-1) |
| Studies on Indian urban heat island / green loss | Correlation between urban expansion and vegetation loss at city scale | Typically at coarser resolution; correlation computed at city scale, not 1 km grid with Gi* hotspots. (UNVERIFIED — search query INDIA-1) |

---

## 2. Positioning Summary

```
GEO-INTEL fills the intersection of three areas:
  [Agentic GIS]  x  [Reproducible RS change detection]  x  [Hallucination guards]

No prior system (as of September 2026, to the best of our knowledge) implements
all three simultaneously with:
  - A validated Pydantic plan schema
  - A causal-language linter at inference time
  - A number-provenance check on LLM narratives
  - An offline-capable demo from cached COGs
```

---

## 3. Google Scholar / OpenAlex Search Queries

Copy these into https://scholar.google.com or https://openalex.org to find and
verify each paper. Report the DOI and open-access status back to the agent.

| Query ID | Search string | Target paper / topic |
|---|---|---|
| LLM-GEO-1 | `"LLM-Geo" OR "Autonomous GIS" large language model geospatial` | Li et al. or similar; agentic GIS with LLM |
| GEOGPT-1 | `"GeoGPT" conversational GIS ChatGPT geospatial` | GeoGPT system paper |
| CA-1 | `"Change-Agent" remote sensing change detection LLM agent` | Change-Agent paper |
| GLLE-1 | `"GeoLLM" OR "GeoLLM-Engine" geospatial instruction following` | GeoLLM-Engine paper |
| RSA-1 | `"RS-Agent" remote sensing tool-using agent large language model` | RS-Agent or similar |
| GAR-1 | `"GeoAgentic-RAG" OR "geospatial RAG agent" retrieval augmented generation` | GeoAgentic-RAG concept |
| SHAFIQUE-1 | `Shafique 2022 deep learning change detection review remote sensing` | Shafique et al. 2022 survey |
| SEGFORMER-1 | `SegFormer Xie 2021 semantic segmentation transformer` | Xie et al. 2021 NeurIPS |
| GEOCHAT-1 | `GeoChat vision language model remote sensing Kuckreja` | GeoChat paper |
| DW-1 | `"Dynamic World" near-real-time land cover Sentinel-2 Brown 2022 Nature Communications` | Brown et al. 2022 |
| WORLDCOVER-1 | `"ESA WorldCover" 10m global land cover 2020 Sentinel` | Zanaga et al. or ESA tech report |
| GHSL-1 | `"Global Human Settlement Layer" GHSL built-up Florczyk OR Pesaresi` | GHSL documentation paper |
| HANSEN-1 | `Hansen 2013 high resolution global forest cover change Science` | Hansen et al. 2013 Science |
| OLOFSSON-1 | `Olofsson 2014 good practices estimating area accuracy land change Remote Sensing Environment` | Olofsson et al. 2014 RSE |
| ZHA-1 | `Zha 2003 "use of normalized difference built-up index" Landsat` | Zha et al. 2003 IJRS |
| DEH-1 | `Dehradun urban expansion land use change Sentinel OR Landsat 2015 2020 2025` | Any Dehradun-specific study |
| INDIA-1 | `Indian city urban expansion vegetation loss NDVI 10m Sentinel-2` | Comparable Indian city studies |
