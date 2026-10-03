# GEO-INTEL — Problem Statement
**Version:** 0.1 (Midterm Prototype)
**Study Area:** Dehradun, Uttarakhand, India
**Date:** September 2026

---

## 1. Problem

Rapid urbanisation in mid-Himalayan foothill cities like Dehradun is transforming
land cover at scales that are difficult to track manually. Planners, researchers, and
policymakers need timely, spatially explicit evidence of where urban expansion
co-occurs with vegetation loss so they can prioritise ecological interventions and
enforce land-use regulations.

Current approaches require expert GIS knowledge and days of manual processing.
Large Language Models (LLMs) offer the ability to accept natural-language questions
and produce structured analytical plans, but they cannot reliably compute geospatial
quantities and are prone to hallucination of numbers and causal claims.

**The core problem this project addresses:**
> *How can we build a system that accepts a natural-language environmental question
> and produces a reproducible, traceable, quantitative geospatial answer — while
> keeping all numerical computation in deterministic tools and all narrative in an
> LLM that is constrained by those numbers?*

---

## 2. Research Questions (Midterm Scope)

1. **RQ-1 (Change mapping):** Which areas within the Dehradun study region showed
   built-up gain between the dry seasons of 2018–19 and 2024–25, and how much area
   (in hectares) was affected?

2. **RQ-2 (Vegetation loss):** Which areas showed a transition from vegetation
   (tree cover or cropland/grass) to non-vegetation during the same period?

3. **RQ-3 (Spatial association):** Do urban-gain pixels and vegetation-loss pixels
   spatially co-occur at the 1 km grid-cell scale, and is this co-occurrence
   statistically significant (Spearman rho, Moran's I, Getis-Ord Gi*)?

4. **RQ-4 (Agentic workflow):** Can an LLM planner reliably convert natural-language
   queries into validated, parameterised workflow plans that a deterministic executor
   can run without human intervention?

> **Scope note:** These questions are limited to the Dehradun AOI, two epochs, one
> 10 m satellite sensor, and one query type. Multi-city, multi-sensor, or causal
> inference is **out of scope** before the final submission.

---

## 3. Scope

| In scope (midterm) | Out of scope before midterm |
|---|---|
| Dehradun 30 x 30 km AOI | Other cities or regions |
| Sentinel-2 L2A, 10 m, two dry-season composites | SAR, LiDAR, Planet, Landsat |
| 5-class Random Forest land-cover map | SegFormer / deep-learning models |
| Post-classification transition matrix + NDVI-diff baseline | Pixel-based direct change detection |
| 1 km grid Spearman / Moran / Gi* association | Causal inference, regression modelling |
| Single LLM plan-then-execute agent | Multi-agent frameworks, GraphRAG |
| FastAPI + Streamlit, offline-capable | React frontend, cloud deployment |
| FAISS / Chroma RAG with open-access PDFs | LLM fine-tuning, full RAG evaluation |
| PostGIS spatial database | Distributed compute, cloud SQL |

---

## 4. Non-Goals

The following are explicitly deferred to post-midterm phases:

- SegFormer or any deep-learning change-detection model
- Flood / road use case (Use Case 2)
- Multi-city scaling or cross-sensor comparison
- GraphRAG, agent memory, or multi-agent orchestration
- LLM fine-tuning
- Full Olofsson-style area-adjusted accuracy (requires ~400 reference labels;
  a placeholder is included so the infrastructure is ready)
- React frontend or cloud deployment

---

## 5. Claim Policy (mandatory — cite in every report and viva)

The system applies strict language constraints enforced at multiple levels:

| Prohibited language | Required replacement |
|---|---|
| "Urban growth caused vegetation loss" | "Urban gain co-occurs with vegetation loss in the same 1 km cell" |
| "Deforestation resulted from expansion" | "Vegetation loss spatially overlaps with built-up gain" |
| "The area will flood" | Out of scope; not produced |
| "Building-level accuracy" | "10 m pixel-level accuracy; sub-pixel features are not distinguishable" |
| Any number not in the results JSON | Blocked by number-provenance validator |

A causal-language linter in `agent/validator.py` scans every LLM-generated
narrative and raises an error if blacklisted phrases are detected.

---

## 6. Metric Plan

All metrics below are computed by deterministic code. Numbers are NOT filled in
here — they will be populated after each phase executes.

### 6.1 Land-cover classification accuracy

| Metric | Tool | Level | Target |
|---|---|---|---|
| Overall Accuracy (OA) | accuracy.py | Map-level | >= 0.80 |
| Per-class F1 | accuracy.py | Per class | Report all 5; built-up F1 is primary |
| Cohen's Kappa | accuracy.py | Map-level | >= 0.75 |
| Confusion matrix | accuracy.py | Full matrix | Always reported |
| Spatial block CV scores | classify_rf.py | CV fold | Within +/-5% of holdout OA |

> **Warning:** With < 150 labels per class the metrics will be noisy. The code
> prints a warning and confidence intervals when N < 150.

### 6.2 Change detection metrics

| Metric | Tool | Unit |
|---|---|---|
| Urban-expansion area | transition.py | hectares |
| Vegetation-loss area | transition.py | hectares |
| Overlap area (veg-to-built) | transition.py | hectares |
| NDVI-diff change area | ndvi_diff.py | hectares |
| Agreement with GHSL built-up | validate_external.py | % overlap |
| Agreement with Hansen tree loss | validate_external.py | % overlap |

### 6.3 Spatial association metrics

| Metric | Tool | Interpretation |
|---|---|---|
| Spearman rho (urban gain vs veg loss per 1 km cell) | association.py | Monotonic co-occurrence |
| Moran's I (urban gain) | association.py | Spatial clustering of gain |
| Getis-Ord Gi* (hotspots) | association.py | Significant hotspot cells (alpha=0.05) |
| Top-N hotspot cells | association.py | Cell IDs + coordinates |

### 6.4 Agent accuracy

| Metric | Tool | Definition |
|---|---|---|
| Parameter-extraction accuracy | benchmark/eval.py | Fraction of fields correctly parsed |
| Correct refusal rate | benchmark/eval.py | Fraction of OOS queries correctly refused |
| Plan validity rate | agent/validator.py | Fraction passing all validation checks |

### 6.5 RAG quality (qualitative for midterm)

- Citation rate: fraction of report claims citing a chunk ID
- Number-provenance pass rate: fraction of numbers in narrative traceable to results JSON
- Full retrieval precision/recall deferred to final submission
