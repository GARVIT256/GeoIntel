# GEO-INTEL — Formal Definitions
**Version:** 0.1 (Midterm Prototype)
**Status:** FIXED — do not change without updating config.yaml and rerunning the pipeline.

---

## 1. Study Area

**Name:** Dehradun 30 x 30 km AOI
**Bounding box (EPSG:4326):** West 77.90, South 30.20, East 78.20, North 30.50
**GeoJSON file:** `config/aoi_dehradun.geojson` (status: PROPOSED — awaiting user confirmation)
**Compute CRS:** EPSG:32644 (UTM Zone 44N). All area, length, and distance
computations are performed in this CRS. Results reported in m2 or hectares.
**Display CRS:** EPSG:4326 (WGS 84) for coordinates; EPSG:3857 for web maps.

> **Why EPSG:32644?** Dehradun lies in UTM Zone 44N. This is the standard
> projected CRS for this region, giving metric units with < 0.04% distortion
> across the 30 km AOI. Computing area or distance in degrees is physically
> meaningless and is **never done** in this project.

---

## 2. Epochs

| Label | Composite window | Rationale |
|---|---|---|
| T1 | 2018-11-01 to 2019-03-31 | Dry season; avoids monsoon cloud; earliest clean Sentinel-2 window |
| T2 | 2024-11-01 to 2025-03-31 | Most recent dry season; ~6-year gap is detectable at 10 m |

Both windows span the same calendar months (Nov–Mar) so seasonal NDVI
differences are minimised. Monsoon months (Jun–Sep) are excluded because
cloud cover in Dehradun routinely exceeds 80% during that period.

---

## 3. Imagery and Preprocessing

- **Sensor:** Sentinel-2 L2A (bottom-of-atmosphere reflectance, ESA Sen2Cor)
- **Catalogue:** Microsoft Planetary Computer STAC or AWS Earth Search STAC
- **Cloud/shadow masking:** Scene Classification Layer (SCL) — mask values
  0 (no data), 1 (saturated/defective), 2 (dark area), 3 (cloud shadow),
  8 (medium cloud), 9 (high cloud), 10 (thin cirrus), 11 (snow/ice)
- **Compositing:** Pixel-wise median over all valid observations in each window
- **Bands used:**

| Band | Wavelength | Native res | Use in this project |
|---|---|---|---|
| B02 (Blue) | ~490 nm | 10 m | Feature, MNDWI |
| B03 (Green) | ~560 nm | 10 m | Feature, MNDWI |
| B04 (Red) | ~665 nm | 10 m | Feature, NDVI |
| B08 (NIR) | ~842 nm | 10 m | Feature, NDVI, NDBI |
| B11 (SWIR-1) | ~1610 nm | 20 m | Feature, NDBI, MNDWI, BSI — resampled to 10 m (bilinear) |
| B12 (SWIR-2) | ~2190 nm | 20 m | Feature, BSI — resampled to 10 m (bilinear) |

- **Reference grid:** 10 m, EPSG:32644, aligned to Sentinel-2 grid
- **Categorical resampling:** nearest-neighbour (class maps, SCL)
- **Output format:** Cloud-Optimized GeoTIFF (COG) in `data/cache/`

---

## 4. Land-Cover Class Scheme

5 classes; codes are fixed integers used everywhere in the pipeline.

| Code | Name | Description | Key spectral cues |
|---|---|---|---|
| 1 | built-up | Impervious surfaces: buildings, roads, concrete, paved areas | High NDBI, low NDVI, low MNDWI |
| 2 | tree/dense-vegetation | Dense tree canopy, forest, orchards | High NDVI (> ~0.5), high NIR |
| 3 | cropland/grass/low-veg | Agricultural fields, grass, sparse/low vegetation | Moderate NDVI (0.2–0.5) |
| 4 | bare/sparse | Bare soil, gravel, sparse vegetation, construction sites | High BSI, low NDVI (< 0.2) |
| 5 | water | Rivers, lakes, ponds | Negative NDVI, high MNDWI |

> **Known confusion risks:** built-up (1) vs bare/sparse (4) in dry season;
> tree (2) vs cropland (3) at field edges; hill shadows may be confused with
> water (5) or bare soil (4). All documented in `docs/limitations.md`.

---

## 5. Spectral Indices

All computed in `src/geointel/rs/indices.py`. Formulae are exact; no rounding
inside the computation (float32 throughout).

| Index | Formula | Bands | Range | Sensitivity |
|---|---|---|---|---|
| NDVI | (B08 - B04) / (B08 + B04) | NIR, Red | [-1, 1] | Photosynthetic vegetation density |
| NDBI | (B11 - B08) / (B11 + B08) | SWIR1, NIR | [-1, 1] | Built-up surfaces |
| MNDWI | (B03 - B11) / (B03 + B11) | Green, SWIR1 | [-1, 1] | Open water |
| BSI | ((B11+B04)-(B08+B02)) / ((B11+B04)+(B08+B02)) | SWIR1, Red, NIR, Blue | [-1, 1] | Bare soil |

> **Why these indices?** NDVI discriminates vegetation from non-vegetation. NDBI
> was designed specifically to highlight built-up areas in Sentinel/Landsat imagery
> (Zha et al. 2003 — UNVERIFIED DOI, verify via Scholar). MNDWI suppresses built-up
> and highlights water better than NDWI in urban areas. BSI helps separate bare soil
> from built-up when NDBI alone is ambiguous — important for Dehradun's construction
> sites and gravel riverbeds.

---

## 6. Change Detection Definitions

### 6.1 Urban Expansion (primary definition)

> **Urban expansion** is defined as any pixel that is classified as **NOT built-up
> (class != 1) in T1** and classified as **built-up (class = 1) in T2**.

- Unit of measurement: number of pixels x (10 m)^2 = area in m^2, converted to
  hectares (1 ha = 10,000 m^2). All area computation is in EPSG:32644.
- Represented as a binary raster (1 = urban expansion, 0 = no change).
- File: `data/processed/urban_expansion_mask.tif`

### 6.2 Vegetation Loss (primary definition)

> **Vegetation loss** is defined as any pixel that is classified as **tree/dense-
> vegetation (class 2) OR cropland/grass (class 3) in T1** and classified as
> **NOT vegetation (class != 2 AND class != 3) in T2**.

- Unit of measurement: hectares (same computation as above).
- Represented as a binary raster (1 = vegetation loss, 0 = no change).
- File: `data/processed/vegetation_loss_mask.tif`

### 6.3 NDVI-Difference Baseline (secondary, for comparison only)

> **NDVI-difference change** is defined as pixels where
> NDVI(T2) - NDVI(T1) < threshold (default: -0.15, configurable in config.yaml).

- This is a raw spectral change signal, NOT a land-cover classification.
- It is reported **separately** and **never substituted** for the primary definition.
- Seasonal caution: even with same-month compositing, phenological variation
  between years may produce spurious NDVI differences. This is documented as
  a limitation.
- File: `data/processed/ndvi_diff.tif`, `data/processed/ndvi_change_mask.tif`

### 6.4 "Associated With" Definition

> Two change types are "associated" if they **co-occur in the same 1 km grid cell**
> AND the co-occurrence is statistically significant at alpha = 0.05 using at least
> one of: Spearman rank correlation, Moran's I, or Getis-Ord Gi*.

- Grid: 1 km x 1 km fishnet over the AOI in EPSG:32644
- Each cell stores: urban-gain area (ha), vegetation-loss area (ha),
  veg-to-built overlap area (ha), total valid pixels
- Association is a **spatial correlation claim** only. It does NOT imply that
  urban expansion caused vegetation loss.

---

## 7. Post-Processing

| Parameter | Value | Config key |
|---|---|---|
| Majority filter window | 3 x 3 pixels | `change.majority_filter_size` |
| Minimum mapping unit | 0.5 ha | `change.minimum_mapping_unit_ha` |

> **Why majority filter?** Random Forest classification produces salt-and-pepper
> noise at 10 m. A 3x3 majority filter replaces isolated misclassified pixels
> with their neighbourhood majority class — standard post-processing in published
> studies. The filter is applied BEFORE transition matrix computation.
> **Why 0.5 ha MMU?** Below this size, the patch is smaller than 50 pixels and
> is likely a classification artefact rather than a real land-cover unit.

---

## 8. Accuracy Assessment

Following best practices (Olofsson et al. 2014 — UNVERIFIED, see papers_index.csv):

- **Spatial block cross-validation:** 5 km blocks; test set = 2 held-out blocks.
  This prevents spatial autocorrelation from inflating CV scores.
- **Reference labels:** Stratified random sample, ~150 points per class per epoch.
  Labels collected manually (you must do this) using QGIS or the Streamlit
  labelling page. Labels stored in `data/labels/labels.csv`.
- **Metrics reported:** OA, per-class precision/recall/F1, Cohen's Kappa,
  95% confidence intervals on OA (Wilson score interval).
- **Warning threshold:** If any class has < 30 test points, a warning is printed
  and metrics for that class are flagged as unreliable.

---

## 9. Assumptions and Limitations

| Assumption | Implication |
|---|---|
| Sentinel-2 L2A atmospheric correction is adequate | Residual haze in some scenes may bias reflectance |
| Seasonal compositing (same Nov–Mar window) removes phenological bias | Year-to-year weather variation (drought, late monsoon) can still cause NDVI shifts |
| 10 m resolution captures urban expansion patches | Individual buildings, narrow roads, and alleys < 10 m are NOT resolved |
| Labels are representative of ground truth | Labelling errors or sampling bias propagate to the classifier |
| 5-class scheme covers all land cover | Rare classes (flooded fallow, construction debris) may be misclassified |
| OSM data reflects real-world features | OSM completeness varies; OSM growth does not necessarily reflect road construction |

Full limitations list: `docs/limitations.md` (Phase 7).
