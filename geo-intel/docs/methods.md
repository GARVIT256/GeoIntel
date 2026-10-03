# GEO-INTEL — Methodology Specification
**Phase 3 — Land Cover Classification and Change Detection**

This document details the mathematical formulas, physical rules, model architectures, and validation methodology used in Phase 3 of GEO-INTEL.

---

## 1. Spectral Index Formulas

Spectral indices exploit differential reflectance characteristics across satellite bands to highlight specific land cover types. All formulas operate on surface reflectance (Sentinel-2 Level-2A) clipped to $[-1.0, 1.0]$.

### 1.1 Normalized Difference Vegetation Index (NDVI)
$$\text{NDVI} = \frac{\text{B08 (NIR)} - \text{B04 (Red)}}{\text{B08 (NIR)} + \text{B04 (Red)}}$$
* **Physical Basis**: Chlorophyll absorbs Red light (665 nm) for photosynthesis while mesophyll leaf cell structure strongly scatters NIR light (842 nm).
* **Thresholds**: Dense forest $> 0.5$, sparse/grassland $0.25 - 0.50$, bare soil $0.0 - 0.20$, water $< 0.0$.

### 1.2 Normalized Difference Built-up Index (NDBI)
$$\text{NDBI} = \frac{\text{B11 (SWIR1)} - \text{B08 (NIR)}}{\text{B11 (SWIR1)} + \text{B08 (NIR)}}$$
* **Physical Basis**: Impervious surfaces (concrete, asphalt, brick) exhibit higher reflectance in SWIR1 (1610 nm) than in NIR (842 nm).
* **Thresholds**: Built-up areas $> 0.0$, vegetated/water areas $< 0.0$.

### 1.3 Modified Normalized Difference Water Index (MNDWI)
$$\text{MNDWI} = \frac{\text{B03 (Green)} - \text{B11 (SWIR1)}}{\text{B03 (Green)} + \text{B11 (SWIR1)}}$$
* **Physical Basis**: Water bodies absorb SWIR wavelengths completely while scattering Green light (560 nm). MNDWI suppresses built-up background noise better than standard NDWI.
* **Thresholds**: Open water bodies $> 0.0$, non-water land $\le 0.0$.

### 1.4 Bare Soil Index (BSI)
$$\text{BSI} = \frac{(\text{B11} + \text{B04}) - (\text{B08} + \text{B02})}{(\text{B11} + \text{B04}) + (\text{B08} + \text{B02})}$$
* **Physical Basis**: Combines Blue, Red, NIR, and SWIR bands to isolate exposed mineral soil from built-up structures and dry vegetation.
* **Thresholds**: Exposed soil / construction earth $> 0.0$, vegetation/water $< 0.0$.

---

## 2. Baseline Classifiers

GEO-INTEL uses a five-class scheme, matching `config/config.yaml`:

### LULC Class Scheme
| Code | Class Name | Description | Color Code |
| :---: | :--- | :--- | :---: |
| **1** | Built-up | Dense urban, roads, industrial zones, structures | `#E31A1C` |
| **2** | Tree/dense vegetation | Forests, plantations, dense shrubs | `#33A02C` |
| **3** | Cropland/grass/low vegetation | Cropland, grass, sparse vegetation | `#B2DF8A` |
| **4** | Bare/sparse | Riverbed gravel, fallow land, exposed earth | `#FDBF6F` |
| **5** | Water | Rivers, reservoirs, lakes, ponds | `#1F78B4` |
| **0** | Unclassified | No-data / cloud-masked pixels | `#000000` |

The five classes are the approved scheme. Unclassified pixels use code 0 and are excluded from class-area comparisons.

### 2.1 Baseline 1: Rule-Based Decision Tree Classifier
Strict priority: water > built-up > tree > cropland/grass > bare > unclassified.
1. **Water (class 5)**: $\text{MNDWI} > 0.0$
2. **Built-up (class 1)**: $\text{NDBI} > \text{NDVI} \land \text{NDBI} > 0.0$
3. **Tree (class 2)**: $\text{NDVI} > 0.50$
4. **Cropland/grass (class 3)**: $0.25 < \text{NDVI} \le 0.50$
5. **Bare (class 4)**: $\text{BSI} > 0.0 \land \text{NDVI} \le 0.25$; remaining valid pixels are unclassified (0). BSI <= 0 and NDVI <= 0.25 is unclassified absent a higher-priority match.

### 2.2 Baseline 2: Supervised Random Forest Classifier
* **Feature Vector (13 features)**:
  * 6 Spectral Bands: B02, B03, B04, B08, B11, B12
  * 4 Spectral Indices: NDVI, NDBI, MNDWI, BSI
  * 3 Terrain Features: DEM slope, $\sin(\text{aspect})$, $\cos(\text{aspect})$
* **Bootstrap only**: Pseudo-labels are extracted from the rule-based classifier. This is circular; RF agreement with those labels is not independent accuracy and must not be reported as such.
* **Model Parameters**: 100 decision trees, out-of-bag scoring enabled (`oob_score=True`), balanced class weighting (`class_weight='balanced'`).
* **Independent validation**: Use `src/geointel/change/accuracy.py` with supplied `data/labels/labels.csv`; metrics are evaluated on spatially held-out blocks and are unavailable until reference labels exist.
* **Terrain resampling**: Copernicus DEM elevation is 30 m. Derive slope and aspect on that native grid; encode aspect to sine and cosine there, then bilinearly resample elevation, slope, aspect-sine, and aspect-cosine independently to the Sentinel-2 10 m reference grid. Do not interpolate aspect degrees across the 0°/360° seam. RF consumes the resampled sine/cosine features.

### 2.3 Terrain preparation convention

Copernicus DEM elevation is 30 m. Calculate slope, aspect, `sin(aspect)`, and
`cos(aspect)` at the native 30 m grid. Resample elevation, slope, aspect-sine,
and aspect-cosine to the Sentinel-2 10 m reference grid with bilinear
interpolation. Never interpolate aspect degrees across its circular 0°/360°
boundary. This records the pipeline convention; no real DEM output is claimed.

---

## 3. Transition Matrix & Change Statistics

For two LULC maps $L_1$ (Epoch T1) and $L_2$ (Epoch T2) at 10 m resolution ($\text{Pixel Area} = 100\text{ m}^2 = 0.0001\text{ km}^2$), the matrix is $5 \times 5$ over class IDs 1–5. A pixel contributes only when both epochs are finite and have a class ID in 1–5; zero, NaN, and out-of-range IDs are excluded jointly. Each result includes the jointly excluded pixel count, excluded area in km², and percentage of the full raster footprint (not the independently measured AOI polygon area).

$$M_{ij} = \text{Count of pixels where } L_1(x, y) = i \land L_2(x, y) = j \quad \text{for } i, j \in \{1, 2, 3, 4, 5\}$$

$$\text{Area}_{ij} (\text{km}^2) = M_{ij} \times 0.0001$$

Key environmental trajectory metrics:
* **Tree/cropland conversion to built-up**: $\text{Area}_{2 \to 1} + \text{Area}_{3 \to 1}$
* **Bare/sparse conversion to built-up**: $\text{Area}_{4 \to 1}$
* **Urban Expansion Mask**: $\mathbb{I}(L_1 \neq 1 \land L_2 = 1)$
* **Vegetation Loss Mask**: $\mathbb{I}(L_1 \in \{2,3\} \land L_2 \notin \{2,3\})$
