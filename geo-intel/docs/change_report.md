# GEO-INTEL — Change Report
**Phase 3 — Land Cover Classification and Change Detection**  
**Status:** NOT RUN — fill only from real composite/classification outputs; do not infer metrics from synthetic tests  

> All numbers below marked **[NOT YET RUN]** will be populated automatically by `geointel.change.detection` execution on actual satellite composites.

---

## 1. Study Area Summary
* **AOI**: Dehradun District, Uttarakhand, India
* **Epoch T1**: 2018-19 Dry Season (Nov 2018 – Mar 2019)
* **Epoch T2**: 2024-25 Dry Season (Nov 2024 – Mar 2025)
* **Resolution**: 10 m native grid (EPSG:32644)
* **Total Analyzed Area**: **[NOT YET RUN]** km²

---

## 2. Land Cover Class Area Summary (T1 vs T2)

| Class ID | Land Cover Class | T1 Area (km²) | T1 Share (%) | T2 Area (km²) | T2 Share (%) | Net Change (km²) | Relative Change (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Built-up | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **2** | Tree/dense vegetation | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **3** | Cropland/grass/low vegetation | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **4** | Bare/sparse | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **5** | Water | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |

---

## 3. Transition Matrix (T1 → T2 in km²)

| T1 \ T2 Class | 1: Built-up | 2: Tree/dense | 3: Cropland/grass | 4: Bare/sparse | 5: Water | T1 Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1: Built-up** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **2: Tree/dense** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **3: Cropland/grass** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **4: Bare/sparse** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **5: Water** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |
| **T2 Total** | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] | [NOT YET RUN] |

---

## 4. Key Environmental Trajectory Findings
* **Vegetation Conversion to Built-up**: **[NOT YET RUN]** km²
* **Bare Soil Conversion to Built-up**: **[NOT YET RUN]** km²
* **Total Urban Expansion**: **[NOT YET RUN]** km²
* **Total Vegetation Loss**: **[NOT YET RUN]** km²

---

## 5. Random Forest Bootstrap Diagnostics
Rule-based pseudo-labels are circular with this classifier. OOB and cross-validation scores on these labels are **bootstrap-only diagnostics, not independent accuracy**.

## 6. Independent Accuracy
Run `python -m geointel.change.accuracy --labels data/labels/labels.csv --prediction-raster <classified-map.tif>` after supplying reference labels. Until then, OA, per-class metrics, kappa, and confusion matrix remain **unavailable**.

## 7. Bootstrap Feature Diagnostics
* **Top 3 Features by Importance**:
  1. **[NOT YET RUN]**
  2. **[NOT YET RUN]**
  3. **[NOT YET RUN]**
