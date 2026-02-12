# SILVER-Pain Dataset

Multimodal, privacy-preserving pain assessment data collected from **two cohorts**:

- **Older cohort:** 7 subjects (Empatica **EmbracePlus**)
- **Young cohort:** 18 subjects (Empatica **E4**)

This repository includes **both original files and processed, ML-ready outputs**, plus **baseline utilities** to parse, synchronize, and preprocess the raw recordings.

---

## Table of Contents

- [What’s inside](#whats-inside)
- [Cohorts &amp; devices](#cohorts--devices)
- [Modalities](#modalities)
- [Data characteristics &amp; known issues](#data-characteristics--known-issues)
- [Repository organization](#repository-organization)
- [Quick start](#quick-start)
- [Preprocessing philosophy](#preprocessing-philosophy)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## What’s inside

The SILVER-Pain Dataset is designed to support research on **automatic pain assessment**, with an emphasis on **older adults** and **privacy-preserving sensing**.

This repo provides:

- ✅ **Original data files** (as collected/exported)
- ✅ **Processed data** (cleaned + aligned tables/arrays where available)
- ✅ **Utility code** for parsing and basic preprocessing
- ✅ **Per-cohort notes** (see the README in each cohort subfolder)

> **Tip:** If you build your own preprocessing pipeline, you can still use our loaders to read the raw files reliably.

---

## Cohorts & devices

| Cohort |  N | Wristband                     | Notes                                                                       |
| -----: | -: | ----------------------------- | --------------------------------------------------------------------------- |
|  Young | 18 | Empatica**E4**          | Earlier device generation                                                   |
|  Older |  7 | Empatica**EmbracePlus** | Newer device; per Empatica, an updated successor to E4 (E4 has been sunset) |

---

## Modalities

### Physiological (wrist-worn)

Collected from Empatica devices (availability may vary by cohort and session):

- **BVP / PPG-derived signals**
- **EDA**
- **Skin temperature**
- **(Optional / derived)** heart rate / inter-beat intervals depending on processing

### Video-derived / privacy-preserving sensing

De-identified sensing and derived features:

- **Depth video** (Intel RealSense D435i)
- **Thermal video** (thermal camera)
- **Facial expression features** extracted from RGB via **OpenFace**
  - OpenFace (GitHub): https://github.com/TadasBaltrusaitis/OpenFace

> We emphasize privacy: depth/thermal and derived facial features can reduce identification risk compared to raw RGB. Please see cohort subfolder READMEs for exactly what is distributed.

---

## Data characteristics & known issues

Real-world multimodal data is messy. Please read this section before modeling.

- **Not regularly sampled / not perfectly synchronized:**Raw streams may have irregular timestamps and sensor-specific sampling behavior.
- **Missing values:**Some channels contain gaps; some sessions may have partial recordings.
- **Signal artifacts:**Motion and contact issues can introduce noise, discontinuities, and outliers.
- **Format differences across cohorts:**
  The young and older cohorts were collected with different Empatica devices and may differ in file formats/fields.

✅ We provide **basic preprocessing functions with configurable options** (e.g., resampling strategy, interpolation vs. snapping, filtering).
📌 For cohort-specific details and file conventions, **see the README inside each cohort folder**.

---

## Repository organization

Please refer to subfolder READMEs for the authoritative structure. Typical contents include:

- Cohort folders (e.g., `young/`, `older/`)
- `raw/` or equivalent original exports
- `processed/` ML-ready tables/arrays (if included)
- `src/` / `scripts/` utilities for loading and preprocessing
- `configs/` optional configuration templates

---

## Quick start

### 1) Explore the cohort subfolders

Start with:

- `./young/README.md`
- `./older/README.md`

They document:

- file naming conventions
- available modalities per session
- known caveats and recommended preprocessing defaults

### 2) Use the provided preprocessing utilities (recommended)

We provide baseline functions to transform raw recordings into:

- a **machine-learning-ready dataframe** (aligned on a chosen time grid)
- and/or **numpy arrays** for model input

Because repo layouts vary across labs, we avoid hard-coding commands here. The intended workflow is:

1. Select a cohort/session
2. Load raw streams (physiology + video-derived features)
3. Choose a target time grid (e.g., BVP grid or 1 Hz grid)
4. Resample other channels onto the grid (snap or interpolate)
5. Clean/filter artifacts
6. Export ML-ready tables/arrays

**Example (illustrative) Python usage**

```python
...
```
