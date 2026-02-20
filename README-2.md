# SILVER-Pain Dataset

Persistent and chronic pain has been shown to be strongly associated with risks of severe health conditions, including psychiatric comorbidities (e.g., depression and anxiety), cardiovascular disease, and ultimately, higher mortality rates[^1],[^2],[^3]. More than 36% of older adults (≥65 y.o.) in the U.S. report chronic pain[^4]. However, among nearly **1.4 million** long-stay nursing home residents with persistent pain, **6.4%** **receive** **no analgesics;** and >30% receive no scheduled analgesics[^5]. Here, **communication barrier** (e.g., Alzheimer's Disease and Related Dementia) is one of the most significant sources of under-treatment of pain. We hereby publish our SILVER-Pain Dataset, aiming to facilitate research in the timely detection of unexpressed pain and hence the prevention of the development of more comorbidities using non-invasive, objective sensing with interpretable and intelligent pain assessment.

The SILVER (Senior Inclusive Laboratory Video and Empatica Recordings of) Pain Dataset is a multi-modal pain assessment dataset consisting of recordings of videos and physiological signals of 25 subjects (**18** **young** + **7 old**) on their responses to experimental pain with real-time self annotation of pain level.

![1771521848993](https://file+.vscode-resource.vscode-cdn.net/Users/xxd/Documents/PhD/PainAssessment/SILVER-Pain%20Dataset/image/README-2/1771521848993.png)


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

The SILVER-Pain Dataset is designed to support research on **automatic pain assessment**, with an inclusion on **older adults** and **privacy-preserving sensing**. 

Physiological signals in the young cohort were collected with the Empatica E4 wristband. Those from the older cohort was collected with the newer Empatica EmbracePlus wristband (which is, according to Empatica, a updated version of E4 after E4 was sunset).

Deidentified depth video, thermal video and facial expression data in both cohorts were collected by (1) an Intel RealSense 435i Camera, (2) a TOPDON TC004 thermal camera, and (3) extractions from the RGB video colleceted by the RealSense camera with the open-source OpenFace library: [https://github.com/TadasBaltrusaitis/OpenFace]().

Note that raw data is not synchronously and regularly sampled. There are also missing values and signal artifacts, and young adults' data format is slightly different from older adults (see readmes in subfolders). We provide basic functions with some configuration options to extract and preprocess the raw data into machine learning ready dataframe and data arrays. If you wish, you can build and use your own functions to extract and process the signals.

Please see subfolders for more specific information.

This repository includes both original files and processed, ML-ready outputs, plus baseline utilities to parse, synchronize, and preprocess the raw recordings.

This repo provides:

- ✅ **Original data files** (as collected/exported)
- ✅ **Processed data** (cleaned + aligned tables/arrays where available)
- ✅ **Utility code** for parsing and basic preprocessing
- ✅ **Per-cohort notes** (see the README in each cohort subfolder)

> **Tip:** If you build your own preprocessing pipeline, you can still use our loaders to read the raw files reliably.

---

## Cohorts & devices

| Cohort |  N | Wristband                           | Notes                                                                       |
| -----: | -: | ----------------------------------- | --------------------------------------------------------------------------- |
|  Young | 18 | Empatica<br />**E4**          | Earlier device generation                                                   |
|  Older |  7 | Empatica<br />**EmbracePlus** | Newer device; per Empatica, an updated successor to E4 (E4 has been sunset) |

---

## Modalities

### Physiological (wrist-worn)

Collected from Empatica devices (availability may vary by cohort and session):

- **BVP** (blood volume pulse) **/ PPG-derived signals**
- **EDA** (electrodermal activities)
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

- a **machine-learning-ready dataframe**, (almost) aligned on a chosen time grid. See readme in older adults' subfolder on why time grid is not perfectly 64 Hz.
- and/or **numpy arrays** for model input

Because repo layouts vary across labs, we **do not** hard-code a fixed pipeline here. The intended workflow is:

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

[^1]: Mills, S. E., Nicolson, K. P., & Smith, B. H. (2019). Chronic pain: a review of its epidemiology and associated factors in population-based studies. *British journal of anaesthesia* , *123* (2), e273-e283.
    
[^2]: Stretanski, M. F., Kopitnik, N. L., Matha, A., & Conermann, T. (2025). Chronic pain. In *StatPearls [Internet]*. StatPearls Publishing.
    
[^3]: Ray, B. M., Kelleran, K. J., Fodero, J. G., & Harvell-Bowman, L. A. (2024). Examining the relationship between chronic pain and mortality in US adults. *The Journal of Pain* , *25* (10), 104620.
    
[^4]: Lucas, J. W., & Sohi, I. (2024). Chronic pain and high-impact chronic pain in US adults, 2023.
    
[^5]: Hunnicutt, J. N., Ulbricht, C. M., Tjia, J., & Lapane, K. L. (2017). Pain and pharmacologic pain management in long-stay nursing home residents. Pain, 158(6), 1091-1099.
