# MONIPY – ECG-Based Seizure Detection Pipeline

## Overview

This repository contains a ECG seizure detection pipeline
developed for retrospective analysis of long-term ECG recordings.
The pipeline implements signal preprocessing, R-peak detection and correction,
feature extraction, dataset construction, and seizure detection evaluation.


The pipeline is intended for **offline analysis and model evaluation** rather
than real-time deployment.

---

## Pipeline Summary

The processing workflow consists of the following sequential stages:

1. ECG loading and R-peak detection  
2. R-peak and RR-interval correction  
3. Feature extraction with quality control  
4. Training sample generation  
5. Seizure detection evaluation  

Each stage produces explicit artifacts that are reused by subsequent steps,
enabling complete reconstruction of intermediate results.

---

## Project Structure

```text
monipy/
│
├── data/
│ └── Core data structures, feature tables, and windowing logic
│
├── models/
│ └── Machine-learning models for seizure detection
│
├── utils/
│ └── Shared utilities (metrics, scaling, detection helpers, configuration)
│
├── evaluations/
│ │
│ ├── ECGlabeling.py
│ │ └── ECG loading, preprocessing, and R-peak detection
│ │
│ ├── RRcorrect.py
│ │ └── Physiologically constrained R-peak and RR-interval correction
│ │
│ ├── run_data_correction.py
│ │ └── Applies R-peak correction to raw ECG recordings
│ │
│ ├── run_feature_comp.py
│ │ └── Computes RR-based and heart-rate-variability features
│ │
│ ├── compute_training_samples.py
│ │ └── Generates labeled seizure / non-seizure training datasets
│ │
│ └── model_evalution.py
│ └── Performs seizure detection evaluation and result aggregation
│
├── reproducibility.py
│ └── Global random seed handling and deterministic execution control
│
└── README.md



---

## Execution Scripts (Main Entry Points)

The following scripts represent the **primary entry points** of the pipeline.
They are typically executed sequentially to reproduce the full analysis.

---

### `run_data_correction.py`

Applies R-peak correction to raw ECG recordings.
Correction is performed only on segments that violate physiological
or temporal constraints.

```bash
# python run_data_correction.py
```

---

### `run_feature_comp.py`

Computes RR-interval and heart-rate-variability features from corrected ECG data.
Feature computation is performed in fixed sliding windows and includes
quality-based filtering.

```bash
# python run_feature_comp.py
```

---

### `compute_training_samples.py`

Generates labeled training samples for machine-learning-based seizure detection.
Windows are labeled as seizure or non-seizure based on clinically annotated
seizure onset times.

```bash
# python compute_training_samples.py
```

---

### `model_evalution.py`

Evaluates seizure detection performance on prepared datasets.
Predictions are converted into detection events and matched to seizure onsets
using a fixed temporal tolerance window.

```bash
# python model_evalution.py
```

---

---

### `reproducibility.py`

Defines a single entry point for setting global random seeds across Python,
NumPy, and PyTorch.
Optional deterministic execution can be enforced to ensure full reproducibility.

---

### `ECGlabeling.py`

Handles ECG signal loading, optional preprocessing, and R-peak detection.
Multiple detector backends are supported to allow method comparison.

---

### `RRcorrect.py`

Implements RR-interval correction logic based on local likelihood optimization
and physiological constraints.
Used internally by `run_data_correction.py`.

---

## Reproducibility Considerations

- All scripts explicitly set random seeds
- Deterministic execution can be enabled globally
- Intermediate results are stored as files and reused across stages
- No hidden state or database dependencies are present

---


The code is not optimized for real-time inference or embedded deployment.

---

## Citation

```bibtex
@unpublished{Alhaskir2026TrustworthyAI,
  author = {Alhaskir, Mohamed and Bienzeisler, Jonas and Klett, Kevin and Hofmeister, Julian and Lutz, Florian and Schriewer, Elisabeth and Wolking, Stefan and Röhrig, Rainer and Koch, Henner and Weber, Yvonne and Kutafina, Ekaterina and Spicher, Nicolai},
  title  = {Trustworthy Artificial Intelligence: Seizure Detection by Wearables Following International Consensus Guidelines},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```
