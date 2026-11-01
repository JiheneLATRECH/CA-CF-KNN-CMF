# Codes

This directory contains the Python implementations used to reproduce the experiments reported in the paper:

**Probabilistic Context-Aware Collaborative Filtering with Distribution-Based Neighborhoods and Contextual Matrix Factorization**

The repository provides one dataset-specific pipeline for each experimental dataset:

- `ldos_comoda_pipeline.py`
- `depaulmovie_pipeline.py`
- `frappe_pipeline.py`

Each pipeline follows the same general experimental framework while preserving the dataset-specific preprocessing and contextual representation described in the paper.

---

## Directory Structure

```text
Codes/
├── README.md
├── ldos_comoda_pipeline.py
├── depaulmovie_pipeline.py
└── frappe_pipeline.py
```

The corresponding datasets are stored in the `Datasets/` directory.

---

## Experimental Pipeline

Each script implements the three main stages of the proposed approach.

### Phase 1 — Contextual Representation

This phase performs:

- dataset-specific preprocessing;
- construction of contextual dimensions;
- contextual-state modelling;
- observation-level soft-membership estimation;
- probabilistic user-profile construction.

The contextual representation is adapted to each dataset.

### LDOS-CoMoDa

Three contextual dimensions are considered:

- Temporal;
- Spatial-environmental;
- Affective-physical.

Candidate contextual models are evaluated according to the clustering-selection procedure described in the paper.

### DePaulMovie

Three categorical contextual dimensions are used:

- Temporal (`Time`);
- Spatial (`Location`);
- Social (`Companion`).

The contextual states correspond directly to the observed modalities, with cardinalities 2, 2 and 3, respectively.

### Frappe

Three contextual dimensions are considered:

- Temporal;
- Spatial-environmental;
- Situational.

Since Frappe provides implicit usage counts rather than explicit ratings, usage counts are transformed into bounded pseudo-ratings before recommendation evaluation.

---

## Phase 2 — Contextual Similarity

Probabilistic contextual profiles are compared using distributional distances.

The implementation supports:

- Hellinger distance;
- Jensen-Shannon distance;
- Wasserstein distance, where applicable.

The dimension-specific dissimilarities are then combined using:

- uniform weighting;
- reliability-aware weighting.

Contextual reliability incorporates both observation support and membership confidence.

---

## Phase 3 — Recommendation and Evaluation

The final phase evaluates the recommendation models using five-fold outer cross-validation with inner validation.

The proposed methods include:

- **Context-KNN-unif**
- **Context-KNN-rel**
- **Context-KNN-pred**
- **CMF-unif**
- **CMF-rel**
- **CMF-relw**
- **CMF-pred**

The comparison baselines include:

- Global Mean;
- User Mean;
- Item Mean;
- regularized user/item bias predictor;
- Pearson user-based KNN;
- contextual pre-filtering KNN;
- Biased Matrix Factorization;
- Matrix Factorization with contextual biases.

---

## Evaluation Regimes

The experiments are conducted under three evaluation regimes:

1. **Full-data regime** — all available training interactions and contextual evidence are used.
2. **Limited-evidence regime** — the amount of contextual evidence available for user-profile construction is reduced.
3. **Controlled cold-start regime** — selected users retain only three training interactions.

---

## Statistical Testing

Statistical comparisons are performed using paired Wilcoxon signed-rank tests.

Prediction errors are first aggregated at the **user level** using user-level MAE. These values constitute the paired observations used in the statistical tests.

The comparisons include:

- CMF-pred vs. CMF-unif;
- CMF-pred vs. Biased MF;
- CMF-pred vs. contextual pre-filtering;
- Context-KNN-pred vs. Context-KNN-unif;
- Context-KNN-pred vs. Pearson user-KNN.

---

## Reproducibility Settings

The main experimental settings are:

```text
Outer folds:               5
Evaluation seed:           2024
Inner validation fraction: 0.20
Validation objective:      RMSE
Clustering seed:           42
```

The default execution uses a single worker.

---

## Main Hyperparameter Grids

```text
K nearest neighbours:
{5, 10, 20, 30, 50}

Similarity contrast:
{1, 2, 4, 8}

Reliability saturation:
{1, 0.5, 0.25} × median observation support

Bias regularization:
{2, 5, 10}

Neighbourhood shrinkage:
{0, 2, 5, 10}

Latent factors:
10

MF / CMF epochs:
40

Learning rate:
0.01

MF regularization:
{0.02, 0.05, 0.10}

CMF reliability parameter:
{1, 3, 10}

Context-KNN prediction-aware simplex step:
0.10

CMF prediction-aware simplex step:
0.25

Prediction-aware regularization:
1e-3
```

Dataset-specific restrictions are implemented directly in the corresponding scripts.

---

## Running the Code

The scripts can be executed independently.

### LDOS-CoMoDa

```bash
python Codes/ldos_comoda_pipeline.py --input Datasets/LDOS-CoMoDa.csv
```

### DePaulMovie

```bash
python Codes/depaulmovie_pipeline.py --input Datasets/ratings.txt
```

### Frappe

```bash
python Codes/frappe_pipeline.py --input Datasets/frappe.csv
```

Individual phases can also be executed separately using:

```bash
python Codes/ldos_comoda_pipeline.py --phase 1 --input Datasets/LDOS-CoMoDa.csv
python Codes/ldos_comoda_pipeline.py --phase 2
python Codes/ldos_comoda_pipeline.py --phase 3
```

The same principle applies to the DePaulMovie and Frappe pipelines.

---

## Generated Outputs

Depending on the dataset and execution phase, the scripts generate:

- cleaned datasets;
- preprocessing reports;
- selected contextual-state configurations;
- soft-membership matrices;
- probabilistic user profiles;
- contextual distance matrices;
- reliability values;
- contextual similarity matrices;
- selected hyperparameters;
- fold-level predictions;
- MAE and RMSE results;
- ranking metrics;
- statistical-test results;
- experimental figures and tables.

Generated outputs should not be edited manually.

---

## Software Requirements

The implementation relies mainly on:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
numba
```

`psutil` is optional and is used for runtime and memory instrumentation when available.

For exact reproducibility, the tested package versions should be recorded in the repository-level `requirements.txt`.

---

## Reproducibility Note

The code corresponds to the experimental protocol reported in the paper.

Dataset-specific differences are intentional and reflect differences in dataset structure, contextual information and scale.

Outer test folds are never used for hyperparameter selection. Model-selection decisions are based only on training data and inner-validation data.
