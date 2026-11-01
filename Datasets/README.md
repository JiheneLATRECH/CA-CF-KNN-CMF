# Datasets

This directory contains information and/or data files used in the experiments reported in the paper:

**Probabilistic Context-Aware Collaborative Filtering with Distribution-Based Neighborhoods and Contextual Matrix Factorization**

The experiments were conducted on three context-aware recommendation datasets:

- LDOS-CoMoDa
- DePaulMovie
- Frappe

## 1. LDOS-CoMoDa

LDOS-CoMoDa is a context-aware movie recommendation dataset containing explicit user ratings together with contextual information describing the conditions under which movies were consumed.

The experiments use three contextual dimensions:

- **Temporal:** `time`, `daytype`, `season`
- **Spatial-environmental:** `location`, `weather`
- **Affective-physical:** `mood`, `dominantEmo`, `endEmo`, `physical`

After preprocessing, the dataset contains:

- 2,294 rating observations
- 121 users
- 1,232 items

Missing contextual values are treated as unavailable information rather than as valid contextual modalities.

Suggested location:

```text
Datasets/
└── LDOS-CoMoDa.csv
```

---

## 2. DePaulMovie

DePaulMovie is a context-aware movie recommendation dataset containing explicit ratings and three contextual variables.

The contextual dimensions used in the experiments are:

- **Temporal:** `Time`
- **Spatial:** `Location`
- **Social:** `Companion`

After removing exact duplicate observations, the dataset contains:

- 5,035 ratings
- 97 users
- 79 items

A subset of 3,589 observations contains usable contextual information, involving 81 users.

Suggested location:

```text
Datasets/
└── DePaulMovie.txt
```

---

## 3. Frappe

Frappe is a context-aware mobile application recommendation dataset based on implicit application usage.

The experiments use the following contextual dimensions:

- **Temporal:** `daytime`, `weekday`
- **Spatial-environmental:** `homework`, `weather`
- **Situational:** `cost`

The dataset contains:

- 96,203 observations
- 957 users
- 4,082 applications

Because Frappe provides implicit usage counts rather than explicit ratings, the usage count is transformed into a bounded pseudo-rating before model evaluation. The transformation is implemented directly in the corresponding preprocessing pipeline


## Data Preprocessing

Dataset-specific preprocessing is implemented in the corresponding scripts available in the `Codes/` directory.

The preprocessing procedures include, depending on the dataset:

- removal of exact duplicate observations;
- handling of missing contextual information;
- validation of ratings or usage counts;
- construction of contextual dimensions;
- treatment of rare contextual modalities;
- generation of Frappe pseudo-ratings.

The processed datasets are generated automatically by the reproducibility scripts and should not be edited manually.

## Data Availability

The datasets remain subject to the terms and conditions of their original providers.

If redistribution of a dataset is not permitted, the raw data file should not be included in this repository. Instead, this directory should contain instructions describing how to obtain the dataset from its original source and how to place it in the expected directory before running the experiments.

## Reproducibility

The code assumes the following default file locations:

```text
Datasets/LDOS-CoMoDa.csv
Datasets/DePaulDeMovie.txt
Datasets/frappe.csv
```


