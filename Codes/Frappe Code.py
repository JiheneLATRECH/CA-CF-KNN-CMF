"""
Frappe - complete context-driven collaborative-filtering pipeline (self-contained)
==================================================================================
Phases 1-3 of the approach with the Frappe configuration built in (frappe.csv, tab-separated):
pseudo-ratings from usage counts, three contextual dimensions (temporal, spatial-environmental
with "unknown" as missing, situational), exploratory clustering on a sample then refit on all
observations, O(n) clustering families, Wasserstein disabled at this scale, contextual
factorisation on fuzzy clusters with the weighting schemes (unif / rel / relw / pred),
baselines, three evaluation regimes, paired tests, parallel layer and timings.
  python frappe_full_pipeline.py --input frappe.csv                    # phases 1, 2, 3
  python frappe_full_pipeline.py --phase 3 --regimes full,cold_start --repeats 3
  CONTEXT_CF_N_JOBS=8 python frappe_full_pipeline.py                   # parallel folds (Linux/macOS)
Recommended: pip install numba (20x faster factorisation kernels).
"""

from __future__ import annotations
import time



import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                      # pure-python fallback (slower, identical results)
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)



# =============================================================================
# PARALLEL EXECUTION AND INSTRUMENTATION LAYER
# =============================================================================
#
# The approach is unchanged; this layer only executes its independent
# blocks concurrently and records their runtime:
#   * Phase 1 : clustering candidates (family x hyperparameters x K),
#               each with its initialisation/bootstrap stability runs
#   * Phase 2 : user-pair blocks of the distributional distance matrices
#               (Jensen-Shannon rows, Wasserstein LP pairs)
#   * Phase 3 : folds (x regimes x repeats), and the candidate alpha
#               vectors of the prediction-aware selection
# N_JOBS = 1 reproduces the sequential execution exactly. Workers are
# forked (Linux/macOS) so that dataset adapters' configuration overrides
# are inherited; on platforms without fork, N_JOBS is forced to 1.

import multiprocessing as _mp
import os as _os
import time as _time
from contextlib import contextmanager

try:                                   # Unix only; on Windows peak memory is read through psutil when available
    import resource as _resource
except ImportError:
    _resource = None
try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _peak_rss_mb() -> float:
    if _resource is not None:
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0
    if _psutil is not None:
        return _psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    return float("nan")

N_JOBS = int(_os.environ.get("CONTEXT_CF_N_JOBS", "1"))
# Parallelism inside a fold (alpha candidates). Kept at 1 when folds are
# already run in parallel, to avoid nested worker pools.
N_JOBS_INNER = 1
TIMINGS: dict[str, list[dict]] = {}


def _fork_available() -> bool:
    try:
        return "fork" in _mp.get_all_start_methods()
    except Exception:
        return False


def parallel_map(func, items, n_jobs: int | None = None, chunksize: int = 1):
    """Ordered map over independent items; sequential when n_jobs == 1."""
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    items = list(items)
    in_worker = _mp.current_process().name != "MainProcess"     # no nested pools: one level of parallelism at a time
    if n_jobs <= 1 or len(items) <= 1 or in_worker or not _fork_available():
        return [func(x) for x in items]
    with _mp.get_context("fork").Pool(processes=min(n_jobs, len(items))) as pool:
        return pool.map(func, items, chunksize=chunksize)


@contextmanager
def timed(label: str, **meta):
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        dt = _time.perf_counter() - t0
        TIMINGS.setdefault(label, []).append({"seconds": round(dt, 3), "n_jobs": N_JOBS, "peak_rss_mb": round(_peak_rss_mb(), 1), **meta})


def save_timings(path: Path) -> None:
    path.write_text(json.dumps({"n_jobs": N_JOBS, "cpu_count": _os.cpu_count(), "timings": TIMINGS}, indent=2), encoding="utf-8")


# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================
#
# What the approach REQUIRES per contextual dimension:
#   (1) an independent clustering process,
#   (2) a normalized observation-level soft-membership vector M_{u,n}^d,
#   (3) the preservation of N_u^d and M_{u,n}^d for the reliability
#       mechanism (observation support x membership confidence),
#   (4) cluster representatives usable to build a transport-cost matrix
#       when the Wasserstein distance is retained for the dimension.
#
# What the approach leaves OPEN: algorithm, K, hyperparameters, metrics.
#
# Design of this version (v3):
#   - Candidate algorithm FAMILIES are restricted per dimension by the
#     NATURE OF ITS VARIABLES (nominal / ordinal / cardinality), which is
#     a principled rule and not an a priori assignment by name. The
#     retained algorithm therefore legitimately differs across
#     dimensions, as the approach intends ("an appropriate algorithm
#     per dimension").
#   - Selection is done in TWO STAGES: (1) best K / hyperparameters
#     within each family, using the family's NATIVE criterion (ICL for
#     probabilistic models, silhouette for distance-based models);
#     (2) family winners are compared only on COMMON metrics, with
#     STABILITY as the primary criterion, then separation, then the
#     informativeness of the membership confidence.
#   - A spectral -> LCA hybrid provides stable, balanced partitions
#     with genuinely graded posterior probabilities.
#   - Continuous-embedding models (GMM on MCA) are eligible only when the
#     dimension has enough unique patterns for the embedding to form a
#     cloud rather than duplicated point masses, and use a strong
#     covariance floor to prevent degenerate near-crisp solutions.
#   - The K search range is deliberately narrow per dimension.
# =============================================================================

DEFAULT_INPUT_FILENAME = "LDOS-CoMoDa.csv"

# Narrow K ranges, per dimension. The previous wide ranges (up to 10)
# repeatedly produced boundary or near-enumeration solutions on
# low-cardinality dimensions. Ranges are bounded by the number of
# distinct users (see MAX_CLUSTERS_FRACTION_OF_USERS) anyway.
K_CANDIDATES = {
    "temporal": [2, 3, 4],
    "spatial_environmental": [2, 3, 4, 5, 6],
    "affective_physical": [2, 3, 4, 5, 6],
}

# Variable nature. "nominal": unordered categories (mismatch distance).
# "ordinal": ordered codes (normalized absolute rank difference).
FEATURE_TYPES = {
    "time": "nominal",          # morning / afternoon / evening / night
    "daytype": "nominal",       # working day / weekend / holiday
    "season": "nominal",
    "location": "nominal",
    "weather": "nominal",
    "mood": "ordinal",          # negative < neutral < positive
    "dominantEmo": "nominal",
    "endEmo": "nominal",
    "physical": "nominal",      # healthy / ill (binary)
    "social": "nominal",        # alone / partner / friends / colleagues / parents / public / family
    "decision": "nominal",      # user chose the movie / movie was suggested
    "interaction": "nominal",   # first / repeated interaction with the item
}

# Algorithm families and their eligibility rules.
#   fuzzy_k_modes         : all features nominal (matching distance)
#   fuzzy_k_medoids_gower : any categorical mix (mixed Gower)
#   latent_class_analysis : any categorical mix
#   spectral_lca_hybrid   : any categorical mix
#   gmm_mca               : only if unique patterns >= MIN_UNIQUE_PATTERNS_FOR_EMBEDDING
ALGORITHM_FAMILIES = [
    "fuzzy_k_modes",
    "fuzzy_k_medoids_gower",
    "latent_class_analysis",
    "spectral_lca_hybrid",
    "gmm_mca",
    "modality_identity",
]
# Single-variable dimensions (e.g. DePaulMovie: Time, Location, Companion):
# clustering a single nominal variable is trivial - its natural clusters
# are its modalities. The "modality_identity" family makes this explicit:
# K = number of modalities, one-hot memberships. The profile P_u^d is then
# the user's empirical distribution over modalities, and the reliability
# rests on observation support (confidence is 1 by construction).

MIN_UNIQUE_PATTERNS_FOR_EMBEDDING = 150

# Hyperparameter grids (modest by default)
FUZZINESS_GRID = [1.5, 2.0]
MCA_COMPONENTS_GRID = [5]
GMM_COVARIANCE_GRID = ["diag"]      # "full" is prone to collapse on discrete data
GMM_REG_COVAR_FRACTION = 0.05       # floor = fraction of mean embedding variance

N_INIT = 15
MAX_ITER = 200
TOL = 1e-6

INIT_STABILITY_RUNS = 5
BOOTSTRAP_RUNS = 8
BOOTSTRAP_FRACTION = 0.85

RANDOM_STATE = 42
LCA_SMOOTHING = 1e-6

# ---- Admissibility gates (hard) --------------------------------------------
MIN_CLUSTERS = 2
MAX_CLUSTERS_FRACTION_OF_USERS = 0.5
MIN_CLUSTER_FRACTION = 0.02
MIN_MEAN_OBS_PER_USER_PER_CLUSTER = 1.0
MIN_MEMBERSHIP_CONFIDENCE = 0.15
MAX_MEMBERSHIP_CONFIDENCE = 0.98   # near-crisp = non-informative for c_u^d
MIN_CONFIDENCE_DISPERSION = 0.03   # std of per-observation entropy
MIN_BOOTSTRAP_STABILITY = 0.60     # relaxed progressively if nothing passes
STABILITY_RELAXATION_STEPS = [0.60, 0.50, 0.40, 0.00]

# ---- Stage-2 comparison rule ------------------------------------------------
# Family winners within STABILITY_TOLERANCE of the most stable winner are
# considered tied on stability and are then separated by silhouette, then
# by confidence dispersion (informativeness), then by mean confidence.
STABILITY_TOLERANCE = 0.05

# Rare-modality merging (per dimension flag below)
RARE_MODALITY_MERGE_THRESHOLD = 0.01


# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================
# Human-readable labels for the LDOS-CoMoDa contextual codes, used only
# in figures and tables. VERIFY against the dataset documentation; any
# code without a label falls back to its raw integer value.
MODALITY_LABELS = {
    "time": {1: "Morning", 2: "Afternoon", 3: "Evening", 4: "Night"},
    "daytype": {1: "Working day", 2: "Weekend", 3: "Holiday"},
    "season": {1: "Spring", 2: "Summer", 3: "Autumn", 4: "Winter"},
    "location": {1: "Home", 2: "Public place", 3: "Friend's house"},
    "weather": {1: "Sunny/clear", 2: "Rainy", 3: "Stormy", 4: "Snowy", 5: "Cloudy"},
    "mood": {1: "Positive", 2: "Neutral", 3: "Negative"},
    "physical": {1: "Healthy", 2: "Ill"},
    "dominantEmo": {1: "Sad", 2: "Happy", 3: "Scared", 4: "Surprised", 5: "Angry", 6: "Disgusted", 7: "Neutral"},
    "endEmo": {1: "Sad", 2: "Happy", 3: "Scared", 4: "Surprised", 5: "Angry", 6: "Disgusted", 7: "Neutral"},
    "social": {1: "Alone", 2: "Partner", 3: "Friends", 4: "Colleagues", 5: "Parents", 6: "Public", 7: "Family"},
    "decision": {1: "User decided", 2: "Given a movie"},
    "interaction": {1: "First", 2: "Repeated"},
}
RARE_MERGED_CODE = -999999
RARE_MERGED_LABEL = "Rare (merged)"

DIMENSION_DISPLAY_NAMES = {
    "temporal": "Temporal",
    "spatial_environmental": "Spatial-environmental",
    "affective_physical": "Affective-physical",
    "social": "Social-situational",
}
FAMILY_DISPLAY_NAMES = {
    "fuzzy_k_modes": "Fuzzy K-Modes",
    "fuzzy_k_medoids_gower": "Fuzzy K-Medoids (Gower)",
    "latent_class_analysis": "Latent Class Analysis",
    "spectral_lca_hybrid": "Spectral-initialized LCA",
    "gmm_mca": "GMM on MCA embedding",
    "modality_identity": "Modality identity (single variable)",
}

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "pdf")
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.constrained_layout.use": True,
})


# =============================================================================
# DATASET STRUCTURE
# =============================================================================

EXPECTED_COLUMNS = [
    "userID", "itemID", "rating", "age", "sex", "city", "country",
    "time", "daytype", "season", "location", "weather", "social",
    "endEmo", "dominantEmo", "mood", "physical", "decision",
    "interaction", "director", "movieCountry", "movieLanguage",
    "movieYear", "genre1", "genre2", "genre3", "actor1", "actor2",
    "actor3", "budget",
]

ID_COLUMNS = ["userID", "itemID"]
TARGET_COLUMN = "rating"
USER_STATIC_COLUMNS = ["age", "sex", "city", "country"]
ALL_CONTEXT_COLUMNS = [
    "time", "daytype", "season", "location", "weather", "social",
    "endEmo", "dominantEmo", "mood", "physical", "decision", "interaction",
]
ITEM_METADATA_COLUMNS = [
    "director", "movieCountry", "movieLanguage", "movieYear",
    "genre1", "genre2", "genre3", "actor1", "actor2", "actor3", "budget",
]
UNUSED_CONTEXT_COLUMNS = ["social", "decision", "interaction"]
NUMERIC_FEATURE_COLUMNS = ["age", "movieYear", "budget"]
CATEGORICAL_FEATURE_COLUMNS = [
    c for c in EXPECTED_COLUMNS
    if c not in (ID_COLUMNS + [TARGET_COLUMN] + NUMERIC_FEATURE_COLUMNS)
]

MISSING_SENTINEL = -1
VALID_RATING_MIN, VALID_RATING_MAX = 1, 5
VALID_AGE_MIN, VALID_AGE_MAX = 10, 100


# =============================================================================
# CONTEXTUAL DIMENSIONS (only "features" is a constraint of the approach)
# =============================================================================

CONTEXT_DIMENSIONS = {
    "temporal": {
        "features": ["time", "daytype", "season"],
        "merge_rare_modalities": False,
        "description": "Temporal context (three low-cardinality nominal variables).",
    },
    "spatial_environmental": {
        "features": ["location", "weather"],
        "merge_rare_modalities": True,
        "description": "Spatial-environmental context (two nominal variables).",
    },
    "affective_physical": {
        "features": ["mood", "dominantEmo", "endEmo", "physical"],
        "merge_rare_modalities": False,
        "description": "Affective-physical context (ordinal mood + nominal emotions/physical).",
    },
}


# =============================================================================
# INPUT / READING / CLEANING
# =============================================================================

def find_input_csv(script_dir: Path, explicit_input: str | None) -> Path:
    if explicit_input:
        p = Path(explicit_input)
        if not p.is_absolute():
            p = script_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        return p
    preferred = script_dir / DEFAULT_INPUT_FILENAME
    if preferred.exists():
        return preferred
    excluded = ["_clean", "_dimension_", "_preprocessed", "_clustering",
                "_membership", "_metrics", "_profile", "_summary"]
    candidates = [p for p in script_dir.glob("LDOS-CoMoDa*.csv")
                  if not any(t in p.name.lower() for t in excluded)]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No LDOS-CoMoDa CSV source file was found in {script_dir}. This script runs the LDOS-CoMoDa "
            "pipeline: place LDOS-CoMoDa.csv next to it or pass --input <file>. For the other datasets use "
            "depaulmovie_full_pipeline.py (ratings.txt) or frappe_full_pipeline.py (frappe.csv).")
    names = "\n  - ".join(p.name for p in sorted(candidates))
    raise RuntimeError(f"Several possible source files found. Use --input:\n  - {names}")


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra = sorted(set(df.columns) - set(EXPECTED_COLUMNS))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if extra:
        print(f"WARNING: unexpected columns found and preserved: {extra}")
    return df


def clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df_raw.copy()
    report: dict = {"rows_initial": int(len(df)), "columns_initial": int(df.shape[1])}

    for c in EXPECTED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    essential_missing = df[ID_COLUMNS + [TARGET_COLUMN]].isna().any(axis=1)
    report["rows_removed_missing_user_item_rating"] = int(essential_missing.sum())
    df = df.loc[~essential_missing].copy()

    invalid_rating = ~df[TARGET_COLUMN].between(VALID_RATING_MIN, VALID_RATING_MAX)
    report["rows_removed_invalid_rating"] = int(invalid_rating.sum())
    df = df.loc[~invalid_rating].copy()

    feature_cols = [c for c in df.columns if c not in ID_COLUMNS + [TARGET_COLUMN]]
    sentinel_counts = (df[feature_cols] == MISSING_SENTINEL).sum()
    report["missing_sentinel_minus1_total"] = int(sentinel_counts.sum())
    report["missing_sentinel_minus1_by_column"] = {
        c: int(v) for c, v in sentinel_counts.items() if int(v) > 0
    }
    df[feature_cols] = df[feature_cols].astype("float64").replace(MISSING_SENTINEL, np.nan)

    invalid_age = df["age"].notna() & ~df["age"].between(VALID_AGE_MIN, VALID_AGE_MAX)
    report["age_values_set_missing_as_invalid"] = int(invalid_age.sum())
    if invalid_age.any():
        report["invalid_age_values"] = sorted(float(v) for v in df.loc[invalid_age, "age"].unique())
        df.loc[invalid_age, "age"] = np.nan

    dup = df.duplicated(keep="first")
    report["exact_duplicate_rows_removed"] = int(dup.sum())
    df = df.loc[~dup].copy().reset_index(drop=True)
    df.insert(0, "observationID", np.arange(len(df), dtype=int))

    for c in [c for c in EXPECTED_COLUMNS if c != "budget"]:
        df[c] = pd.array(df[c], dtype="Int64")
    df["budget"] = pd.array(df["budget"], dtype="Float64")

    report["rows_final"] = int(len(df))
    report["users"] = int(df["userID"].nunique())
    report["items"] = int(df["itemID"].nunique())
    report["rating_distribution"] = {
        str(int(r)): int(n) for r, n in df[TARGET_COLUMN].value_counts().sort_index().items()
    }
    missing_after = df.isna().sum()
    report["missing_after_cleaning_by_column"] = {
        c: int(v) for c, v in missing_after.items() if int(v) > 0
    }
    pair_sizes = df.groupby(["userID", "itemID"], dropna=False).size()
    report["repeated_user_item_pairs_preserved"] = int((pair_sizes > 1).sum())
    report["maximum_interactions_per_user_item_pair"] = int(pair_sizes.max())
    distinct = df.groupby(["userID", "itemID"], dropna=False)[TARGET_COLUMN].nunique()
    report["user_item_pairs_with_multiple_distinct_ratings"] = int((distinct > 1).sum())
    report["users_with_inconsistent_static_attributes"] = {
        c: int((df.groupby("userID")[c].nunique(dropna=False) > 1).sum())
        for c in USER_STATIC_COLUMNS
    }
    return df, report


# =============================================================================
# DIMENSION PREPARATION
# =============================================================================

def merge_rare_categories(df: pd.DataFrame, features: list[str], threshold: float) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    report = {}
    for f in features:
        freq = df[f].value_counts(normalize=True, dropna=True)
        rare = freq[freq < threshold].index.tolist()
        if rare:
            df.loc[df[f].isin(rare), f] = -999999
        report[f] = {
            "n_original_modalities": int(freq.shape[0]),
            "n_rare_modalities_merged": int(len(rare)),
            "n_final_modalities": int(df[f].nunique(dropna=True)),
        }
    return df, report


def prepare_context_dimensions(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    prepared, report = {}, {}
    total_rows, total_users = len(df), int(df["userID"].nunique())

    for dim, cfg in CONTEXT_DIMENSIONS.items():
        feats = list(cfg["features"])
        ddf = df[["observationID", "userID", "itemID", TARGET_COLUMN] + feats].copy()
        usable = ddf.loc[ddf[feats].notna().all(axis=1)].copy()

        rare_report = None
        if cfg.get("merge_rare_modalities") and RARE_MODALITY_MERGE_THRESHOLD:
            usable, rare_report = merge_rare_categories(usable, feats, RARE_MODALITY_MERGE_THRESHOLD)

        for f in feats:
            usable[f] = usable[f].astype(int)
        usable.reset_index(drop=True, inplace=True)
        prepared[dim] = usable

        obs_per_user = usable.groupby("userID").size()
        unique_patterns = int(usable[feats].drop_duplicates().shape[0])
        eligible = eligible_families(feats, unique_patterns)

        report[dim] = {
            "features": feats,
            "feature_types": {f: FEATURE_TYPES.get(f, "nominal") for f in feats},
            "eligible_algorithm_families": eligible["eligible"],
            "excluded_algorithm_families": eligible["excluded"],
            "candidate_k": K_CANDIDATES[dim],
            "total_observations": int(total_rows),
            "usable_observations": int(len(usable)),
            "coverage_percent": round(100.0 * len(usable) / total_rows, 2) if total_rows else 0.0,
            "users_with_available_dimension": int(usable["userID"].nunique()),
            "users_without_available_dimension": int(total_users - usable["userID"].nunique()),
            "missing_by_feature": {f: int(v) for f, v in ddf[feats].isna().sum().items()},
            "number_of_modalities": {f: int(usable[f].nunique()) for f in feats},
            "rare_modality_merge": rare_report,
            "unique_contextual_patterns": unique_patterns,
            "minimum_observations_per_user": int(obs_per_user.min()) if len(obs_per_user) else 0,
            "mean_observations_per_user": round(float(obs_per_user.mean()), 2) if len(obs_per_user) else 0.0,
            "median_observations_per_user": round(float(obs_per_user.median()), 2) if len(obs_per_user) else 0.0,
            "maximum_observations_per_user": int(obs_per_user.max()) if len(obs_per_user) else 0,
        }
    return prepared, report


def eligible_families(features: list[str], unique_patterns: int) -> dict:
    """Family eligibility derived from the nature of the variables."""
    types = [FEATURE_TYPES.get(f, "nominal") for f in features]
    all_nominal = all(t == "nominal" for t in types)
    eligible, excluded = [], {}

    if len(features) == 1:
        return {"eligible": ["modality_identity"],
                "excluded": {fam: "single-variable dimension: clusters are the modalities themselves"
                             for fam in ALGORITHM_FAMILIES if fam != "modality_identity"}}

    for fam in ALGORITHM_FAMILIES:
        if fam == "modality_identity":
            excluded[fam] = "only for single-variable dimensions"
            continue
        if fam == "fuzzy_k_modes":
            if all_nominal:
                eligible.append(fam)
            else:
                excluded[fam] = "matching distance ignores ordinal structure of at least one feature"
        elif fam == "gmm_mca":
            if unique_patterns >= MIN_UNIQUE_PATTERNS_FOR_EMBEDDING:
                eligible.append(fam)
            else:
                excluded[fam] = (
                    f"only {unique_patterns} unique patterns (< {MIN_UNIQUE_PATTERNS_FOR_EMBEDDING}): "
                    "MCA embedding would consist of duplicated point masses"
                )
        else:
            eligible.append(fam)
    return {"eligible": eligible, "excluded": excluded}


# =============================================================================
# DISTANCES AND ENCODINGS
# =============================================================================

def mixed_gower_distance_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """
    Gower dissimilarity handling nominal (mismatch) and ordinal
    (normalized absolute rank difference) features, averaged over
    features. Bounded in [0, 1].
    """
    n = len(df)
    total = np.zeros((n, n), dtype=float)
    for f in features:
        values = df[f].to_numpy()
        if FEATURE_TYPES.get(f, "nominal") == "ordinal":
            ranks = pd.Series(values).rank(method="dense").to_numpy(dtype=float)
            span = ranks.max() - ranks.min()
            col = np.abs(ranks[:, None] - ranks[None, :]) / span if span > 0 else np.zeros((n, n))
        else:
            col = (values[:, None] != values[None, :]).astype(float)
        total += col
    return total / len(features)


def representative_distance(rep_a: np.ndarray, rep_b: np.ndarray, features: list[str], df: pd.DataFrame) -> float:
    d = 0.0
    for j, f in enumerate(features):
        if FEATURE_TYPES.get(f, "nominal") == "ordinal":
            vals = np.sort(df[f].unique())
            span = len(vals) - 1
            ra = np.searchsorted(vals, rep_a[j]); rb = np.searchsorted(vals, rep_b[j])
            d += abs(ra - rb) / span if span > 0 else 0.0
        else:
            d += float(rep_a[j] != rep_b[j])
    return d / len(features)


def categorical_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return df[features].to_numpy(dtype=int)


def one_hot_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    data = df[features].astype(str)
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    return enc.fit_transform(data)


def mca_embedding(x: np.ndarray, n_components: int) -> np.ndarray:
    indicator = pd.get_dummies(pd.DataFrame(x.astype(str))).to_numpy(dtype=float)
    p = indicator / indicator.sum()
    r, c = p.sum(axis=1), p.sum(axis=0)
    expected = np.outer(r, c)
    s = (p - expected) / np.sqrt(expected + 1e-12)
    k = max(1, int(min(n_components, s.shape[0] - 1, s.shape[1] - 1)))
    u, sv, _ = np.linalg.svd(s, full_matrices=False)
    return (u[:, :k] * sv[:k]) / np.sqrt(np.clip(r, 1e-12, None))[:, None]


# =============================================================================
# MEMBERSHIP UTILITIES
# =============================================================================

def memberships_from_dissimilarities(distances: np.ndarray, fuzziness: float) -> np.ndarray:
    if fuzziness <= 1.0:
        raise ValueError("fuzziness must be > 1")
    n, k = distances.shape
    m = np.zeros((n, k))
    expo = 1.0 / (fuzziness - 1.0)
    for i in range(n):
        row = distances[i].astype(float)
        zero = np.isclose(row, 0.0)
        if zero.any():
            m[i, zero] = 1.0 / zero.sum()
            continue
        inv = row ** (-expo)
        m[i] = inv / inv.sum()
    return m


def per_observation_entropy(memberships: np.ndarray) -> np.ndarray:
    k = memberships.shape[1]
    if k <= 1:
        return np.zeros(memberships.shape[0])
    safe = np.clip(memberships, 1e-15, 1.0)
    return -np.sum(safe * np.log(safe), axis=1) / np.log(k)


def fuzzy_partition_coefficient(memberships: np.ndarray) -> float:
    return float(np.mean(np.sum(memberships ** 2, axis=1)))


def unique_rows_with_indices(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = pd.DataFrame(x)
    mask = ~tmp.duplicated(keep="first")
    return tmp.loc[mask].to_numpy(), np.flatnonzero(mask.to_numpy())


# =============================================================================
# FAMILY 1 - FUZZY K-MODES (nominal only)
# =============================================================================

def matching_distance_to_modes(x: np.ndarray, modes: np.ndarray) -> np.ndarray:
    d = np.zeros((x.shape[0], modes.shape[0]))
    for c in range(modes.shape[0]):
        d[:, c] = np.sum(x != modes[c], axis=1)
    return d


def fit_fuzzy_k_modes(x, k, fuzziness, n_init, max_iter, tol, random_state) -> dict:
    n, p = x.shape
    uniq, _ = unique_rows_with_indices(x)
    if k > len(uniq):
        raise ValueError(f"K={k} exceeds unique patterns ({len(uniq)})")
    best, best_obj = None, np.inf
    for init in range(n_init):
        rng = np.random.default_rng(random_state + init)
        modes = uniq[rng.choice(len(uniq), size=k, replace=False)].copy()
        prev = np.inf
        for it in range(max_iter):
            mem = memberships_from_dissimilarities(matching_distance_to_modes(x, modes), fuzziness)
            w = mem ** fuzziness
            new_modes = np.empty((k, p), dtype=x.dtype)
            for c in range(k):
                for j in range(p):
                    vals = np.unique(x[:, j])
                    counts = np.array([w[x[:, j] == v, c].sum() for v in vals])
                    new_modes[c, j] = vals[int(np.argmax(counts))]
            d = matching_distance_to_modes(x, new_modes)
            mem = memberships_from_dissimilarities(d, fuzziness)
            obj = float(np.sum((mem ** fuzziness) * d))
            modes = new_modes
            if np.isfinite(prev) and abs(prev - obj) < tol:
                break
            prev = obj
        if obj < best_obj:
            best_obj = obj
            best = {"memberships": mem.copy(), "labels": np.argmax(mem, axis=1),
                    "modes": modes.copy(), "objective": obj, "iterations": it + 1}
    return best


# =============================================================================
# FAMILY 2 - FUZZY K-MEDOIDS ON MIXED GOWER
# =============================================================================

def fit_fuzzy_k_medoids(distance_matrix, x, k, fuzziness, n_init, max_iter, tol, random_state) -> dict:
    _, uniq_idx = unique_rows_with_indices(x)
    if k > len(uniq_idx):
        raise ValueError(f"K={k} exceeds unique patterns ({len(uniq_idx)})")
    best, best_obj = None, np.inf
    for init in range(n_init):
        rng = np.random.default_rng(random_state + init)
        medoids = rng.choice(uniq_idx, size=k, replace=False).astype(int)
        prev = np.inf
        for it in range(max_iter):
            mem = memberships_from_dissimilarities(distance_matrix[:, medoids], fuzziness)
            w = mem ** fuzziness
            new_medoids, used = [], set()
            for c in range(k):
                costs = distance_matrix @ w[:, c]
                order = uniq_idx[np.argsort(costs[uniq_idx])]
                sel = next((int(i) for i in order if int(i) not in used), None)
                if sel is None:
                    raise RuntimeError("Unable to select distinct medoids")
                used.add(sel); new_medoids.append(sel)
            new_medoids = np.asarray(new_medoids)
            d = distance_matrix[:, new_medoids]
            mem = memberships_from_dissimilarities(d, fuzziness)
            obj = float(np.sum((mem ** fuzziness) * d))
            unchanged = np.array_equal(np.sort(medoids), np.sort(new_medoids))
            medoids = new_medoids
            if unchanged or (np.isfinite(prev) and abs(prev - obj) < tol):
                break
            prev = obj
        if obj < best_obj:
            best_obj = obj
            best = {"memberships": mem.copy(), "labels": np.argmax(mem, axis=1),
                    "medoid_indices": medoids.copy(), "medoids": x[medoids].copy(),
                    "objective": obj, "iterations": it + 1}
    return best


# =============================================================================
# FAMILY 3 - LATENT CLASS ANALYSIS (random or label-based initialization)
# =============================================================================

def encode_lca_features(x: np.ndarray) -> tuple[np.ndarray, list[list[str]]]:
    enc = np.zeros(x.shape, dtype=int)
    cats = []
    for j in range(x.shape[1]):
        vals, inv = np.unique(x[:, j], return_inverse=True)
        enc[:, j] = inv
        cats.append([str(v) for v in vals])
    return enc, cats


def logsumexp_numpy(v: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(v, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(v - m), axis=axis, keepdims=True)), axis=axis)


def lca_em(x_enc, cats, k, memberships, max_iter, tol, smoothing) -> dict:
    n, p = x_enc.shape
    counts = [len(c) for c in cats]
    prev_ll = -np.inf
    for it in range(max_iter):
        priors = (memberships.sum(axis=0) + smoothing) / (n + k * smoothing)
        cond = []
        for j in range(p):
            theta = np.zeros((k, counts[j]))
            for cat in range(counts[j]):
                mask = x_enc[:, j] == cat
                theta[:, cat] = memberships[mask].sum(axis=0) + smoothing
            theta /= theta.sum(axis=1, keepdims=True)
            cond.append(theta)
        logp = np.tile(np.log(np.clip(priors, 1e-15, 1)), (n, 1))
        for j in range(p):
            logp += np.log(np.clip(cond[j][:, x_enc[:, j]].T, 1e-15, 1))
        lognorm = logsumexp_numpy(logp, axis=1)
        ll = float(lognorm.sum())
        memberships = np.exp(logp - lognorm[:, None])
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    n_params = (k - 1) + sum(k * (c - 1) for c in counts)
    bic = np.log(n) * n_params - 2 * ll
    aic = 2 * n_params - 2 * ll
    ent = -np.sum(np.clip(memberships, 1e-15, 1) * np.log(np.clip(memberships, 1e-15, 1)))
    return {"memberships": memberships, "labels": np.argmax(memberships, axis=1),
            "priors": priors, "conditional_probabilities": cond, "category_values": cats,
            "log_likelihood": ll, "number_parameters": int(n_params),
            "aic": float(aic), "bic": float(bic), "icl": float(bic + 2 * ent), "iterations": it + 1}


def fit_lca(x, k, n_init, max_iter, tol, random_state, initial_labels_list=None, smoothing=LCA_SMOOTHING) -> dict:
    x_enc, cats = encode_lca_features(x)
    n = x_enc.shape[0]
    best = None
    inits = initial_labels_list if initial_labels_list is not None else [None] * n_init
    for i, labels in enumerate(inits):
        rng = np.random.default_rng(random_state + i)
        if labels is None:
            mem = rng.random((n, k)); mem /= mem.sum(axis=1, keepdims=True)
        else:
            eps = 0.05
            mem = np.full((n, k), eps / k)
            mem[np.arange(n), labels] += 1.0 - eps
        res = lca_em(x_enc, cats, k, mem, max_iter, tol, smoothing)
        if best is None or res["log_likelihood"] > best["log_likelihood"]:
            best = res
    return best


# =============================================================================
# FAMILY 4 - SPECTRAL -> LCA HYBRID
# =============================================================================

def spectral_labels(distance_matrix: np.ndarray, k: int, random_state: int) -> np.ndarray:
    affinity = 1.0 - distance_matrix
    np.fill_diagonal(affinity, 1.0)
    model = SpectralClustering(n_clusters=k, affinity="precomputed",
                               random_state=random_state, assign_labels="kmeans")
    return model.fit_predict(affinity)


def fit_spectral_lca_hybrid(distance_matrix, x, k, n_init, max_iter, tol, random_state) -> dict:
    """Stable, balanced spectral partition used to initialize LCA, whose
    posterior provides genuinely graded probabilities."""
    label_sets = []
    for i in range(max(1, n_init // 3)):
        try:
            label_sets.append(spectral_labels(distance_matrix, k, random_state + i))
        except Exception:
            continue
    if not label_sets:
        raise RuntimeError("Spectral initialization failed")
    res = fit_lca(x, k, len(label_sets), max_iter, tol, random_state, initial_labels_list=label_sets)
    res["spectral_init_labels"] = label_sets[0]
    return res


# =============================================================================
# FAMILY 5 - GMM ON MCA EMBEDDING (high-cardinality dimensions only)
# =============================================================================

def fit_gmm_mca(x, k, n_components, covariance_type, n_init, random_state) -> dict:
    emb = mca_embedding(x, n_components)
    reg = max(1e-4, GMM_REG_COVAR_FRACTION * float(np.mean(np.var(emb, axis=0))))
    model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                            n_init=max(1, n_init // 3), max_iter=MAX_ITER, tol=TOL,
                            random_state=random_state, reg_covar=reg).fit(emb)
    mem = model.predict_proba(emb)
    mem = mem / np.clip(mem.sum(axis=1, keepdims=True), 1e-15, None)
    ent = -np.sum(np.clip(mem, 1e-15, 1) * np.log(np.clip(mem, 1e-15, 1)))
    bic = float(model.bic(emb))
    return {"memberships": mem, "labels": np.argmax(mem, axis=1), "means": model.means_.copy(),
            "n_components_mca": emb.shape[1], "reg_covar": reg,
            "log_likelihood": float(model.score(emb) * emb.shape[0]),
            "aic": float(model.aic(emb)), "bic": bic, "icl": float(bic + 2 * ent),
            "iterations": int(model.n_iter_)}


# =============================================================================
# UNIFIED DISPATCHER
# =============================================================================

def fit_modality_identity(x: np.ndarray) -> dict:
    values = np.sort(np.unique(x[:, 0]))
    labels = np.searchsorted(values, x[:, 0])
    mem = np.zeros((len(x), len(values)))
    mem[np.arange(len(x)), labels] = 1.0
    return {"memberships": mem, "labels": labels, "modes": values.reshape(-1, 1), "iterations": 1}


def fit_candidate(family, df, features, k, hp, random_state, distance_matrix=None) -> dict:
    x = categorical_matrix(df, features)
    n_init = hp.get("n_init", N_INIT)
    if family == "modality_identity":
        if len(features) != 1:
            raise ValueError("modality_identity requires a single feature")
        model = fit_modality_identity(x)
        if k is not None and k != model["memberships"].shape[1]:
            raise ValueError(f"modality_identity has K fixed to the number of modalities ({model['memberships'].shape[1]})")
        return model
    if distance_matrix is None and family in ("fuzzy_k_medoids_gower", "spectral_lca_hybrid"):
        distance_matrix = mixed_gower_distance_matrix(df, features)

    if family == "fuzzy_k_modes":
        return fit_fuzzy_k_modes(x, k, hp["fuzziness"], n_init, MAX_ITER, TOL, random_state)
    if family == "fuzzy_k_medoids_gower":
        return fit_fuzzy_k_medoids(distance_matrix, x, k, hp["fuzziness"], n_init, MAX_ITER, TOL, random_state)
    if family == "latent_class_analysis":
        return fit_lca(x, k, n_init, MAX_ITER, TOL, random_state)
    if family == "spectral_lca_hybrid":
        return fit_spectral_lca_hybrid(distance_matrix, x, k, n_init, MAX_ITER, TOL, random_state)
    if family == "gmm_mca":
        return fit_gmm_mca(x, k, hp["n_components"], hp["covariance_type"], n_init, random_state)
    raise ValueError(f"Unsupported family: {family}")


def hyperparameter_grid(family: str) -> list[dict]:
    if family == "modality_identity":
        return [{}]
    if family in ("fuzzy_k_modes", "fuzzy_k_medoids_gower"):
        return [{"fuzziness": f} for f in FUZZINESS_GRID]
    if family == "gmm_mca":
        return [{"n_components": nc, "covariance_type": cov}
                for nc in MCA_COMPONENTS_GRID for cov in GMM_COVARIANCE_GRID]
    return [{}]


def native_criterion(family: str) -> tuple[str, bool]:
    """(metric name, higher_is_better) used for stage-1 selection."""
    if family in ("latent_class_analysis", "spectral_lca_hybrid", "gmm_mca"):
        return "icl", False
    return "silhouette", True


# =============================================================================
# METRICS AND STABILITY
# =============================================================================

def mean_pairwise_ari(label_sets: list[np.ndarray]) -> float:
    if len(label_sets) < 2:
        return np.nan
    return float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]))


def evaluate_partition(labels, distance_matrix, one_hot) -> dict:
    k, n = len(np.unique(labels)), len(labels)
    out = {"effective_clusters": int(k), "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if k < 2 or k >= n:
        return out
    for key, fn in (("silhouette", lambda: silhouette_score(distance_matrix, labels, metric="precomputed")),
                    ("davies_bouldin", lambda: davies_bouldin_score(one_hot, labels)),
                    ("calinski_harabasz", lambda: calinski_harabasz_score(one_hot, labels))):
        try:
            out[key] = float(fn())
        except Exception:
            pass
    return out


def cluster_size_statistics(labels) -> dict:
    u, c = np.unique(labels, return_counts=True)
    return {"minimum_cluster_size": int(c.min()), "maximum_cluster_size": int(c.max()),
            "smallest_cluster_fraction": float(c.min() / c.sum()),
            "cluster_sizes": json.dumps({str(int(l) + 1): int(n) for l, n in zip(u, c)})}


def mean_obs_per_user_per_cluster(labels, user_ids, k) -> float:
    n_users = pd.Series(user_ids).nunique()
    return float(len(labels) / (n_users * k)) if n_users and k else 0.0


def compute_init_stability(family, df, features, k, hp, seed, distance_matrix) -> float:
    sets = []
    for r in range(INIT_STABILITY_RUNS):
        hp1 = dict(hp); hp1["n_init"] = 1
        try:
            sets.append(fit_candidate(family, df, features, k, hp1, seed + 20000 + r, distance_matrix)["labels"])
        except Exception:
            continue
    return mean_pairwise_ari(sets)


def compute_bootstrap_stability(family, df, features, k, hp, seed, distance_matrix) -> float:
    n = len(df)
    size = max(int(BOOTSTRAP_FRACTION * n), 2 * k)
    sets, idxs = [], []
    for r in range(BOOTSTRAP_RUNS):
        rng = np.random.default_rng(seed + 30000 + r)
        idx = np.sort(rng.choice(n, size=size, replace=False))
        sub = df.iloc[idx].reset_index(drop=True)
        subd = distance_matrix[np.ix_(idx, idx)] if distance_matrix is not None else None
        hp1 = dict(hp); hp1["n_init"] = max(1, N_INIT // 3)
        try:
            sets.append(fit_candidate(family, sub, features, k, hp1, seed + 30000 + r, subd)["labels"])
            idxs.append(idx)
        except Exception:
            continue
    scores = []
    for (la, ia), (lb, ib) in combinations(zip(sets, idxs), 2):
        common = np.intersect1d(ia, ib)
        if len(common) < max(10, 2 * k):
            continue
        scores.append(adjusted_rand_score(la[np.searchsorted(ia, common)], lb[np.searchsorted(ib, common)]))
    return float(np.mean(scores)) if scores else np.nan


# =============================================================================
# ADMISSIBILITY
# =============================================================================

def evaluate_admissibility(m: dict, k: int, max_k: int, min_bootstrap: float) -> list[str]:
    fails = []
    crisp_by_construction = m.get("family") == "modality_identity"
    if k < MIN_CLUSTERS:
        fails.append(f"k_below_minimum({k}<{MIN_CLUSTERS})")
    if k > max_k:
        fails.append(f"k_above_user_based_ceiling({k}>{max_k})")
    if m["smallest_cluster_fraction"] < MIN_CLUSTER_FRACTION:
        fails.append(f"smallest_cluster_fraction({m['smallest_cluster_fraction']:.4f}<{MIN_CLUSTER_FRACTION})")
    if m["mean_obs_per_user_per_cluster"] < MIN_MEAN_OBS_PER_USER_PER_CLUSTER:
        fails.append(f"mean_obs_per_user_per_cluster({m['mean_obs_per_user_per_cluster']:.2f}<{MIN_MEAN_OBS_PER_USER_PER_CLUSTER})")
    if not crisp_by_construction:
        if m["membership_confidence"] < MIN_MEMBERSHIP_CONFIDENCE:
            fails.append(f"confidence_too_low({m['membership_confidence']:.3f}<{MIN_MEMBERSHIP_CONFIDENCE})")
        if m["membership_confidence"] > MAX_MEMBERSHIP_CONFIDENCE:
            fails.append(f"confidence_near_crisp({m['membership_confidence']:.3f}>{MAX_MEMBERSHIP_CONFIDENCE})")
        if m["confidence_dispersion"] < MIN_CONFIDENCE_DISPERSION:
            fails.append(f"confidence_not_informative(std={m['confidence_dispersion']:.3f}<{MIN_CONFIDENCE_DISPERSION})")
    if not pd.isna(m["bootstrap_stability_ari"]) and m["bootstrap_stability_ari"] < min_bootstrap:
        fails.append(f"bootstrap_stability({m['bootstrap_stability_ari']:.3f}<{min_bootstrap})")
    return fails


def diagnostic_notes(row: pd.Series, k_grid: list[int]) -> str:
    notes = []
    if row["k"] == max(k_grid):
        notes.append("selected_K_at_upper_search_boundary")
    if row["stability_combined"] < 0.5:
        notes.append("low_combined_stability")
    if row["silhouette"] < 0.3:
        notes.append("weak_geometric_separation")
    if row["membership_confidence"] < 0.5:
        notes.append("moderately_diffuse_memberships")
    if row["smallest_cluster_fraction"] < MIN_CLUSTER_FRACTION + 0.01:
        notes.append("near_minimum_cluster_fraction")
    return "; ".join(notes) if notes else "no_major_diagnostic_flag"


def _evaluate_clustering_candidate(args):
    """Fits one clustering candidate and computes its metrics, including
    initialisation and bootstrap stability. Independent across candidates."""
    dimension, fam, hp, k, df, features, distance_matrix, one_hot, user_ids, max_k = args
    seed = RANDOM_STATE + 1000 * k + (sum(ord(c) for c in fam) % 97)   # deterministic across processes
    try:
        model = fit_candidate(fam, df, features, k, hp, seed, distance_matrix)
    except Exception as exc:
        print(f"  [skip] {fam} k={k} {hp}: {exc}")
        return None
    labels, mem = model["labels"], model["memberships"]
    if not np.allclose(mem.sum(axis=1), 1.0, atol=1e-6):
        print(f"  [skip] {fam} k={k}: memberships not normalized"); return None
    eff_k = int(len(np.unique(labels)))
    if eff_k < 2:
        return None

    m = evaluate_partition(labels, distance_matrix, one_hot)
    m.update(cluster_size_statistics(labels))
    ent = per_observation_entropy(mem)
    m["membership_entropy"] = float(ent.mean())
    m["membership_confidence"] = float(1.0 - ent.mean())
    m["confidence_dispersion"] = float(ent.std())
    m["mean_obs_per_user_per_cluster"] = mean_obs_per_user_per_cluster(labels, user_ids, eff_k)
    if fam == "modality_identity":
        m["init_stability_ari"] = 1.0; m["bootstrap_stability_ari"] = 1.0
    else:
        m["init_stability_ari"] = compute_init_stability(fam, df, features, k, hp, seed, distance_matrix)
        m["bootstrap_stability_ari"] = compute_bootstrap_stability(fam, df, features, k, hp, seed, distance_matrix)
    m["stability_combined"] = float(np.nanmean([m["init_stability_ari"], m["bootstrap_stability_ari"]]))
    for key in ("log_likelihood", "aic", "bic", "icl"):
        m[key] = float(model[key]) if key in model else np.nan
    m["fuzzy_partition_coefficient"] = (fuzzy_partition_coefficient(mem)
                                        if fam in ("fuzzy_k_modes", "fuzzy_k_medoids_gower") else np.nan)
    m["family"] = fam
    m["hyperparams"] = json.dumps(hp)
    m["k"] = eff_k
    m["k_requested"] = k
    m["max_admissible_clusters"] = max_k
    return m, model


# =============================================================================
# TWO-STAGE SEARCH FOR ONE DIMENSION
# =============================================================================

def run_dimension_search(dimension: str, df: pd.DataFrame, dim_report: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    features = list(CONTEXT_DIMENSIONS[dimension]["features"])
    user_ids = df["userID"].to_numpy()
    n_users = int(pd.Series(user_ids).nunique())
    max_k = max(MIN_CLUSTERS, int(MAX_CLUSTERS_FRACTION_OF_USERS * n_users))
    unique_patterns = dim_report["unique_contextual_patterns"]
    families = dim_report["eligible_algorithm_families"]
    k_grid = [k for k in K_CANDIDATES[dimension] if k <= unique_patterns and k <= max_k]

    distance_matrix = mixed_gower_distance_matrix(df, features)
    one_hot = one_hot_matrix(df, features)

    print("\n" + "=" * 100)
    print(f"DIMENSION: {dimension} | features={features} | n={len(df)} | users={n_users} | patterns={unique_patterns}")
    print(f"Eligible families: {families}")
    for fam, why in dim_report["excluded_algorithm_families"].items():
        print(f"  excluded {fam}: {why}")
    print(f"K grid: {k_grid} (user-based ceiling {max_k})")
    print("=" * 100)

    rows, models = [], {}
    specs = []
    for fam in families:
        fam_k_grid = [int(df[features[0]].nunique())] if fam == "modality_identity" else k_grid
        for hp in hyperparameter_grid(fam):
            for k in fam_k_grid:
                specs.append((fam, hp, k))
    with timed("phase1.search", dimension=dimension, n_obs=int(len(df)), n_candidates=len(specs)):
      results = parallel_map(_evaluate_clustering_candidate,
                           [(dimension, fam, hp, k, df, features, distance_matrix, one_hot, user_ids, max_k) for fam, hp, k in specs])
    for (fam, hp, k), res in zip(specs, results):
        if res is None:
            continue
        m, model = res
        eff_k = m["k"]
        rows.append(m)
        models[(fam, json.dumps(hp), k)] = model
        print(f"  {fam:<24} k={eff_k:<2} {str(hp):<22} sil={m['silhouette']:.3f} "
              f"stab(init/boot)={m['init_stability_ari']:.2f}/{m['bootstrap_stability_ari']:.2f} "
              f"conf={m['membership_confidence']:.2f} disp={m['confidence_dispersion']:.3f}"
              + (f" icl={m['icl']:.0f}" if not pd.isna(m['icl']) else ""))
    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError(f"No candidate could be fitted for {dimension}")

    # ---- Admissibility with progressive relaxation of the stability gate
    used_gate = None
    for gate in STABILITY_RELAXATION_STEPS:
        fails = [evaluate_admissibility(r, int(r["k"]), max_k, gate) for _, r in all_df.iterrows()]
        all_df["admissibility_failures"] = ["none" if not f else "; ".join(f) for f in fails]
        all_df["admissible"] = [len(f) == 0 for f in fails]
        if all_df["admissible"].any():
            used_gate = gate
            break
    if used_gate is None:
        print(f"\nWARNING: no admissible candidate for {dimension}; using full pool")
        all_df["admissible"] = True
        used_gate = 0.0
    all_df["bootstrap_stability_gate_used"] = used_gate
    adm = all_df.loc[all_df["admissible"]].copy()

    # ---- Stage 1: best candidate within each family by native criterion
    winners = []
    for fam, group in adm.groupby("family"):
        metric, higher = native_criterion(fam)
        g = group.dropna(subset=[metric])
        if g.empty:
            continue
        g = g.sort_values([metric, "stability_combined"], ascending=[not higher, False])
        w = g.iloc[0].copy()
        w["stage1_criterion"] = metric
        winners.append(w)
    winners_df = pd.DataFrame(winners).reset_index(drop=True)
    if winners_df.empty:
        raise RuntimeError(f"No family winner for {dimension}")

    # ---- Stage 2: stability first (within tolerance), then silhouette,
    #      then confidence informativeness, then mean confidence
    best_stab = winners_df["stability_combined"].max()
    winners_df["stability_tier"] = (winners_df["stability_combined"] >= best_stab - STABILITY_TOLERANCE).astype(int)
    winners_df = winners_df.sort_values(
        ["stability_tier", "silhouette", "confidence_dispersion", "membership_confidence"],
        ascending=[False, False, False, False]).reset_index(drop=True)
    winners_df["stage2_rank"] = np.arange(1, len(winners_df) + 1)
    best = winners_df.iloc[0]
    best_model = models[(best["family"], best["hyperparams"], int(best["k_requested"]))]

    notes = diagnostic_notes(best, k_grid)
    print("\n" + "-" * 100)
    print("FAMILY WINNERS (stage 1) and stage-2 ranking:")
    print(winners_df[["stage2_rank", "family", "k", "hyperparams", "stage1_criterion", "silhouette",
                      "init_stability_ari", "bootstrap_stability_ari", "membership_confidence",
                      "confidence_dispersion"]].to_string(index=False))
    print(f"\nSELECTED for {dimension}: {best['family']} K={int(best['k'])} {best['hyperparams']}")
    print(f"Bootstrap stability gate used: {used_gate} | Notes: {notes}")
    print("-" * 100)

    def f(v):
        return None if pd.isna(v) else float(v)

    info = {
        "dimension": dimension,
        "features": features,
        "feature_types": dim_report["feature_types"],
        "selected_family": best["family"],
        "selected_hyperparams": json.loads(best["hyperparams"]),
        "selected_k": int(best["k"]),
        "stage1_criterion": best["stage1_criterion"],
        "stage2_rule": ("stability first (tolerance %.2f), then silhouette, then confidence "
                        "dispersion, then mean confidence" % STABILITY_TOLERANCE),
        "bootstrap_stability_gate_used": used_gate,
        "silhouette": f(best["silhouette"]),
        "davies_bouldin": f(best["davies_bouldin"]),
        "calinski_harabasz": f(best["calinski_harabasz"]),
        "init_stability_ari": f(best["init_stability_ari"]),
        "bootstrap_stability_ari": f(best["bootstrap_stability_ari"]),
        "stability_combined": f(best["stability_combined"]),
        "membership_entropy": f(best["membership_entropy"]),
        "membership_confidence": f(best["membership_confidence"]),
        "confidence_dispersion": f(best["confidence_dispersion"]),
        "mean_obs_per_user_per_cluster": f(best["mean_obs_per_user_per_cluster"]),
        "minimum_cluster_size": int(best["minimum_cluster_size"]),
        "maximum_cluster_size": int(best["maximum_cluster_size"]),
        "smallest_cluster_fraction": f(best["smallest_cluster_fraction"]),
        "cluster_sizes": json.loads(best["cluster_sizes"]),
        "aic": f(best["aic"]), "bic": f(best["bic"]), "icl": f(best["icl"]),
        "fuzzy_partition_coefficient": f(best["fuzzy_partition_coefficient"]),
        "quality_notes": notes,
        "eligible_families": families,
        "excluded_families": dim_report["excluded_algorithm_families"],
        "k_grid": k_grid,
        "admissibility_constraints": {
            "min_clusters": MIN_CLUSTERS, "max_clusters_fraction_of_users": MAX_CLUSTERS_FRACTION_OF_USERS,
            "max_admissible_clusters": max_k, "min_cluster_fraction": MIN_CLUSTER_FRACTION,
            "min_mean_obs_per_user_per_cluster": MIN_MEAN_OBS_PER_USER_PER_CLUSTER,
            "min_membership_confidence": MIN_MEMBERSHIP_CONFIDENCE,
            "max_membership_confidence": MAX_MEMBERSHIP_CONFIDENCE,
            "min_confidence_dispersion": MIN_CONFIDENCE_DISPERSION,
            "min_bootstrap_stability": MIN_BOOTSTRAP_STABILITY,
        },
        "n_candidates_evaluated": int(len(all_df)),
        "n_admissible_candidates": int(len(adm)),
        "important_note": ("Exploratory full-dataset selection. For the final evaluation, the same "
                           "two-stage search must be repeated on training/validation data only "
                           "within each fold; the test fold must not participate in selection."),
    }
    return all_df, winners_df, best_model, info


# =============================================================================
# REPRESENTATIVE DISTANCE MATRIX (phase-2 preparation)
# =============================================================================

def build_representative_distance_matrix(family, model, df, features) -> np.ndarray | None:
    if family in ("fuzzy_k_modes", "modality_identity"):
        reps = model["modes"]
        d = np.array([[representative_distance(a, b, features, df) for b in reps] for a in reps])
    elif family == "fuzzy_k_medoids_gower":
        reps = model["medoids"]
        d = np.array([[representative_distance(a, b, features, df) for b in reps] for a in reps])
    elif family in ("latent_class_analysis", "spectral_lca_hybrid"):
        theta = np.concatenate(model["conditional_probabilities"], axis=1)
        d = np.nan_to_num(cdist(theta, theta, metric="jensenshannon"))
    elif family == "gmm_mca":
        d = pairwise_distances(model["means"])
    else:
        return None
    return d / d.max() if d.max() > 0 else d


# =============================================================================
# VISUALIZATION (publication figures for the clustering section)
# =============================================================================

def modality_label(feature: str, code) -> str:
    code = int(code)
    if code == RARE_MERGED_CODE:
        return RARE_MERGED_LABEL
    return MODALITY_LABELS.get(feature, {}).get(code, str(code))


def save_figure(fig, path_without_ext: Path) -> None:
    for fmt in FIGURE_FORMATS:
        fig.savefig(f"{path_without_ext}.{fmt}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def cluster_modality_profiles(df: pd.DataFrame, features: list[str], memberships: np.ndarray) -> dict[str, pd.DataFrame]:
    """Membership-weighted distribution of modalities within each cluster:
    profiles[feature] is a (clusters x modalities) DataFrame of proportions."""
    k = memberships.shape[1]
    profiles = {}
    for f in features:
        codes = np.sort(df[f].unique())
        mat = np.zeros((k, len(codes)))
        for j, code in enumerate(codes):
            mask = (df[f].to_numpy() == code)
            mat[:, j] = memberships[mask].sum(axis=0)
        mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-12, None)
        profiles[f] = pd.DataFrame(mat, index=[f"C{c + 1}" for c in range(k)],
                                   columns=[modality_label(f, c) for c in codes])
    return profiles


def plot_cluster_profiles(profiles: dict[str, pd.DataFrame], dimension: str, family: str, out: Path) -> None:
    """Membership-weighted contextual-state profiles.

    The contextual-state figures use the same physical size and the same
    0-1 color scale for every Frappe dimension.
    """
    features = list(profiles.keys())

    # Same figure scale for Temporal, Spatial-environmental and Situational.
    fig = plt.figure(figsize=(10.0, 3.8))
    gs = gridspec.GridSpec(
        1, len(features),
        width_ratios=[1.0] * len(features),
        figure=fig,
    )

    im = None
    for i, f in enumerate(features):
        ax = fig.add_subplot(gs[0, i])
        p = profiles[f]
        im = ax.imshow(
            p.to_numpy(),
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.set_xticks(range(p.shape[1]))
        ax.set_xticklabels(p.columns, rotation=45, ha="right")
        ax.set_yticks(range(p.shape[0]))
        ax.set_yticklabels(p.index if i == 0 else [])
        ax.set_title(f)

        for r in range(p.shape[0]):
            for c in range(p.shape[1]):
                v = p.iat[r, c]
                if v >= 0.08:
                    ax.text(
                        c, r, f"{v:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black",
                    )

    fig.colorbar(im, ax=fig.axes, shrink=0.8, label="Within-state proportion")
    n_states = profiles[features[0]].shape[0]

    # "cluster profiles" deliberately removed from the visible title.
    fig.suptitle(
        f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - contextual states "
        f"({FAMILY_DISPLAY_NAMES.get(family, family)}, {n_states} states)"
    )

    save_figure(fig, out / f"{dimension}_cluster_profiles")


def plot_mca_scatter(df: pd.DataFrame, features: list[str], memberships: np.ndarray, dimension: str, out: Path, seed: int = 0) -> None:
    """2-D MCA projection; identical categorical patterns are jittered so
    that the mass of each pattern is visible; colour = hard cluster,
    opacity = membership confidence."""
    x = categorical_matrix(df, features)
    emb = mca_embedding(x, 2)
    if emb.shape[1] < 2:
        emb = np.hstack([emb, np.zeros((emb.shape[0], 1))])
    rng = np.random.default_rng(seed)
    scale = 0.04 * (emb.max(axis=0) - emb.min(axis=0) + 1e-9)
    jitter = rng.normal(0, 1, emb.shape) * scale
    labels = np.argmax(memberships, axis=1)
    conf = 1.0 - per_observation_entropy(memberships)
    k = memberships.shape[1]
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for c in range(k):
        m = labels == c
        ax.scatter(emb[m, 0] + jitter[m, 0], emb[m, 1] + jitter[m, 1], s=9,
                   color=cmap(c), alpha=np.clip(0.15 + 0.75 * conf[m], 0.1, 0.9)[0] if m.sum() else 0.5,
                   label=f"State {c + 1} ({m.sum()} observations)", linewidths=0)
    ax.set_xlabel("MCA component 1"); ax.set_ylabel("MCA component 2")
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - MCA projection (jittered)")
    ax.legend(frameon=False, markerscale=2, loc="best")
    save_figure(fig, out / f"{dimension}_mca_scatter")


def plot_membership_confidence(memberships: np.ndarray, dimension: str, out: Path) -> None:
    """Distribution of per-observation membership confidence, i.e. the
    quantity feeding the reliability term c_u^d in the next phase."""
    conf = 1.0 - per_observation_entropy(memberships)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    safe_hist(ax, conf, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(conf.mean(), color="k", ls="--", lw=1, label=f"mean = {conf.mean():.2f}")
    ax.set_xlabel("Membership confidence"); ax.set_ylabel("Observations")
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - soft-membership confidence")
    ax.legend(frameon=False)
    save_figure(fig, out / f"{dimension}_membership_confidence")


def plot_cluster_sizes(memberships: np.ndarray, dimension: str, out: Path) -> None:
    labels = np.argmax(memberships, axis=1)
    k = memberships.shape[1]
    hard = np.array([(labels == c).sum() for c in range(k)])
    soft = memberships.sum(axis=0)
    xs = np.arange(k)

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.bar(xs - 0.2, hard, width=0.4, label="Hard assignment", color="#4C72B0")
    ax.bar(xs + 0.2, soft, width=0.4, label="Soft membership mass", color="#DD8452")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"State {c + 1}" for c in range(k)])
    ax.set_ylabel("Observations")

    # "cluster sizes" deliberately removed from the title.
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - state distribution")

    ax.legend(frameon=False)
    save_figure(fig, out / f"{dimension}_cluster_sizes")


def plot_model_selection(all_df: pd.DataFrame, family: str, hyperparams: str, selected_k: int, dimension: str, out: Path) -> None:
    """Selection criterion and stability for the retained clustering family."""
    sub = all_df[
        (all_df["family"] == family) &
        (all_df["hyperparams"] == hyperparams)
    ].sort_values("k")
    if sub.empty:
        return

    metric, higher = native_criterion(family)
    fig, ax1 = plt.subplots(figsize=(4.8, 3.6))

    criterion_label = metric.upper() if metric == "icl" else "Silhouette"
    ax1.plot(
        sub["k"], sub[metric], "o-",
        color="#4C72B0",
        label=criterion_label,
    )
    ax1.set_xlabel("Number of contextual states")
    ax1.set_ylabel(criterion_label, color="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(
        sub["k"], sub["init_stability_ari"], "s--",
        color="#55A868",
        label="Initialization stability (ARI)",
    )
    ax2.plot(
        sub["k"], sub["bootstrap_stability_ari"], "^--",
        color="#C44E52",
        label="Bootstrap stability (ARI)",
    )
    ax2.set_ylabel("Stability (ARI)")
    ax2.set_ylim(0, 1.05)

    ax1.axvline(
        selected_k,
        color="k",
        lw=1,
        ls=":",
        label=f"Retained: {selected_k} states",
    )
    ax1.set_xticks(sub["k"].astype(int))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7,
    )

    # Keep dimension and retained technique only.
    ax1.set_title(
        f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - "
        f"{FAMILY_DISPLAY_NAMES.get(family, family)}"
    )

    save_figure(fig, out / f"{dimension}_model_selection")


def plot_family_winners(winners_df: pd.DataFrame, dimension: str, out: Path) -> None:
    """Comparison of family winners on common diagnostics."""
    w = winners_df.sort_values("stage2_rank")
    names = [
        f"{FAMILY_DISPLAY_NAMES.get(f, f)}\n({int(k)} states)"
        for f, k in zip(w["family"], w["k"])
    ]

    xs = np.arange(len(w))
    bw = 0.2 if len(w) > 1 else 0.12
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(w) + 1.8), 3.2))

    ax.bar(xs - 1.5 * bw, w["silhouette"], width=bw, label="Silhouette", color="#4C72B0")
    ax.bar(xs - 0.5 * bw, w["stability_combined"], width=bw, label="Stability (mean ARI)", color="#55A868")
    ax.bar(xs + 0.5 * bw, w["membership_confidence"], width=bw, label="Confidence", color="#DD8452")
    ax.bar(xs + 1.5 * bw, w["confidence_dispersion"], width=bw, label="Confidence dispersion", color="#8172B2")

    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_xlim(-0.6, len(w) - 0.4)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")

    # "(stage 2)" deliberately removed.
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - family winners")

    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
    )
    save_figure(fig, out / f"{dimension}_family_winners")


def plot_representative_distance(rep: np.ndarray, dimension: str, out: Path) -> None:
    k = rep.shape[0]
    fig, ax = plt.subplots(figsize=(0.6 * k + 2.0, 0.6 * k + 1.6))
    im = ax.imshow(rep, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"C{i + 1}" for i in range(k)]); ax.set_yticklabels([f"C{i + 1}" for i in range(k)])
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{rep[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if rep[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Normalized distance")
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dimension, dimension)} - state distance matrix")
    save_figure(fig, out / f"{dimension}_representative_distance")


def generate_dimension_figures(dimension, df, features, model, all_df, winners_df, info, rep, fig_root: Path) -> list[str]:
    out = fig_root / dimension
    out.mkdir(parents=True, exist_ok=True)
    mem = model["memberships"]
    profiles = cluster_modality_profiles(df, features, mem)
    for f, p in profiles.items():
        p.to_csv(out / f"{dimension}_profile_{f}.csv")
    plot_cluster_profiles(profiles, dimension, info["selected_family"], out)
    plot_mca_scatter(df, features, mem, dimension, out)
    plot_membership_confidence(mem, dimension, out)
    plot_cluster_sizes(mem, dimension, out)
    plot_model_selection(all_df, info["selected_family"], json.dumps(info["selected_hyperparams"]), info["selected_k"], dimension, out)
    plot_family_winners(winners_df, dimension, out)
    if rep is not None:
        plot_representative_distance(rep, dimension, out)
    return sorted(str(p.relative_to(fig_root)) for p in out.glob("*.png"))


def plot_cross_dimension_summary(all_info: dict, fig_root: Path) -> None:
    dims = list(all_info.keys())
    names = [f"{DIMENSION_DISPLAY_NAMES.get(d, d)}\n{FAMILY_DISPLAY_NAMES.get(all_info[d]['selected_family'], '')}, {all_info[d]['selected_k']} states"
             for d in dims]
    metrics = [("silhouette", "Silhouette"), ("init_stability_ari", "Init. stability"),
               ("bootstrap_stability_ari", "Bootstrap stability"), ("membership_confidence", "Confidence"),
               ("confidence_dispersion", "Confidence dispersion")]
    xs = np.arange(len(dims)); w = 0.16
    fig, ax = plt.subplots(figsize=(max(5.5, 2.2 * len(dims) + 1), 3.4))
    for i, (key, lab) in enumerate(metrics):
        ax.bar(xs + (i - 2) * w, [all_info[d][key] or 0 for d in dims], width=w, label=lab)
    ax.set_xticks(xs); ax.set_xticklabels(names); ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Retained clustering configuration per contextual dimension")
    ax.legend(frameon=False, ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    save_figure(fig, fig_root / "cross_dimension_summary")


def write_paper_tables(all_info: dict, root: Path) -> pd.DataFrame:
    """Compact recap table (CSV + Markdown + LaTeX booktabs) for the paper."""
    rows = []
    for d, i in all_info.items():
        rows.append({
            "Dimension": DIMENSION_DISPLAY_NAMES.get(d, d),
            "Features": ", ".join(i["features"]),
            "Algorithm": FAMILY_DISPLAY_NAMES.get(i["selected_family"], i["selected_family"]),
            "K": i["selected_k"],
            "Selection": i["stage1_criterion"].upper() if i["stage1_criterion"] == "icl" else "Silhouette",
            "Silhouette": i["silhouette"], "DBI": i["davies_bouldin"], "CHI": i["calinski_harabasz"],
            "ICL": i["icl"],
            "Stab. init": i["init_stability_ari"], "Stab. boot": i["bootstrap_stability_ari"],
            "Confidence": i["membership_confidence"], "Conf. disp.": i["confidence_dispersion"],
            "Obs/user/cluster": i["mean_obs_per_user_per_cluster"],
            "Min cluster (%)": 100 * i["smallest_cluster_fraction"],
            "Phase-2 distance": "Wasserstein" if i["representative_distance_matrix_available"] else "Hellinger / JS",
        })
    t = pd.DataFrame(rows)
    t.to_csv(root / "paper_table_clustering.csv", index=False)

    fmt = t.copy()
    for c in ["Silhouette", "DBI", "Stab. init", "Stab. boot", "Confidence", "Conf. disp.", "Obs/user/cluster"]:
        fmt[c] = fmt[c].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
    fmt["CHI"] = fmt["CHI"].map(lambda v: "--" if pd.isna(v) else f"{v:.0f}")
    fmt["ICL"] = fmt["ICL"].map(lambda v: "--" if pd.isna(v) else f"{v:.0f}")
    fmt["Min cluster (%)"] = fmt["Min cluster (%)"].map(lambda v: f"{v:.1f}")

    md = ["| " + " | ".join(fmt.columns) + " |", "|" + "|".join(["---"] * len(fmt.columns)) + "|"]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in fmt.to_numpy()]
    (root / "paper_table_clustering.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    cols = list(fmt.columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Retained clustering configuration and quality diagnostics per contextual dimension. "
             r"Stability is the mean adjusted Rand index over re-initialization (init) and bootstrap resampling (boot); "
             r"confidence is $1-$ normalized membership entropy, averaged over observations.}",
             r"\label{tab:clustering-summary}",
             r"\begin{tabular}{l" + "l" * 2 + "r" + "l" + "r" * (len(cols) - 6) + "l}", r"\toprule",
             " & ".join(esc(c) for c in cols) + r" \\", r"\midrule"]
    for r in fmt.to_numpy():
        lines.append(" & ".join(esc(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (root / "paper_table_clustering.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return t


# =============================================================================
# SAVING
# =============================================================================

def save_dimension_outputs(root: Path, dimension, df, all_df, winners_df, model, info) -> dict:
    d = root / dimension
    d.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(d / "algorithm_comparison.csv", index=False)
    winners_df.to_csv(d / "family_winners.csv", index=False)

    features = list(CONTEXT_DIMENSIONS[dimension]["features"])
    fam, k = info["selected_family"], info["selected_k"]

    out = df.copy()
    out["cluster"] = model["labels"] + 1
    for c in range(k):
        out[f"membership_cluster_{c + 1}"] = model["memberships"][:, c]
    out.to_csv(d / "soft_memberships.csv", sep=";", index=False)

    if fam in ("fuzzy_k_modes", "modality_identity"):
        r = pd.DataFrame(model["modes"], columns=features); r.insert(0, "cluster", np.arange(1, k + 1))
        r.to_csv(d / "cluster_modes.csv", index=False)
    elif fam == "fuzzy_k_medoids_gower":
        r = df.iloc[model["medoid_indices"]][["observationID", "userID", "itemID"] + features].copy()
        r.insert(0, "cluster", np.arange(1, k + 1)); r.to_csv(d / "cluster_medoids.csv", sep=";", index=False)
    elif fam in ("latent_class_analysis", "spectral_lca_hybrid"):
        params = {"k": k, "features": features, "priors": model["priors"].tolist(),
                  "category_values": model["category_values"],
                  "conditional_probabilities": [t.tolist() for t in model["conditional_probabilities"]],
                  "log_likelihood": model["log_likelihood"], "aic": model["aic"],
                  "bic": model["bic"], "icl": model["icl"], "initialization": fam}
        (d / "lca_model.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    elif fam == "gmm_mca":
        params = {"k": k, "features": features, "n_components_mca": model["n_components_mca"],
                  "reg_covar": model["reg_covar"], "means": model["means"].tolist(),
                  "bic": model["bic"], "icl": model["icl"]}
        (d / "gmm_mca_model.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    rep = build_representative_distance_matrix(fam, model, df, features)
    info = dict(info)
    if rep is not None:
        names = [f"cluster_{i + 1}" for i in range(rep.shape[0])]
        pd.DataFrame(rep, index=names, columns=names).to_csv(d / "cluster_representative_distance_matrix.csv")
        info["representative_distance_matrix_available"] = True
        info["recommended_phase2_distance_family"] = "wasserstein_or_distributional"
    else:
        info["representative_distance_matrix_available"] = False
        info["recommended_phase2_distance_family"] = "hellinger_or_jensen_shannon"

    try:
        info["figures"] = generate_dimension_figures(
            dimension, df, features, model, all_df, winners_df, info, rep, root / "figures")
    except Exception as exc:
        print(f"  WARNING: figure generation failed for {dimension}: {exc}")
        info["figures"] = []

    (d / "selected_configuration.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def build_summary_table(all_info: dict) -> pd.DataFrame:
    rows = []
    for dim, i in all_info.items():
        rows.append({
            "Dimension": dim, "Features": ", ".join(i["features"]),
            "Selected family": i["selected_family"], "Hyperparameters": json.dumps(i["selected_hyperparams"]),
            "Selected K": i["selected_k"], "Stage-1 criterion": i["stage1_criterion"],
            "Silhouette": i["silhouette"], "ICL": i["icl"],
            "Init stability ARI": i["init_stability_ari"], "Bootstrap stability ARI": i["bootstrap_stability_ari"],
            "Membership confidence": i["membership_confidence"], "Confidence dispersion": i["confidence_dispersion"],
            "Mean obs/user/cluster": i["mean_obs_per_user_per_cluster"],
            "Smallest cluster fraction": i["smallest_cluster_fraction"],
            "Stability gate used": i["bootstrap_stability_gate_used"],
            "Recommended phase-2 distance": i["recommended_phase2_distance_family"],
            "Diagnostic notes": i["quality_notes"],
        })
    return pd.DataFrame(rows)


def save_and_print_summary(table: pd.DataFrame, root: Path) -> None:
    table.to_csv(root / "clustering_summary_table.csv", index=False)
    disp = table.copy()
    for c in ["Silhouette", "ICL", "Init stability ARI", "Bootstrap stability ARI", "Membership confidence",
              "Confidence dispersion", "Mean obs/user/cluster", "Smallest cluster fraction"]:
        disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    text = disp.to_string(index=False)
    (root / "clustering_summary_table.txt").write_text("LDOS-CoMoDa - CLUSTERING SUMMARY (v3)\n" + "=" * 160 + "\n" + text + "\n", encoding="utf-8")
    print("\n" + "=" * 160 + "\nCLUSTERING SUMMARY TABLE\n" + "=" * 160)
    print(text)


def run_all_clusterings(datasets: dict, dim_reports: dict, root: Path) -> dict:
    all_info = {}
    for dim, df in datasets.items():
        all_df, winners_df, model, info = run_dimension_search(dim, df, dim_reports[dim])
        all_info[dim] = save_dimension_outputs(root, dim, df, all_df, winners_df, model, info)
    (root / "clustering_summary.json").write_text(json.dumps(all_info, indent=2, ensure_ascii=False), encoding="utf-8")
    save_and_print_summary(build_summary_table(all_info), root)
    try:
        plot_cross_dimension_summary(all_info, root / "figures")
        paper_table = write_paper_tables(all_info, root)
        print("\nPAPER TABLE (also saved as paper_table_clustering.csv / .md / .tex):")
        print(paper_table.to_string(index=False))
    except Exception as exc:
        print(f"WARNING: cross-dimension figure/table generation failed: {exc}")
    return all_info


# =============================================================================
# SCHEMA
# =============================================================================

def build_schema() -> dict:
    return {
        "id_columns": ID_COLUMNS, "target_column": TARGET_COLUMN,
        "user_static_columns": USER_STATIC_COLUMNS, "all_context_columns": ALL_CONTEXT_COLUMNS,
        "unused_context_columns": UNUSED_CONTEXT_COLUMNS, "item_metadata_columns": ITEM_METADATA_COLUMNS,
        "numeric_feature_columns": NUMERIC_FEATURE_COLUMNS, "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "context_dimensions": {d: {"features": c["features"], "description": c["description"]}
                               for d, c in CONTEXT_DIMENSIONS.items()},
        "feature_types": FEATURE_TYPES, "candidate_k": K_CANDIDATES,
        "algorithm_families": ALGORITHM_FAMILIES,
        "family_eligibility_rules": {
            "fuzzy_k_modes": "all features nominal",
            "fuzzy_k_medoids_gower": "any categorical mix (mixed Gower: nominal mismatch + ordinal rank difference)",
            "latent_class_analysis": "any categorical mix",
            "spectral_lca_hybrid": "any categorical mix; spectral partition on Gower affinity initializes LCA",
            "gmm_mca": f"unique patterns >= {MIN_UNIQUE_PATTERNS_FOR_EMBEDDING}; covariance floor = "
                       f"{GMM_REG_COVAR_FRACTION} x mean embedding variance",
        },
        "hyperparameter_grids": {"fuzziness": FUZZINESS_GRID, "mca_components": MCA_COMPONENTS_GRID,
                                 "gmm_covariance_type": GMM_COVARIANCE_GRID},
        "clustering_parameters": {
            "n_init": N_INIT, "max_iter": MAX_ITER, "tol": TOL, "init_stability_runs": INIT_STABILITY_RUNS,
            "bootstrap_runs": BOOTSTRAP_RUNS, "bootstrap_fraction": BOOTSTRAP_FRACTION,
            "random_state": RANDOM_STATE, "lca_smoothing": LCA_SMOOTHING,
        },
        "admissibility_gates": {
            "min_clusters": MIN_CLUSTERS, "max_clusters_fraction_of_users": MAX_CLUSTERS_FRACTION_OF_USERS,
            "min_cluster_fraction": MIN_CLUSTER_FRACTION,
            "min_mean_obs_per_user_per_cluster": MIN_MEAN_OBS_PER_USER_PER_CLUSTER,
            "min_membership_confidence": MIN_MEMBERSHIP_CONFIDENCE,
            "max_membership_confidence": MAX_MEMBERSHIP_CONFIDENCE,
            "min_confidence_dispersion": MIN_CONFIDENCE_DISPERSION,
            "min_bootstrap_stability": MIN_BOOTSTRAP_STABILITY,
            "stability_relaxation_steps": STABILITY_RELAXATION_STEPS,
        },
        "selection_procedure": (
            "Per dimension: (0) eligible families derived from variable nature and cardinality; "
            "(1) admissibility gates including per-user coverage, informative (neither diffuse nor crisp) "
            "confidence, and bootstrap stability; (2) stage 1 = best K/hyperparameters within each "
            "family by its native criterion (ICL for probabilistic models, silhouette on mixed Gower for "
            "distance-based models); (3) stage 2 = family winners compared on common metrics only, "
            "stability first, then silhouette, then confidence dispersion. The retained family "
            "therefore legitimately differs across dimensions."
        ),
        "methodological_rules": [
            "Contextual integer codes are categorical labels; 'mood' is treated as ordinal.",
            "Only complete observations per dimension are retained; no global imputation.",
            "Each retained model returns a normalized observation-level soft-membership vector.",
            "Observation counts N_u^d and membership vectors are preserved for the reliability mechanism.",
            "Cluster representatives and their normalized distance matrix are exported for the Wasserstein cost matrix.",
            "The full-dataset clustering is exploratory; the search must be repeated within each evaluation fold.",
        ],
    }


# =============================================================================
# PHASE 2 CONFIGURATION
# =============================================================================

# Smoothing constant eps_m of Eq. (user-cluster-probability). Small enough
# not to distort the profiles, large enough to guarantee strict positivity
# (required by the KL terms inside the Jensen-Shannon distance).
EPSILON_M = 0.01

# Distributional distance selected per dimension (Eq. dimension-dissimilarity).
# "wasserstein" requires the inter-cluster cost matrix exported by Phase 1;
# if the matrix is unavailable the code falls back to FALLBACK_DISTANCE.
# All three distances are computed anyway and saved for the empirical
# assessment announced in the approach.
DISTANCE_PER_DIMENSION = {
    "temporal": "wasserstein",
    "spatial_environmental": "wasserstein",
    "affective_physical": "hellinger",
}
FALLBACK_DISTANCE = "hellinger"
ALL_DISTANCES = ("hellinger", "jensen_shannon", "wasserstein")
# Set to False on large user sets: the pairwise LP is the only O(|U|^2 * LP) step.
COMPUTE_WASSERSTEIN = True

# Saturation parameter tau_d of the observation-support score
# s_u^d = N_u^d / (N_u^d + tau_d). "median" sets tau_d to the median of
# N_u^d in the dimension (the median user gets s = 0.5); a positive
# number fixes tau_d explicitly for every dimension.
TAU_RULE = "median"

DIMENSIONS = list(CONTEXT_DIMENSIONS.keys())



# =============================================================================
# PHASE 1 CLOSURE - PROBABILISTIC USER PROFILES
# =============================================================================

def membership_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith("membership_cluster_")],
                  key=lambda c: int(c.rsplit("_", 1)[1]))
    if not cols:
        raise ValueError("No membership_cluster_* column found.")
    return cols




def build_user_profiles(memberships_df: pd.DataFrame, epsilon_m: float = EPSILON_M) -> dict:
    """
    From observation-level memberships (one row per observation with a
    userID and K membership columns) to:
      profiles      : DataFrame (users x K) of p_{u,k}^d, rows sum to 1, > 0
      counts        : Series N_u^d
      obs_entropy   : DataFrame (observationID, userID, entropy e_{u,n}^d)
      confidence    : Series c_u^d = 1 - mean_n e_{u,n}^d
    """
    cols = membership_columns(memberships_df)
    k = len(cols)
    m = memberships_df[cols].to_numpy(dtype=float)
    if not np.allclose(m.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Observation-level memberships do not sum to 1.")

    users = memberships_df["userID"].to_numpy()
    sums = pd.DataFrame(m, columns=[f"cluster_{i + 1}" for i in range(k)])
    sums["userID"] = users
    agg = sums.groupby("userID").sum()
    counts = memberships_df.groupby("userID").size().reindex(agg.index)

    profiles = (agg.to_numpy() + epsilon_m) / (counts.to_numpy()[:, None] + k * epsilon_m)
    profiles = pd.DataFrame(profiles, index=agg.index, columns=agg.columns)

    if not np.allclose(profiles.sum(axis=1), 1.0, atol=1e-9) or (profiles.to_numpy() <= 0).any():
        raise RuntimeError("Profiles are not valid probability distributions.")

    entropy = per_observation_entropy(m)
    obs_entropy = pd.DataFrame({
        "observationID": memberships_df["observationID"].to_numpy() if "observationID" in memberships_df else np.arange(len(m)),
        "userID": users,
        "entropy": entropy,
    })
    confidence = 1.0 - obs_entropy.groupby("userID")["entropy"].mean().reindex(agg.index)

    return {"k": k, "profiles": profiles, "counts": counts.astype(int),
            "obs_entropy": obs_entropy, "confidence": confidence}


def profile_diagnostics(prof: dict) -> dict:
    p = prof["profiles"].to_numpy()
    k = prof["k"]
    prof_entropy = -np.sum(p * np.log(p), axis=1) / np.log(k) if k > 1 else np.zeros(len(p))
    dominant = p.max(axis=1)
    return {
        "k": int(k),
        "n_users_profiled": int(len(p)),
        "observations_per_user": {
            "min": int(prof["counts"].min()), "median": float(prof["counts"].median()),
            "mean": float(prof["counts"].mean()), "max": int(prof["counts"].max()),
        },
        "profile_entropy_normalized": {"mean": float(prof_entropy.mean()), "std": float(prof_entropy.std())},
        "dominant_cluster_probability": {"mean": float(dominant.mean()),
                                         "share_users_above_0.8": float((dominant > 0.8).mean())},
        "membership_confidence_c_u": {"mean": float(prof["confidence"].mean()),
                                      "std": float(prof["confidence"].std())},
        "mean_profile_over_users": {c: float(v) for c, v in prof["profiles"].mean().items()},
    }


# =============================================================================
# PHASE 2 - DISTRIBUTIONAL DISTANCES
# =============================================================================

def hellinger_matrix(p: np.ndarray) -> np.ndarray:
    """Pairwise Hellinger distances (Eq. hellinger-distance), in [0, 1]."""
    r = np.sqrt(p)
    sq = ((r[:, None, :] - r[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(np.clip(sq, 0, None) / 2.0)


def _js_rows(args):
    p, logp, rows = args
    out = np.zeros((len(rows), p.shape[0]))
    for r, i in enumerate(rows):
        m = 0.5 * (p[i][None, :] + p)
        logm = np.log2(m)
        kl_i = np.sum(p[i][None, :] * (logp[i][None, :] - logm), axis=1)
        kl_j = np.sum(p * (logp - logm), axis=1)
        out[r] = np.sqrt(np.clip(0.5 * kl_i + 0.5 * kl_j, 0, None))
    return out


def _row_blocks(n: int, n_blocks: int) -> list[np.ndarray]:
    return [b for b in np.array_split(np.arange(n), max(1, min(n_blocks, n))) if len(b)]


def jensen_shannon_matrix(p: np.ndarray) -> np.ndarray:
    """Pairwise Jensen-Shannon distances (square root of the divergence,
    log base 2), in [0, 1] (Eq. jensen-shannon-distance). Row blocks are
    computed in parallel."""
    n = p.shape[0]
    logp = np.log2(p)
    blocks = _row_blocks(n, max(N_JOBS * 4, 1))
    parts = parallel_map(_js_rows, [(p, logp, b) for b in blocks])
    out = np.vstack(parts) if parts else np.zeros((n, n))
    np.fill_diagonal(out, 0.0)
    return out


def wasserstein_distance(p: np.ndarray, q: np.ndarray, cost: np.ndarray) -> float:
    """Discrete Wasserstein distance (Eq. wasserstein-distance): minimum-cost
    transport plan between p and q with the normalized cost matrix."""
    k = len(p)
    if k == 2:
        return float(abs(p[0] - q[0]) * cost[0, 1])
    c = cost.ravel()
    a_eq = np.zeros((2 * k, k * k))
    for i in range(k):
        a_eq[i, i * k:(i + 1) * k] = 1.0          # row marginals = p
        a_eq[k + i, i::k] = 1.0                    # column marginals = q
    b_eq = np.concatenate([p, q])
    res = linprog(c, A_eq=a_eq[:-1], b_eq=b_eq[:-1], bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(f"Wasserstein LP failed: {res.message}")
    return float(res.fun)


def _wasserstein_rows(args):
    p, cost, rows = args
    n = p.shape[0]
    out = np.zeros((len(rows), n))
    for r, i in enumerate(rows):
        for j in range(i + 1, n):
            out[r, j] = wasserstein_distance(p[i], p[j], cost)
    return out


def wasserstein_matrix(p: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Pairwise discrete Wasserstein distances; the LP pairs are split
    into row blocks computed in parallel (upper triangle, symmetrised)."""
    n = p.shape[0]
    # interleave rows so that blocks have similar numbers of (i, j>i) pairs
    order = np.arange(n)
    blocks = [order[k::max(N_JOBS * 4, 1)] for k in range(max(N_JOBS * 4, 1))]
    blocks = [b for b in blocks if len(b)]
    parts = parallel_map(_wasserstein_rows, [(p, cost, b) for b in blocks])
    out = np.zeros((n, n))
    for b, part in zip(blocks, parts):
        out[b] = part
    out = np.triu(out, 1); out = out + out.T
    return out


def compute_all_distance_matrices(profiles: pd.DataFrame, cost: np.ndarray | None) -> dict[str, pd.DataFrame]:
    p = profiles.to_numpy(dtype=float)
    idx = profiles.index
    out = {
        "hellinger": pd.DataFrame(hellinger_matrix(p), index=idx, columns=idx),
        "jensen_shannon": pd.DataFrame(jensen_shannon_matrix(p), index=idx, columns=idx),
    }
    if cost is not None and COMPUTE_WASSERSTEIN:
        if cost.shape[0] != p.shape[1]:
            raise ValueError("Cost matrix size does not match the number of clusters.")
        if cost.max() > 1.0 + 1e-9 or cost.min() < 0:
            raise ValueError("Cost matrix must be normalized in [0, 1].")
        out["wasserstein"] = pd.DataFrame(wasserstein_matrix(p, cost), index=idx, columns=idx)
    return out


# =============================================================================
# PHASE 2 - RELIABILITY
# =============================================================================

def resolve_tau(counts: pd.Series) -> float:
    if TAU_RULE == "median":
        return float(max(counts.median(), 1.0))
    return float(TAU_RULE)


def user_reliability(counts: pd.Series, confidence: pd.Series, tau: float) -> pd.DataFrame:
    """s_u^d, c_u^d and rho_u^d = s_u^d * c_u^d (Eqs. observation-support,
    membership-confidence, user-dimension-reliability)."""
    s = counts.astype(float) / (counts.astype(float) + tau)
    c = confidence.reindex(counts.index).clip(0.0, 1.0)
    return pd.DataFrame({"N_u": counts, "support_s_u": s, "confidence_c_u": c, "reliability_rho_u": s * c})


def pair_reliability_matrix(rho: pd.Series) -> pd.DataFrame:
    """r_uv^d = sqrt(rho_u^d rho_v^d) (Eq. pair-dimension-reliability)."""
    r = rho.to_numpy(dtype=float)
    mat = np.sqrt(np.outer(r, r))
    return pd.DataFrame(mat, index=rho.index, columns=rho.index)


# =============================================================================
# PHASE 2 - WEIGHTING, AGGREGATION AND SIMILARITY
# =============================================================================

def aggregate_dissimilarities(all_users: pd.Index, deltas: dict[str, pd.DataFrame],
                              pair_rel: dict[str, pd.DataFrame]) -> dict:
    """
    deltas[d]   : delta_uv^d matrix (users available in d)
    pair_rel[d] : r_uv^d matrix (same users)
    Returns Delta^unif, Delta^rel, the number of shared dimensions |D_uv|,
    the reliability-aware weights per dimension, and S = 1 - Delta.
    Pairs with D_uv empty get NaN.
    """
    n = len(all_users)
    pos = {u: i for i, u in enumerate(all_users)}
    delta_stack = np.full((len(deltas), n, n), np.nan)
    rel_stack = np.zeros((len(deltas), n, n))
    dims = list(deltas.keys())

    for a, d in enumerate(dims):
        users_d = deltas[d].index
        ix = np.array([pos[u] for u in users_d])
        delta_stack[a][np.ix_(ix, ix)] = deltas[d].to_numpy()
        rel_stack[a][np.ix_(ix, ix)] = pair_rel[d].to_numpy()

    available = ~np.isnan(delta_stack)                 # dimension d in D_uv
    n_shared = available.sum(axis=0)

    # Uniform weights: 1/|D_uv| (Eq. uniform-contextual-weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        delta_unif = np.nansum(delta_stack, axis=0) / n_shared

    # Reliability-aware weights (Eq. reliability-aware-weight)
    rel_masked = np.where(available, rel_stack, 0.0)
    total_rel = rel_masked.sum(axis=0)
    w_rel = np.zeros_like(rel_masked)
    positive = total_rel > 0
    w_rel[:, positive] = rel_masked[:, positive] / total_rel[positive]
    fallback = (~positive) & (n_shared > 0)              # R_uv = 0 -> uniform
    w_rel[:, fallback] = available[:, fallback] / n_shared[fallback]
    delta_rel = np.nansum(np.where(available, delta_stack, 0.0) * w_rel, axis=0)

    empty = n_shared == 0
    delta_unif[empty] = np.nan
    delta_rel[empty] = np.nan
    np.fill_diagonal(delta_unif, np.nan); np.fill_diagonal(delta_rel, np.nan)

    def frame(mat):
        return pd.DataFrame(mat, index=all_users, columns=all_users)

    return {
        "dimensions": dims,
        "n_shared_dimensions": frame(n_shared),
        "delta_unif": frame(delta_unif), "delta_rel": frame(delta_rel),
        "similarity_unif": frame(1.0 - delta_unif), "similarity_rel": frame(1.0 - delta_rel),
        "weights_rel": {d: frame(w_rel[a]) for a, d in enumerate(dims)},
        "weights_unif": {d: frame(np.where(available[a], 1.0 / np.maximum(n_shared, 1), 0.0)) for a, d in enumerate(dims)},
    }


# =============================================================================
# FIGURES
# =============================================================================




def safe_hist(ax, values, bins=30, **kwargs):
    """Histogram robust to constant or empty data (e.g. exactly uniform
    reliability weights when every dimension has the same support)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    if np.ptp(values) < 1e-9:
        bins = 1
    ax.hist(values, bins=bins, **kwargs)

def upper_values(mat: pd.DataFrame) -> np.ndarray:
    a = mat.to_numpy(dtype=float)
    return a[np.triu_indices_from(a, k=1)]


def plot_profile_summary(dim: str, prof: dict, out: Path) -> None:
    p = prof["profiles"]
    k = prof["k"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    order = np.argsort(-p.to_numpy().max(axis=1))
    axes[0].imshow(p.to_numpy()[order], aspect="auto", cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks(range(k)); axes[0].set_xticklabels([f"C{i + 1}" for i in range(k)])
    axes[0].set_ylabel("Users"); axes[0].set_title("User context profiles")
    safe_hist(axes[1], prof["counts"], bins=25, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel("Observations per user"); axes[1].set_title("Observation support")
    pe = -np.sum(p.to_numpy() * np.log(p.to_numpy()), axis=1) / np.log(k) if k > 1 else np.zeros(len(p))
    safe_hist(axes[2], pe, bins=25, color="#DD8452", edgecolor="white")
    axes[2].set_xlabel("Normalized profile entropy"); axes[2].set_title("Contextual diversity of users")
    fig.suptitle(f"{DIMENSION_DISPLAY_NAMES.get(dim, dim)} - probabilistic user profiles ({k} states)")
    save_figure(fig, out / f"{dim}_profiles")


def plot_distance_distributions(dim: str, dmats: dict[str, pd.DataFrame], selected: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, mat in dmats.items():
        vals = upper_values(mat)
        safe_hist(ax, vals, bins=40, histtype="step", lw=1.6 if name == selected else 1.0,
                label=f"{name.replace('_', '-')}{' (selected)' if name == selected else ''}")
    ax.set_xlabel("Pairwise dissimilarity"); ax.set_ylabel("User pairs"); ax.set_xlim(0, 1)
    ax.set_title(f"{DIMENSION_DISPLAY_NAMES.get(dim, dim)} - dimension-specific dissimilarities")
    ax.legend(frameon=False)
    save_figure(fig, out / f"{dim}_dissimilarity_distributions")


def plot_reliability(dim: str, rel: pd.DataFrame, tau: float, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    axes[0].scatter(rel["N_u"], rel["support_s_u"], s=10, color="#4C72B0")
    axes[0].axvline(tau, ls=":", color="k", lw=1); axes[0].set_xlabel("Observations per user"); axes[0].set_ylabel("Support score")
    axes[0].set_title(f"Observation support (threshold = {tau:.1f})")
    axes[1].scatter(rel["confidence_c_u"], rel["reliability_rho_u"], s=10, color="#DD8452")
    axes[1].set_xlabel("Membership confidence"); axes[1].set_ylabel("Reliability score"); axes[1].set_title("User reliability")
    fig.suptitle(f"{DIMENSION_DISPLAY_NAMES.get(dim, dim)} - reliability components")
    save_figure(fig, out / f"{dim}_reliability")


def plot_weights_and_similarity(agg: dict, out: Path) -> None:
    dims = agg["dimensions"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    for d in dims:
        w = upper_values(agg["weights_rel"][d]); n = upper_values(agg["n_shared_dimensions"])
        safe_hist(axes[0], w[n > 0], bins=30, histtype="step", lw=1.3, label=DIMENSION_DISPLAY_NAMES.get(d, d))
    axes[0].set_xlabel("Reliability-aware weight"); axes[0].set_ylabel("User pairs"); axes[0].set_title("Reliability-aware weights")
    axes[0].legend(frameon=False)
    su, sr = upper_values(agg["similarity_unif"]), upper_values(agg["similarity_rel"])
    safe_hist(axes[1], su[~np.isnan(su)], bins=40, histtype="step", lw=1.3, label="uniform")
    safe_hist(axes[1], sr[~np.isnan(sr)], bins=40, histtype="step", lw=1.3, label="reliability-aware")
    axes[1].set_xlabel("Contextual similarity"); axes[1].set_title("Contextual similarity"); axes[1].legend(frameon=False)
    save_figure(fig, out / "weights_and_similarity")


# =============================================================================
# PIPELINE RUNNER (reads Phase-1 outputs)
# =============================================================================

def load_phase1_dimension_inputs(root: Path, dim: str) -> dict | None:
    d = root / dim
    if not (d / "soft_memberships.csv").exists():
        return None
    mem = pd.read_csv(d / "soft_memberships.csv", sep=";")
    cfg = json.loads((d / "selected_configuration.json").read_text(encoding="utf-8")) if (d / "selected_configuration.json").exists() else {}
    cost = None
    if (d / "cluster_representative_distance_matrix.csv").exists():
        cost = pd.read_csv(d / "cluster_representative_distance_matrix.csv", index_col=0).to_numpy(dtype=float)
    return {"memberships": mem, "config": cfg, "cost": cost}


def run_phase2(root: Path, out_root: Path) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root = out_root / "figures"; fig_root.mkdir(exist_ok=True)

    report = {"epsilon_m": EPSILON_M, "tau_rule": TAU_RULE, "dimensions": {}}
    deltas, pair_rel, all_users = {}, {}, set()

    for dim in list(CONTEXT_DIMENSIONS.keys()):
        inputs = load_phase1_dimension_inputs(root, dim)
        if inputs is None:
            print(f"[skip] {dim}: no Phase-1 output found"); continue
        d_out = out_root / dim; d_out.mkdir(exist_ok=True)

        # ---- Phase 1 closure
        prof = build_user_profiles(inputs["memberships"], EPSILON_M)
        prof["profiles"].to_csv(d_out / "user_profiles.csv")
        prof["counts"].rename("N_u").to_csv(d_out / "user_observation_counts.csv")
        prof["obs_entropy"].to_csv(d_out / "observation_entropies.csv", index=False)
        diag = profile_diagnostics(prof)
        if prof["k"] < 2:
            print(f"[skip] {dim}: K_d < 2, dimension is not informative for D_uv"); continue

        # ---- Phase 2: distances
        with timed("phase2.distances", dimension=dim, n_users=int(len(prof["profiles"])), k=int(prof["k"])):
            dmats = compute_all_distance_matrices(prof["profiles"], inputs["cost"])
        wanted = DISTANCE_PER_DIMENSION.get(dim, FALLBACK_DISTANCE)
        selected = wanted if wanted in dmats else FALLBACK_DISTANCE
        for name, mat in dmats.items():
            mat.to_csv(d_out / f"delta_{name}.csv")
        deltas[dim] = dmats[selected]

        # ---- Phase 2: reliability
        tau = resolve_tau(prof["counts"])
        rel = user_reliability(prof["counts"], prof["confidence"], tau)
        rel.to_csv(d_out / "user_reliability.csv")
        pair_rel[dim] = pair_reliability_matrix(rel["reliability_rho_u"])
        all_users |= set(prof["profiles"].index)

        # ---- diagnostics & figures
        dist_stats = {name: {"mean": float(upper_values(m).mean()), "std": float(upper_values(m).std()),
                             "min": float(upper_values(m).min()), "max": float(upper_values(m).max())}
                      for name, m in dmats.items()}
        corr = {}
        names = list(dmats.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                corr[f"{names[i]}_vs_{names[j]}"] = float(np.corrcoef(upper_values(dmats[names[i]]), upper_values(dmats[names[j]]))[0, 1])
        report["dimensions"][dim] = {
            **diag, "selected_distance": selected, "requested_distance": wanted,
            "cost_matrix_available": inputs["cost"] is not None, "tau_d": tau,
            "distance_statistics": dist_stats, "distance_rank_agreement_pearson": corr,
            "reliability": {"mean_s_u": float(rel["support_s_u"].mean()), "mean_c_u": float(rel["confidence_c_u"].mean()),
                            "mean_rho_u": float(rel["reliability_rho_u"].mean()), "std_rho_u": float(rel["reliability_rho_u"].std())},
            "phase1_algorithm": inputs["config"].get("selected_family"),
        }
        plot_profile_summary(dim, prof, fig_root)
        plot_distance_distributions(dim, dmats, selected, fig_root)
        plot_reliability(dim, rel, tau, fig_root)
        print(f"{dim:<24} users={diag['n_users_profiled']:<4} K={prof['k']} distance={selected:<15} "
              f"tau={tau:.1f} mean delta={dist_stats[selected]['mean']:.3f} mean rho={rel['reliability_rho_u'].mean():.3f}")

    if not deltas:
        raise RuntimeError("No dimension could be processed.")

    # ---- Phase 2: aggregation and similarity
    users = pd.Index(sorted(all_users))
    agg = aggregate_dissimilarities(users, deltas, pair_rel)
    agg["similarity_unif"].to_csv(out_root / "similarity_unif.csv")
    agg["similarity_rel"].to_csv(out_root / "similarity_rel.csv")
    agg["delta_unif"].to_csv(out_root / "delta_unif.csv")
    agg["delta_rel"].to_csv(out_root / "delta_rel.csv")
    agg["n_shared_dimensions"].to_csv(out_root / "n_shared_dimensions.csv")
    for d in agg["dimensions"]:
        agg["weights_rel"][d].to_csv(out_root / f"weights_rel_{d}.csv")

    ns = upper_values(agg["n_shared_dimensions"])
    su, sr = upper_values(agg["similarity_unif"]), upper_values(agg["similarity_rel"])
    report["aggregation"] = {
        "n_users": int(len(users)),
        "n_pairs": int(len(ns)),
        "pairs_with_no_shared_dimension": int((ns == 0).sum()),
        "shared_dimension_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(ns, return_counts=True))},
        "similarity_unif": {"mean": float(np.nanmean(su)), "std": float(np.nanstd(su))},
        "similarity_rel": {"mean": float(np.nanmean(sr)), "std": float(np.nanstd(sr))},
        "unif_vs_rel_pearson": float(np.corrcoef(su[~np.isnan(su)], sr[~np.isnan(sr)])[0, 1]),
        "mean_weight_rel_per_dimension": {d: float(np.nanmean(upper_values(agg["weights_rel"][d])[ns > 0])) for d in agg["dimensions"]},
        "note": ("Prediction-aware weighting is learned in Phase 3 through the complete "
                 "neighbourhood-construction and rating-prediction pipeline."),
    }
    plot_weights_and_similarity(agg, fig_root)
    (out_root / "phase2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_timings(out_root / "timings_phase2.json")

    print("\nAggregation:")
    print(f"  users={len(users)} pairs={len(ns)} pairs without shared dimension={(ns == 0).sum()}")
    print(f"  S_unif mean={np.nanmean(su):.3f}  S_rel mean={np.nanmean(sr):.3f}  corr(unif, rel)={report['aggregation']['unif_vs_rel_pearson']:.3f}")
    print(f"  mean reliability-aware weight per dimension: {report['aggregation']['mean_weight_rel_per_dimension']}")
    return report


# =============================================================================
# PHASE 3 CONFIGURATION
# =============================================================================
#
# Phase 3 = K context-driven nearest neighbours for collaborative
# filtering (Algorithm 3), prediction-aware weighting (learned through the
# complete pipeline), fold-wise evaluation protocol and baselines.
#
# Evaluation protocol. Ratings are split into N_FOLDS folds (each user's
# observations are spread across folds), repeated N_REPEATS times with
# different seeds. Within a fold, everything that depends on data is
# recomputed on the training part only. All hyperparameters are selected
# on an INNER validation split of the training part; the test fold never
# participates in any selection.
#
# What is selected on the inner validation split
#   - number of clusters K_d per dimension (end-to-end, by prediction loss;
#     admissibility gates of Phase 1 kept as hard constraints)
#   - distributional distance per dimension
#   - similarity contrast exponent gamma  (S_eff = S^gamma)
#   - reliability saturation tau_d = factor x median(N_u^d)   (rel scheme)
#   - regularisation of the baseline predictor mu + b_u + b_i and the
#     neighbourhood shrinkage n/(n + beta)  (all neighbourhood methods)
#   - number of neighbours K per method
#   - prediction-aware coefficients alpha* (Eq. prediction-aware-optimization)
#
# FOLD_CLUSTERING_MODE: "refit" keeps the family / hyperparameters of the
# exploratory search and refits on the training part (K_d being re-selected
# on validation); "reselect" re-runs the full two-stage family search.

CLEAN_FILENAME = "LDOS-CoMoDa_clean.csv"   # overridden by dataset adapters
N_FOLDS = 5
N_REPEATS = 1
FOLD_SEED = 2024
INNER_VALIDATION_FRACTION = 0.2
FOLD_CLUSTERING_MODE = "refit"

K_NEIGHBORS_GRID = [5, 10, 20, 30, 50]
SELECT_K_CLUSTERS_ON_VALIDATION = True
K_CLUSTER_GRID_VALIDATION = [2, 3, 4, 5, 6, 7, 8]
SELECT_DISTANCE_ON_VALIDATION = True
GAMMA_GRID = [1.0, 2.0, 4.0, 8.0]
TAU_FACTOR_GRID = [1.0, 0.5, 0.25]
BIAS_REG_GRID = [2.0, 5.0, 10.0]
SHRINK_GRID = [0.0, 2.0, 5.0, 10.0]

ALPHA_GRID_STEP = 0.1
LAMBDA_ALPHA = 1e-3              # at the scale of the prediction loss
PREDICTION_LOSS = "rmse"         # "mae" or "rmse"

MIN_POSITIVE_SIMILARITY = 1e-12
WEIGHTING_SCHEMES = ["unif", "rel", "pred"]

# Hybrid neighbourhood similarity (extension of the approach): the
# contextual similarity S^q complements the rating-based similarity
# instead of replacing it,
#     S_hyb = lambda * S_rating^+ + (1 - lambda) * S_context^q,
# with S_rating^+ the positive part of the Pearson similarity and the
# weights renormalised over the available components (pairs without
# co-rated items use the contextual component alone, pairs without
# contextual profiles use the rating component alone). lambda is selected
# on the inner validation split for each weighting scheme. lambda = 1
# reduces to the Pearson baseline, lambda = 0 to the purely contextual
# neighbourhood. The weighting schemes act on the contextual component.
HYBRID_LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
HYBRID_SCHEMES = ["unif", "rel", "pred"]

# Evaluation regimes (each is a full cross-validation run):
#   "full"       : all training observations build the profiles (abundant,
#                  homogeneous contextual evidence for every user).
#   "limited"    : for each user and dimension, only a random number of
#                  training observations (between EVIDENCE_MIN_OBS and N_u^d)
#                  is used to build the profile, while clustering and all
#                  ratings are unchanged. This is the regime the reliability-
#                  aware weighting is designed for: heterogeneous observation
#                  support s_u^d and confidence c_u^d across users and
#                  dimensions ("sparse or ambiguous contextual information").
#   "cold_start" : a share of users is made cold in every fold: only
#                  COLD_START_KEEP of their ratings remain in training (the
#                  rest join the test fold), contextual observations being
#                  kept only for those ratings. Rating-based similarities
#                  collapse; contextual profiles remain defined.
EVALUATION_REGIMES = ["full", "limited", "cold_start"]
EVIDENCE_MIN_OBS = 1
COLD_START_USER_FRACTION = 0.3
COLD_START_KEEP = 3

# Ranking metrics (relevance = rating >= RELEVANCE_THRESHOLD), computed per
# user over that user's test items, for users with at least
# RANKING_MIN_TEST_ITEMS test items and one relevant item.
RELEVANCE_THRESHOLD = 4
RANKING_K = 3
RANKING_MIN_TEST_ITEMS = 3

# Weighting-effect subset: test observations whose target user has
# heterogeneous contextual evidence (top HETEROGENEITY_SHARE of users by the
# spread of rho_u^d across dimensions, or with a missing dimension).
HETEROGENEITY_SHARE = 0.5

# Baselines. "context_prefilter_knn" restricts neighbour ratings to those
# given under the same raw context values (classical contextual
# pre-filtering); "biased_mf" is a regularised matrix factorisation.
BASELINES = ["global_mean", "item_mean", "user_mean", "baseline_bias",
             "user_knn_pearson", "context_prefilter_knn", "biased_mf"]
PREFILTER_CONTEXT_VARS = ["daytype", "location"]
MF_FACTORS = 10
MF_EPOCHS = 40
MF_LR = 0.01
MF_REG_GRID = [0.02, 0.05, 0.1]


# =============================================================================
# PHASE 3 - RATING INDEX, BASELINE PREDICTOR, PREDICTION (ALGORITHM 3)
# =============================================================================

class RatingIndex:
    """Training ratings indexed for neighbourhood formation, with a
    regularised baseline predictor b_ui = mu + b_u + b_i. Repeated
    (user, item) pairs are averaged into one training rating."""

    def __init__(self, train: pd.DataFrame, bias_reg: float = 5.0, context_vars: list[str] | None = None):
        self.global_mean = float(train[TARGET_COLUMN].mean())
        self.user_mean = train.groupby("userID")[TARGET_COLUMN].mean()
        self.item_mean = train.groupby("itemID")[TARGET_COLUMN].mean()
        r = train.groupby(["userID", "itemID"])[TARGET_COLUMN].mean().reset_index()
        self.by_item = {i: (g["userID"].to_numpy(), g[TARGET_COLUMN].to_numpy(dtype=float)) for i, g in r.groupby("itemID")}
        self.pivot = r.pivot(index="userID", columns="itemID", values=TARGET_COLUMN)
        # regularised biases (two alternating passes)
        mu = self.global_mean
        bu = pd.Series(0.0, index=self.user_mean.index); bi = pd.Series(0.0, index=self.item_mean.index)
        for _ in range(2):
            resid = train[TARGET_COLUMN] - mu - bi.reindex(train["itemID"]).to_numpy()
            bu = resid.groupby(train["userID"].to_numpy()).sum() / (train.groupby("userID").size() + bias_reg)
            resid = train[TARGET_COLUMN] - mu - bu.reindex(train["userID"]).to_numpy()
            bi = resid.groupby(train["itemID"].to_numpy()).sum() / (train.groupby("itemID").size() + bias_reg)
        self.b_u, self.b_i = bu, bi
        self.context_vars = context_vars or []
        self.by_item_context = {}
        if self.context_vars:
            g = train.groupby(["itemID"] + self.context_vars + ["userID"])[TARGET_COLUMN].mean().reset_index()
            for key, grp in g.groupby(["itemID"] + self.context_vars):
                self.by_item_context[tuple(key)] = (grp["userID"].to_numpy(), grp[TARGET_COLUMN].to_numpy(dtype=float))

    def baseline(self, u, i, use_bias: bool) -> float:
        if use_bias:
            return self.global_mean + float(self.b_u.get(u, 0.0)) + float(self.b_i.get(i, 0.0))
        m = self.user_mean.get(u, np.nan)
        return float(m) if not np.isnan(m) else (float(self.item_mean.get(i, self.global_mean)))

    def baseline_vector(self, users: np.ndarray, i, use_bias: bool) -> np.ndarray:
        if use_bias:
            return self.global_mean + self.b_u.reindex(users).fillna(0.0).to_numpy() + float(self.b_i.get(i, 0.0))
        return self.user_mean.reindex(users).to_numpy()


def predict_with_similarity(test: pd.DataFrame, idx: RatingIndex, S: pd.DataFrame | None, k: int,
                            shrink_beta: float = 0.0, use_bias: bool = True, context_filter: bool = False) -> pd.DataFrame:
    """
    Algorithm 3 with a robust predictor: candidates = users who rated i in
    training with a defined, strictly positive similarity; the k most
    similar contribute a similarity-weighted deviation from their baseline,
    shrunk by n/(n + beta) for small neighbourhoods:
        r_hat = b_ui + n/(n+beta) * sum_v S_uv (r_vi - b_vi) / sum_v S_uv
    Fallback chain: baseline predictor (b_ui) or user mean -> item mean ->
    global mean when use_bias is False. context_filter restricts candidate
    ratings to those given under the same raw context values (baseline
    "contextual pre-filtering"), relaxing to all ratings when none exists.
    """
    preds = np.empty(len(test)); source = np.empty(len(test), dtype=object)
    s_index = {u: j for j, u in enumerate(S.index)} if S is not None else {}
    s_values = S.to_numpy(dtype=float) if S is not None else None
    ctx_cols = idx.context_vars if context_filter else []
    ctx_values = test[ctx_cols].to_numpy() if ctx_cols else None

    for t, (u, i) in enumerate(zip(test["userID"].to_numpy(), test["itemID"].to_numpy())):
        base = idx.baseline(u, i, use_bias)
        done = False
        if S is not None and u in s_index:
            cand = None
            if ctx_cols:
                key = (i,) + tuple(int(v) if not pd.isna(v) else -1 for v in ctx_values[t])
                cand = idx.by_item_context.get(key)
            if cand is None:
                cand = idx.by_item.get(i)
            if cand is not None:
                cand_users, cand_ratings = cand
                cols = np.array([s_index.get(v, -1) for v in cand_users])
                sims = np.full(len(cand_users), np.nan)
                ok = cols >= 0
                sims[ok] = s_values[s_index[u], cols[ok]]
                keep = np.isfinite(sims) & (sims > MIN_POSITIVE_SIMILARITY) & (cand_users != u)
                if keep.any():
                    sims_k, users_k, ratings_k = sims[keep], cand_users[keep], cand_ratings[keep]
                    if len(sims_k) > k:
                        top = np.argsort(-sims_k)[:k]
                        sims_k, users_k, ratings_k = sims_k[top], users_k[top], ratings_k[top]
                    dev = ratings_k - idx.baseline_vector(users_k, i, use_bias)
                    n = len(sims_k)
                    preds[t] = base + (n / (n + shrink_beta)) * float(np.sum(sims_k * dev) / np.sum(sims_k))
                    source[t] = "neighbours"; done = True
        if not done:
            preds[t], source[t] = base, ("baseline_bias" if use_bias else "user_mean")
    return pd.DataFrame({"prediction": np.clip(preds, VALID_RATING_MIN, VALID_RATING_MAX), "source": source}, index=test.index)


def predict_mean_baseline(test: pd.DataFrame, idx: RatingIndex, kind: str) -> pd.DataFrame:
    if kind == "global_mean":
        p = np.full(len(test), idx.global_mean)
    elif kind == "item_mean":
        p = idx.item_mean.reindex(test["itemID"]).fillna(idx.global_mean).to_numpy()
    elif kind == "user_mean":
        p = idx.user_mean.reindex(test["userID"]).fillna(idx.global_mean).to_numpy()
    elif kind == "baseline_bias":
        p = np.array([idx.baseline(u, i, True) for u, i in zip(test["userID"], test["itemID"])])
    else:
        raise ValueError(kind)
    return pd.DataFrame({"prediction": np.clip(p, VALID_RATING_MIN, VALID_RATING_MAX), "source": kind}, index=test.index)


def pearson_user_similarity(idx: RatingIndex) -> pd.DataFrame:
    """Conventional rating-based user similarity (Pearson on co-rated
    items), the baseline the approach is compared with."""
    piv = idx.pivot
    centred = piv.sub(piv.mean(axis=1), axis=0)
    x = centred.fillna(0.0).to_numpy(); mask = (~centred.isna()).to_numpy().astype(float)
    num = x @ x.T; sq = x ** 2
    den = np.sqrt((sq @ mask.T) * (mask @ sq.T))
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(den > 0, num / den, np.nan)
    np.fill_diagonal(sim, np.nan)
    return pd.DataFrame(sim, index=piv.index, columns=piv.index)


@njit(cache=True)
def _mf_epoch(order, tu, ti, tr, U, V, bu, bi, mu, lr, reg):
    for idx in range(order.shape[0]):
        n = order[idx]; a = tu[n]; b = ti[n]
        dot = 0.0
        for f in range(U.shape[1]):
            dot += U[a, f] * V[b, f]
        e = tr[n] - (mu + bu[a] + bi[b] + dot)
        bu[a] += lr * (e - reg * bu[a]); bi[b] += lr * (e - reg * bi[b])
        for f in range(U.shape[1]):
            ua = U[a, f]
            U[a, f] += lr * (e * V[b, f] - reg * U[a, f])
            V[b, f] += lr * (e * ua - reg * V[b, f])


def biased_mf_predict(train: pd.DataFrame, test: pd.DataFrame, reg: float, seed: int) -> pd.DataFrame:
    """Regularised biased matrix factorisation trained by SGD."""
    rng = np.random.default_rng(seed)
    users = {u: j for j, u in enumerate(train["userID"].unique())}
    items = {i: j for j, i in enumerate(train["itemID"].unique())}
    U = rng.normal(0, 0.1, (len(users), MF_FACTORS)); V = rng.normal(0, 0.1, (len(items), MF_FACTORS))
    bu = np.zeros(len(users)); bi = np.zeros(len(items)); mu = float(train[TARGET_COLUMN].mean())
    tu = train["userID"].map(users).to_numpy(); ti = train["itemID"].map(items).to_numpy(); tr = train[TARGET_COLUMN].to_numpy(dtype=float)
    order = np.arange(len(tr))
    tu = tu.astype(np.int64); ti = ti.astype(np.int64)
    for _ in range(MF_EPOCHS):
        rng.shuffle(order)
        _mf_epoch(order, tu, ti, tr, U, V, bu, bi, mu, MF_LR, reg)
    p = np.empty(len(test))
    for t, (u, i) in enumerate(zip(test["userID"], test["itemID"])):
        a, b = users.get(u), items.get(i)
        p[t] = mu + (bu[a] if a is not None else 0) + (bi[b] if b is not None else 0) + (U[a] @ V[b] if a is not None and b is not None else 0)
    return pd.DataFrame({"prediction": np.clip(p, VALID_RATING_MIN, VALID_RATING_MAX), "source": "biased_mf"}, index=test.index)


def score_predictions(test: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    err = pred["prediction"].to_numpy() - test[TARGET_COLUMN].to_numpy(dtype=float)
    src = pred["source"].to_numpy()
    if mask is not None:
        err, src = err[mask], src[mask]
    if len(err) == 0:
        return {"mae": np.nan, "rmse": np.nan, "coverage": np.nan, "n": 0}
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "coverage": float((src == "neighbours").mean()), "n": int(len(err))}


def loss_value(m: dict) -> float:
    return m["mae"] if PREDICTION_LOSS == "mae" else m["rmse"]


# =============================================================================
# PHASE 3 - CONTEXTUAL MODELS ON A TRAINING SET (PHASES 1-2 IN MEMORY)
# =============================================================================

def memberships_frame(dim_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    mem = model["memberships"]
    out = dim_df[["observationID", "userID", "itemID"]].copy()
    for c in range(mem.shape[1]):
        out[f"membership_cluster_{c + 1}"] = mem[:, c]
    return out


def fit_dimension_model(dim: str, ddf: pd.DataFrame, family: str, k: int, hp: dict, seed: int) -> dict | None:
    features = list(CONTEXT_DIMENSIONS[dim]["features"])
    if family == "modality_identity":
        n_mod = int(ddf[features[0]].nunique())
        if n_mod < 2 or (k is not None and k != n_mod):
            return None
        k = n_mod
    k = min(k, int(ddf[features].drop_duplicates().shape[0]))
    if k < 2:
        return None
    try:
        model = fit_candidate(family, ddf, features, k, dict(hp), seed)
    except Exception:
        return None
    labels = model["labels"]
    _, counts = np.unique(labels, return_counts=True)
    n_users = ddf["userID"].nunique()
    if len(counts) < 2 or counts.min() / counts.sum() < MIN_CLUSTER_FRACTION or len(counts) > max(MIN_CLUSTERS, int(MAX_CLUSTERS_FRACTION_OF_USERS * n_users)):
        return None
    return {"memberships": memberships_frame(ddf, model), "model": model, "features": features,
            "cost": build_representative_distance_matrix(family, model, ddf, features),
            "k": int(model["memberships"].shape[1]), "family": family, "hyperparams": hp}


def fit_contextual_models(train_clean: pd.DataFrame, selected: dict, k_per_dim: dict | None, mode: str, seed: int) -> dict:
    datasets, reports = prepare_context_dimensions(train_clean)
    fitted = {}
    for dim, ddf in datasets.items():
        if len(ddf) < 10 or ddf["userID"].nunique() < 3:
            continue
        if mode == "reselect":
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, model, info = run_dimension_search(dim, ddf, reports[dim])
            family, k, hp = info["selected_family"], info["selected_k"], info["selected_hyperparams"]
        else:
            cfg = selected[dim]
            family, k, hp = cfg["selected_family"], int(cfg["selected_k"]), cfg["selected_hyperparams"]
        if k_per_dim and dim in k_per_dim:
            k = k_per_dim[dim]
        entry = None
        # Robust fit: retry with other seeds, then with neighbouring K values
        # (both directions), rather than silently dropping a dimension that
        # was used during selection.
        k_max = max(K_CLUSTER_GRID_VALIDATION + [k])
        order = [k] + [v for step in range(1, k_max) for v in (k + step, k - step) if MIN_CLUSTERS <= v <= k_max]
        for k_try in order:
            for attempt in range(3):
                entry = fit_dimension_model(dim, ddf, family, k_try, hp, seed + 17 * attempt)
                if entry is not None:
                    break
            if entry is not None:
                break
        if entry is not None:
            fitted[dim] = entry
        else:
            print(f"  WARNING: dimension {dim} could not be fitted admissibly on this training set; it is dropped for this fold.")
    return fitted


def dimension_component(entry: dict, tau_factor: float) -> dict | None:
    prof = build_user_profiles(entry["memberships"], EPSILON_M)
    if prof["k"] < 2:
        return None
    dmats = compute_all_distance_matrices(prof["profiles"], entry["cost"])
    tau = float(max(prof["counts"].median() * tau_factor, 1.0))
    rel = user_reliability(prof["counts"], prof["confidence"], tau)
    return {"deltas": dmats, "pair_rel": pair_reliability_matrix(rel["reliability_rho_u"]),
            "users": prof["profiles"].index, "tau": tau, "k": prof["k"]}


def limit_contextual_evidence(fitted: dict, seed: int) -> dict:
    """'limited' regime: keep, per user and dimension, a random number of
    observations (uniform between EVIDENCE_MIN_OBS and N_u^d) for profile
    construction. Clustering itself is unchanged."""
    rng = np.random.default_rng(seed)
    out = {}
    for dim, entry in fitted.items():
        mem = entry["memberships"]
        keep = np.zeros(len(mem), dtype=bool)
        for _, pos in mem.groupby("userID").indices.items():
            pos = np.array(pos); rng.shuffle(pos)
            n_keep = int(rng.integers(EVIDENCE_MIN_OBS, len(pos) + 1))
            keep[pos[:n_keep]] = True
        e = dict(entry); e["memberships"] = mem.loc[keep].reset_index(drop=True)
        out[dim] = e
    return out


def apply_cold_start(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """'cold_start' regime: a share of users keeps only COLD_START_KEEP
    training observations; the others are moved to the test fold and
    flagged. Returns (train, test, is_cold flag aligned with test)."""
    rng = np.random.default_rng(seed)
    users = train["userID"].unique(); rng.shuffle(users)
    cold = set(users[: int(round(COLD_START_USER_FRACTION * len(users)))])
    moved = []
    keep_mask = np.ones(len(train), dtype=bool)
    for u, pos in train.groupby("userID").indices.items():
        if u in cold and len(pos) > COLD_START_KEEP:
            pos = np.array(pos); rng.shuffle(pos)
            keep_mask[pos[COLD_START_KEEP:]] = False
            moved.append(train.iloc[pos[COLD_START_KEEP:]])
    new_train = train.loc[keep_mask].copy()
    new_test = pd.concat([test] + moved) if moved else test.copy()
    is_cold = new_test["userID"].isin(cold).to_numpy()
    return new_train, new_test, is_cold


def contextual_similarity_components(fitted: dict, tau_factor: float = 1.0) -> dict:
    comp = {}
    for dim, entry in fitted.items():
        c = dimension_component(entry, tau_factor)
        if c is not None:
            comp[dim] = c
    return comp


def user_evidence_heterogeneity(comp: dict) -> pd.Series:
    """Spread of the user reliability rho_u^d across dimensions (max - min),
    with missing dimensions counted as rho = 0."""
    dims = list(comp.keys())
    users = pd.Index(sorted(set().union(*[set(c["users"]) for c in comp.values()])))
    rho = pd.DataFrame(0.0, index=users, columns=dims)
    for d, c in comp.items():
        r = np.sqrt(np.diag(c["pair_rel"].to_numpy()))   # r_uu = rho_u
        rho.loc[c["users"], d] = r
    return rho.max(axis=1) - rho.min(axis=1)


def ranking_metrics(test: pd.DataFrame, pred: pd.DataFrame, mask: np.ndarray | None = None) -> dict:
    """Precision@k, Recall@k and NDCG@k per user over the user's test items."""
    frame = pd.DataFrame({"userID": test["userID"].to_numpy(), "rating": test[TARGET_COLUMN].to_numpy(dtype=float),
                          "score": pred["prediction"].to_numpy()})
    if mask is not None:
        frame = frame.loc[mask]
    precs, recs, ndcgs = [], [], []
    for _, g in frame.groupby("userID"):
        if len(g) < RANKING_MIN_TEST_ITEMS:
            continue
        rel = (g["rating"] >= RELEVANCE_THRESHOLD).to_numpy()
        if rel.sum() == 0 or rel.sum() == len(rel):
            continue
        order = np.argsort(-g["score"].to_numpy(), kind="stable")
        topk = rel[order][:RANKING_K]
        precs.append(topk.mean()); recs.append(topk.sum() / rel.sum())
        dcg = np.sum(topk / np.log2(np.arange(2, len(topk) + 2)))
        ideal = np.sort(rel)[::-1][:RANKING_K]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    if not precs:
        return {"precision_at_k": np.nan, "recall_at_k": np.nan, "ndcg_at_k": np.nan, "n_users_ranked": 0}
    return {"precision_at_k": float(np.mean(precs)), "recall_at_k": float(np.mean(recs)), "ndcg_at_k": float(np.mean(ndcgs)), "n_users_ranked": len(precs)}


def aggregate_with_alpha(all_users: pd.Index, deltas: dict[str, pd.DataFrame], alpha: dict[str, float]) -> pd.DataFrame:
    """Prediction-aware aggregation: global coefficients renormalised over
    D_uv (Eq. prediction-aware-weight), uniform fallback when A_uv = 0."""
    n = len(all_users); pos = {u: i for i, u in enumerate(all_users)}
    dims = list(deltas.keys())
    stack = np.full((len(dims), n, n), np.nan)
    for a, d in enumerate(dims):
        ix = np.array([pos[u] for u in deltas[d].index])
        stack[a][np.ix_(ix, ix)] = deltas[d].to_numpy()
    avail = ~np.isnan(stack); n_shared = avail.sum(axis=0)
    a_vec = np.array([alpha.get(d, 0.0) for d in dims])[:, None, None]
    w = np.where(avail, a_vec, 0.0); total = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(total > 0, w / total, np.where(avail, 1.0 / np.maximum(n_shared, 1), 0.0))
    delta = np.nansum(np.where(avail, stack, 0.0) * w, axis=0)
    delta[n_shared == 0] = np.nan; np.fill_diagonal(delta, np.nan)
    return pd.DataFrame(1.0 - delta, index=all_users, columns=all_users)


def similarity_for_scheme(comp: dict, distances: dict[str, str], scheme: str, alpha: dict | None = None, gamma: float = 1.0) -> pd.DataFrame:
    deltas = {}
    for d, c in comp.items():
        name = distances.get(d, FALLBACK_DISTANCE)
        deltas[d] = c["deltas"][name if name in c["deltas"] else FALLBACK_DISTANCE]
    users = pd.Index(sorted(set().union(*[set(c["users"]) for c in comp.values()])))
    if scheme == "pred":
        S = aggregate_with_alpha(users, deltas, alpha or {})
    else:
        agg = aggregate_dissimilarities(users, deltas, {d: c["pair_rel"] for d, c in comp.items()})
        S = agg["similarity_unif"] if scheme == "unif" else agg["similarity_rel"]
    # contrast transform: monotone, keeps neighbour ranking, sharpens weights
    return S.clip(lower=0.0) ** gamma if gamma != 1.0 else S


def hybrid_similarity(S_context: pd.DataFrame, S_rating: pd.DataFrame, lam: float) -> pd.DataFrame:
    """Convex combination of the positive-part rating similarity and the
    contextual similarity, renormalised over available components."""
    users = pd.Index(sorted(set(S_context.index) | set(S_rating.index)))
    C = S_context.reindex(index=users, columns=users).to_numpy(dtype=float)
    R = S_rating.reindex(index=users, columns=users).to_numpy(dtype=float)
    R = np.where(np.isnan(R), np.nan, np.clip(R, 0.0, None))
    wc, wr = (1.0 - lam) * ~np.isnan(C), lam * ~np.isnan(R)
    total = wc + wr
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (np.nan_to_num(C) * wc + np.nan_to_num(R) * wr) / total
    S[total <= 0] = np.nan
    np.fill_diagonal(S, np.nan)
    return pd.DataFrame(S, index=users, columns=users)


def weighting_diagnostics(comp: dict, distances: dict[str, str]) -> dict:
    """How much room the weighting schemes have: dispersion of the
    reliability-aware weights across pairs, disagreement of the
    dimension-specific dissimilarities, agreement between S_unif and S_rel."""
    deltas = {d: c["deltas"][distances.get(d, FALLBACK_DISTANCE) if distances.get(d, FALLBACK_DISTANCE) in c["deltas"] else FALLBACK_DISTANCE] for d, c in comp.items()}
    users = pd.Index(sorted(set().union(*[set(c["users"]) for c in comp.values()])))
    agg = aggregate_dissimilarities(users, deltas, {d: c["pair_rel"] for d, c in comp.items()})
    ns = upper_values(agg["n_shared_dimensions"])
    out = {"weights_rel_std_across_pairs": {}, "weights_rel_mean": {}}
    for d in agg["dimensions"]:
        w = upper_values(agg["weights_rel"][d])[ns > 0]
        out["weights_rel_std_across_pairs"][d] = float(np.std(w)); out["weights_rel_mean"][d] = float(np.mean(w))
    stack = []
    for d in agg["dimensions"]:
        m = deltas[d].reindex(index=users, columns=users)
        stack.append(upper_values(m))
    stack = np.array(stack)
    full = ~np.isnan(stack).any(axis=0)
    out["delta_std_across_dimensions_mean"] = float(np.mean(np.std(stack[:, full], axis=0))) if full.any() else np.nan
    out["delta_between_dimension_correlation"] = {}
    dims = agg["dimensions"]
    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            out["delta_between_dimension_correlation"][f"{dims[i]}~{dims[j]}"] = float(np.corrcoef(stack[i, full], stack[j, full])[0, 1]) if full.sum() > 2 else np.nan
    su, sr = upper_values(agg["similarity_unif"]), upper_values(agg["similarity_rel"])
    ok = ~np.isnan(su) & ~np.isnan(sr)
    out["similarity_unif_std"] = float(np.std(su[ok])); out["corr_similarity_unif_rel"] = float(np.corrcoef(su[ok], sr[ok])[0, 1]) if ok.sum() > 2 else np.nan
    return out


# =============================================================================
# PHASE 3b - SIMILARITY-REGULARISED MATRIX FACTORISATION (hybridisation)
# =============================================================================
#
# Prediction  r_hat = mu + b_u + b_i + p_u . q_i  [+ context-cluster biases]
# Objective   sum (r - r_hat)^2 + reg * ||params||^2
#             + beta * sum_u rho_u * || p_u - sum_v w_uv p_v ||^2
# with w_uv = S_uv^q / sum_v S_uv^q over the contextual neighbours of u
# (positive, defined similarity), S^q being the Phase-2 similarity under
# weighting scheme q in {unif, rel, pred}. rho_u (mean user reliability
# over dimensions) modulates the regularisation strength: users with
# well-supported contextual profiles are tied more strongly to their
# contextual neighbours. The three weighting schemes therefore define
# three regularisation graphs of the same model.
#
# Optional context-cluster biases (extension of the approach: the fuzzy
# clusters of the TARGET observation enter the prediction):
#   + sum_d sum_k m_{n,k}^d ( b_{u,k}^d + b_{i,k}^d )
# where m_{n,k}^d is the soft membership of the observation to cluster k
# of dimension d (Phase-1 models applied to the observation's context).

SRMF_FACTORS = 10
SRMF_EPOCHS = 30
SRMF_LR = 0.01
SRMF_BETA_GRID = [0.0, 0.3, 1.0, 3.0]
SRMF_ALPHA_GRID_STEP = 0.2
SRMF_USE_RELIABILITY_STRENGTH = True
CONTEXT_BIAS_VARIANTS = [False, True]     # plain SRMF and SRMF + context-cluster biases


def assign_memberships(entry: dict, df_new: pd.DataFrame) -> np.ndarray:
    """Soft memberships of new observations under a fitted Phase-1 model.
    Observations with missing context in the dimension, or unseen
    modalities, receive a uniform membership (no contextual information)."""
    model, family, features = entry["model"], entry["family"], entry["features"]
    k = model["memberships"].shape[1]
    n = len(df_new)
    out = np.full((n, k), 1.0 / k)
    complete = df_new[features].notna().all(axis=1).to_numpy()
    if not complete.any():
        return out
    x = df_new.loc[complete, features].to_numpy(dtype=float).astype(int)

    if family == "modality_identity":
        values = model["modes"][:, 0]
        pos = {int(v): j for j, v in enumerate(values)}
        mem = np.full((len(x), k), 1.0 / k)
        for r, v in enumerate(x[:, 0]):
            if int(v) in pos:
                mem[r] = 0.0; mem[r, pos[int(v)]] = 1.0
    elif family == "fuzzy_k_modes":
        mem = memberships_from_dissimilarities(matching_distance_to_modes(x, model["modes"].astype(int)), entry["hyperparams"].get("fuzziness", 2.0))
    elif family == "fuzzy_k_medoids_gower":
        med = model["medoids"].astype(int)
        d = np.zeros((len(x), k))
        for j, f in enumerate(features):
            if FEATURE_TYPES.get(f, "nominal") == "ordinal":
                vals = np.sort(np.unique(np.concatenate([x[:, j], med[:, j]]))); span = max(len(vals) - 1, 1)
                d += np.abs(np.searchsorted(vals, x[:, j])[:, None] - np.searchsorted(vals, med[:, j])[None, :]) / span
            else:
                d += (x[:, j][:, None] != med[:, j][None, :]).astype(float)
        mem = memberships_from_dissimilarities(d / len(features), entry["hyperparams"].get("fuzziness", 2.0))
    elif family in ("latent_class_analysis", "spectral_lca_hybrid"):
        logp = np.tile(np.log(np.clip(model["priors"], 1e-15, 1)), (len(x), 1))
        for j in range(len(features)):
            cats = {int(float(c)): idx for idx, c in enumerate(model["category_values"][j])}
            theta = model["conditional_probabilities"][j]
            for r, v in enumerate(x[:, j]):
                if int(v) in cats:
                    logp[r] += np.log(np.clip(theta[:, cats[int(v)]], 1e-15, 1))
        logp -= logp.max(axis=1, keepdims=True)
        mem = np.exp(logp); mem /= mem.sum(axis=1, keepdims=True)
    else:
        mem = np.full((len(x), k), 1.0 / k)
    out[complete] = mem
    return out


def context_membership_blocks(fitted: dict, df: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Concatenated soft memberships of the observations of df over all
    fitted dimensions (n x sum_d K_d), plus the (dimension, K_d) layout."""
    blocks, layout = [], []
    for dim, entry in fitted.items():
        m = assign_memberships(entry, df)
        blocks.append(m); layout.append((dim, m.shape[1]))
    if not blocks:
        return np.zeros((len(df), 0)), []
    return np.hstack(blocks), layout


def user_reliability_vector(comp: dict, users: pd.Index) -> np.ndarray:
    rho = pd.DataFrame(0.0, index=users, columns=list(comp.keys()))
    for d, c in comp.items():
        r = pd.Series(np.sqrt(np.diag(c["pair_rel"].to_numpy())), index=c["users"])
        rho.loc[rho.index.intersection(r.index), d] = r.reindex(rho.index.intersection(r.index)).to_numpy()
    return rho.mean(axis=1).to_numpy()


def fit_srmf(train: pd.DataFrame, S: pd.DataFrame | None, reg: float, beta: float, seed: int,
             rho_by_user: pd.Series | None = None, ctx_train: np.ndarray | None = None) -> dict:
    """Similarity-regularised biased MF trained by SGD (ratings) with a
    batch graph-regularisation step per epoch (users). ctx_train: optional
    concatenated soft memberships of the training observations for the
    context-cluster bias terms."""
    rng = np.random.default_rng(seed)
    users = {u: j for j, u in enumerate(sorted(train["userID"].unique()))}
    items = {i: j for j, i in enumerate(sorted(train["itemID"].unique()))}
    nU, nI, F = len(users), len(items), SRMF_FACTORS
    P = rng.normal(0, 0.1, (nU, F)); Q = rng.normal(0, 0.1, (nI, F))
    bu = np.zeros(nU); bi = np.zeros(nI); mu = float(train[TARGET_COLUMN].mean())
    use_ctx = ctx_train is not None and ctx_train.shape[1] > 0
    C = ctx_train.shape[1] if use_ctx else 0
    BU = np.zeros((nU, C)); BI = np.zeros((nI, C))

    tu = train["userID"].map(users).to_numpy(); ti = train["itemID"].map(items).to_numpy(); tr = train[TARGET_COLUMN].to_numpy(dtype=float)

    # regularisation graph W (row-normalised positive similarities) and strength
    W = None
    if S is not None and beta > 0:
        idx_users = list(users.keys())
        Sm = S.reindex(index=idx_users, columns=idx_users).to_numpy(dtype=float)
        Sm = np.where(np.isnan(Sm), 0.0, np.clip(Sm, 0.0, None)); np.fill_diagonal(Sm, 0.0)
        rs = Sm.sum(axis=1)
        W = np.where(rs[:, None] > 0, Sm / np.maximum(rs[:, None], 1e-12), 0.0)
        has_nb = rs > 0
        strength = np.ones(nU)
        if rho_by_user is not None and SRMF_USE_RELIABILITY_STRENGTH:
            strength = rho_by_user.reindex(idx_users).fillna(0.0).to_numpy()
        strength = strength * has_nb

    order = np.arange(len(tr))
    for _ in range(SRMF_EPOCHS):
        rng.shuffle(order)
        for n in order:
            a, b = tu[n], ti[n]
            pred = mu + bu[a] + bi[b] + P[a] @ Q[b]
            if use_ctx:
                pred += ctx_train[n] @ (BU[a] + BI[b])
            e = tr[n] - pred
            bu[a] += SRMF_LR * (e - reg * bu[a]); bi[b] += SRMF_LR * (e - reg * bi[b])
            pa = P[a].copy()
            P[a] += SRMF_LR * (e * Q[b] - reg * P[a]); Q[b] += SRMF_LR * (e * pa - reg * Q[b])
            if use_ctx:
                m = ctx_train[n]
                BU[a] += SRMF_LR * (e * m - reg * BU[a]); BI[b] += SRMF_LR * (e * m - reg * BI[b])
        if W is not None:
            D = P - W @ P                                   # p_u - sum_v w_uv p_v
            G = (strength[:, None] * D) - W.T @ (strength[:, None] * D)
            P -= SRMF_LR * beta * G
    return {"users": users, "items": items, "P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu, "BU": BU, "BI": BI, "use_ctx": use_ctx}


def predict_srmf(model: dict, test: pd.DataFrame, ctx_test: np.ndarray | None = None) -> pd.DataFrame:
    p = np.empty(len(test))
    for t, (u, i) in enumerate(zip(test["userID"], test["itemID"])):
        a, b = model["users"].get(u), model["items"].get(i)
        v = model["mu"] + (model["bu"][a] if a is not None else 0) + (model["bi"][b] if b is not None else 0)
        if a is not None and b is not None:
            v += model["P"][a] @ model["Q"][b]
        if model["use_ctx"] and ctx_test is not None and (a is not None or b is not None):
            cb = np.zeros(ctx_test.shape[1])
            if a is not None:
                cb = cb + model["BU"][a]
            if b is not None:
                cb = cb + model["BI"][b]
            v += float(ctx_test[t] @ cb)
        p[t] = v
    return pd.DataFrame({"prediction": np.clip(p, VALID_RATING_MIN, VALID_RATING_MAX), "source": "srmf"}, index=test.index)


def srmf_method_name(scheme: str | None, ctx_bias: bool) -> str:
    base = "biased_mf" if scheme is None else f"srmf_{scheme}"
    return base + ("_cb" if ctx_bias else "")


# =============================================================================
# PHASE 3c - CONTEXTUAL FACTORISATION ON FUZZY CLUSTERS (CMF)
# =============================================================================
#
#   r_hat(u,i,n) = mu + b_u + b_i + p_u.q_i
#                  + sum_d alpha_d sum_k m_{n,k}^d ( b_{u,k}^d + b_{i,k}^d + p_{u,k}^d . q_i )
#
# m_{n,k}^d : soft membership of the TARGET observation n to cluster k of
#             dimension d (Phase-1 models applied to its context).
# alpha_d   : combination of the contextual dimensions in the prediction
#             "unif" -> 1/D ; "pred" -> learned on the simplex through the
#             complete pipeline (Eq. prediction-aware-optimization).
# rho_u^d   : user reliability in dimension d (observation support x
#             membership confidence, Phase 2). "rel" -> the contextual
#             user parameters (b_{u,k}^d, p_{u,k}^d) are regularised
#             towards the global ones with strength reg*(1 + kappa*(1-rho_u^d)):
#             users with weak or ambiguous contextual evidence in a
#             dimension fall back to their global behaviour there.
#
# The three weighting schemes therefore combine the contextual DIMENSIONS
# inside the prediction (pred: how much each dimension contributes; rel:
# how much each user's contextual deviations can be trusted per
# dimension), instead of combining user-user dissimilarities.

CMF_FACTORS = 10
CMF_EPOCHS = 40
CMF_LR = 0.01
CMF_KAPPA_GRID = [1.0, 3.0, 10.0]
CMF_ALPHA_GRID_STEP = 0.25
CMF_SCHEMES = ["unif", "rel", "relw", "pred"]
# "relw": reliability-aware as per-user DIMENSION WEIGHTS in the prediction,
#         w_{u,d} = rho_u^d / sum_d' rho_u^d' (uniform fallback when all zero),
#         the direct counterpart of the pair-specific reliability weights of
#         the approach; it only has room where the contextual evidence of a
#         user differs across dimensions (missing or weakly supported ones).
ENABLE_HYBRID_KNN = False
ENABLE_SRMF = False
ENABLE_CMF = True


def column_weights(layout: list[tuple[str, int]], alpha: dict[str, float] | None) -> np.ndarray:
    """Per-column alpha_d for the concatenated membership blocks."""
    if not layout:
        return np.zeros(0)
    D = len(layout)
    w = []
    for d, k in layout:
        a = (1.0 / D) if alpha is None else float(alpha.get(d, 0.0))
        w.extend([a] * k)
    return np.asarray(w)


def column_dimension(layout: list[tuple[str, int]]) -> list[str]:
    out = []
    for d, k in layout:
        out.extend([d] * k)
    return out


def user_reliability_by_dimension(comp: dict, users: pd.Index) -> pd.DataFrame:
    rho = pd.DataFrame(0.0, index=users, columns=list(comp.keys()))
    for d, c in comp.items():
        r = pd.Series(np.sqrt(np.diag(c["pair_rel"].to_numpy())), index=c["users"])
        common = rho.index.intersection(r.index)
        rho.loc[common, d] = r.reindex(common).to_numpy()
    return rho


@njit(cache=True)
def _cmf_epoch(order, tu, ti, tr, ctx, A, P, Q, bu, bi, BU, BI, PU, reg_u, mu, lr, reg):
    F = P.shape[1]; C = ctx.shape[1]
    w = np.empty(C); pu_ctx = np.empty(F); qb = np.empty(F)
    for idx in range(order.shape[0]):
        n = order[idx]; a = tu[n]; b = ti[n]
        for c in range(C):
            w[c] = A[a, c] * ctx[n, c]
        for f in range(F):
            s = 0.0
            for c in range(C):
                s += w[c] * PU[a, c, f]
            pu_ctx[f] = s
        pred = mu + bu[a] + bi[b]
        for f in range(F):
            pred += (P[a, f] + pu_ctx[f]) * Q[b, f]
        for c in range(C):
            pred += w[c] * (BU[a, c] + BI[b, c])
        e = tr[n] - pred
        bu[a] += lr * (e - reg * bu[a]); bi[b] += lr * (e - reg * bi[b])
        for f in range(F):
            qb[f] = Q[b, f]
        for f in range(F):
            Q[b, f] += lr * (e * (P[a, f] + pu_ctx[f]) - reg * Q[b, f])
            P[a, f] += lr * (e * qb[f] - reg * P[a, f])
        for c in range(C):
            BU[a, c] += lr * (e * w[c] - reg_u[a, c] * BU[a, c])
            BI[b, c] += lr * (e * w[c] - reg * BI[b, c])
            for f in range(F):
                PU[a, c, f] += lr * (e * w[c] * qb[f] - reg_u[a, c] * PU[a, c, f])


def user_dimension_weights(rho_by_user_dim: pd.DataFrame, layout: list[tuple[str, int]], users: list) -> np.ndarray:
    """Per-user per-column weights w_{u,d} = rho_u^d / sum_d rho_u^d (uniform when all zero)."""
    dims = [d for d, _ in layout]
    rho = rho_by_user_dim.reindex(users)[[d for d in dims if d in rho_by_user_dim.columns]].fillna(0.0)
    for d in dims:
        if d not in rho.columns:
            rho[d] = 0.0
    rho = rho[dims].to_numpy(dtype=float)
    tot = rho.sum(axis=1, keepdims=True)
    w = np.where(tot > 0, rho / np.maximum(tot, 1e-12), 1.0 / len(dims))
    cols = []
    for j, (d, k) in enumerate(layout):
        cols.append(np.repeat(w[:, j:j + 1], k, axis=1))
    return np.hstack(cols) if cols else np.zeros((len(users), 0))


def fit_cmf(train: pd.DataFrame, ctx_train: np.ndarray, layout: list[tuple[str, int]], alpha: dict | None,
            rho_by_user_dim: pd.DataFrame | None, kappa: float, reg: float, seed: int,
            per_user_weights: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    users = {u: j for j, u in enumerate(sorted(train["userID"].unique()))}
    items = {i: j for j, i in enumerate(sorted(train["itemID"].unique()))}
    nU, nI, F = len(users), len(items), CMF_FACTORS
    C = ctx_train.shape[1]
    P = rng.normal(0, 0.1, (nU, F)); Q = rng.normal(0, 0.1, (nI, F))
    bu = np.zeros(nU); bi = np.zeros(nI); mu = float(train[TARGET_COLUMN].mean())
    BU = np.zeros((nU, C)); BI = np.zeros((nI, C)); PU = np.zeros((nU, C, F))
    a_col = column_weights(layout, alpha)
    if per_user_weights and rho_by_user_dim is not None:
        A = user_dimension_weights(rho_by_user_dim, layout, list(users.keys()))
    else:
        A = np.tile(a_col, (nU, 1))
    A = np.ascontiguousarray(A, dtype=float)
    # per-user, per-column regularisation multiplier (reliability-aware, regularisation form)
    reg_u = np.full((nU, C), reg)
    if rho_by_user_dim is not None and kappa > 0:
        cols_dim = column_dimension(layout)
        rho_m = rho_by_user_dim.reindex(list(users.keys())).fillna(0.0)
        for c, d in enumerate(cols_dim):
            reg_u[:, c] = reg * (1.0 + kappa * (1.0 - rho_m[d].to_numpy() if d in rho_m else 1.0))
    tu = train["userID"].map(users).to_numpy().astype(np.int64); ti = train["itemID"].map(items).to_numpy().astype(np.int64)
    tr = train[TARGET_COLUMN].to_numpy(dtype=float)
    ctx_c = np.ascontiguousarray(ctx_train, dtype=float); a_col = np.ascontiguousarray(a_col, dtype=float)
    order = np.arange(len(tr))
    for _ in range(CMF_EPOCHS):
        rng.shuffle(order)
        _cmf_epoch(order, tu, ti, tr, ctx_c, A, P, Q, bu, bi, BU, BI, PU, reg_u, mu, CMF_LR, reg)
    return {"users": users, "items": items, "P": P, "Q": Q, "bu": bu, "bi": bi, "mu": mu, "BU": BU, "BI": BI, "PU": PU,
            "a_col": a_col, "A": A}


def predict_cmf(model: dict, test: pd.DataFrame, ctx_test: np.ndarray) -> pd.DataFrame:
    p = np.empty(len(test))
    for t, (u, i) in enumerate(zip(test["userID"], test["itemID"])):
        a, b = model["users"].get(u), model["items"].get(i)
        w = (model["A"][a] if a is not None else model["a_col"]) * ctx_test[t]
        v = model["mu"] + (model["bu"][a] if a is not None else 0.0) + (model["bi"][b] if b is not None else 0.0)
        if a is not None:
            v += float(w @ model["BU"][a])
        if b is not None:
            v += float(w @ model["BI"][b])
        if a is not None and b is not None:
            v += float((model["P"][a] + w @ model["PU"][a]) @ model["Q"][b])
        p[t] = v
    return pd.DataFrame({"prediction": np.clip(p, VALID_RATING_MIN, VALID_RATING_MAX), "source": "cmf"}, index=test.index)


# =============================================================================
# PHASE 3 - INNER VALIDATION
# =============================================================================

def assign_folds(df: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(df), dtype=int)
    for _, pos in df.groupby("userID").indices.items():
        pos = np.array(pos); rng.shuffle(pos)
        fold[pos] = (np.arange(len(pos)) + rng.integers(n_folds)) % n_folds
    return fold


def simplex_grid(dims: list[str], step: float) -> list[dict]:
    n = int(round(1.0 / step)); out = []
    def rec(prefix, remaining, depth):
        if depth == len(dims) - 1:
            out.append(prefix + [remaining]); return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, depth + 1)
    rec([], n, 0)
    return [{d: v / n for d, v in zip(dims, combo)} for combo in out]


def _cmf_alpha_objective(args):
    itrain, ctx_tr_c, layout, alpha, mf_reg, seed, ival, ctx_va_c, uniform, dims = args
    m = fit_cmf(itrain, ctx_tr_c, layout, alpha, None, 0.0, mf_reg, seed)
    return loss_value(score_predictions(ival, predict_cmf(m, ival, ctx_va_c))) + LAMBDA_ALPHA * sum((alpha[d] - uniform) ** 2 for d in dims)


def inner_validation(train_clean: pd.DataFrame, selected: dict, seed: int, regime: str = "full") -> dict:
    rng = np.random.default_rng(seed)
    is_val = np.zeros(len(train_clean), dtype=bool)
    for _, pos in train_clean.groupby("userID").indices.items():
        pos = np.array(pos); rng.shuffle(pos)
        n_val = int(round(INNER_VALIDATION_FRACTION * len(pos)))
        if len(pos) >= 3 and n_val >= 1:
            is_val[pos[:n_val]] = True
    itrain, ival = train_clean.loc[~is_val].copy(), train_clean.loc[is_val].copy()
    k_ref = K_NEIGHBORS_GRID[len(K_NEIGHBORS_GRID) // 2]

    def ev(S, idx, k, beta, use_bias=True, ctx=False):
        return loss_value(score_predictions(ival, predict_with_similarity(ival, idx, S, k, beta, use_bias, ctx)))

    def prepare(fitted_models):
        return limit_contextual_evidence(fitted_models, seed + 7) if regime == "limited" else fitted_models

    # ---- (a) baseline regularisation and shrinkage (with the exploratory K_d, uniform scheme)
    fitted = prepare(fit_contextual_models(itrain, selected, None, FOLD_CLUSTERING_MODE, seed))
    comp = contextual_similarity_components(fitted, 1.0)
    dims = list(comp.keys())
    distances = {d: DISTANCE_PER_DIMENSION.get(d, FALLBACK_DISTANCE) for d in dims}
    S0 = similarity_for_scheme(comp, distances, "unif")
    best, bias_reg, beta = np.inf, BIAS_REG_GRID[0], SHRINK_GRID[0]
    for br in BIAS_REG_GRID:
        idx_b = RatingIndex(itrain, br)
        for b in SHRINK_GRID:
            l = ev(S0, idx_b, k_ref, b)
            if l < best:
                best, bias_reg, beta = l, br, b
    idx = RatingIndex(itrain, bias_reg, PREFILTER_CONTEXT_VARS)

    # ---- (b) K_d per dimension, end-to-end (coordinate-wise), admissibility kept
    k_per_dim = {d: fitted[d]["k"] for d in dims}
    if SELECT_K_CLUSTERS_ON_VALIDATION:
        for d in dims:
            ddf = prepare_context_dimensions(itrain)[0][d]
            cfg = selected[d]
            best_k, best_l, best_entry = k_per_dim[d], np.inf, fitted[d]
            for kc in K_CLUSTER_GRID_VALIDATION:
                entry = fitted[d] if kc == fitted[d]["k"] else fit_dimension_model(d, ddf, fitted[d]["family"], kc, cfg["selected_hyperparams"], seed + kc)
                if entry is None:
                    continue
                if kc != fitted[d]["k"]:
                    entry = prepare({d: entry})[d]
                c = dimension_component(entry, 1.0)
                if c is None:
                    continue
                trial = dict(comp); trial[d] = c
                l = ev(similarity_for_scheme(trial, distances, "unif"), idx, k_ref, beta)
                if l < best_l - 1e-12:
                    best_k, best_l, best_entry = entry["k"], l, entry
            k_per_dim[d] = best_k; fitted[d] = best_entry; comp[d] = dimension_component(best_entry, 1.0)

    # ---- (c) distance per dimension (greedy)
    if SELECT_DISTANCE_ON_VALIDATION:
        for d in dims:
            best_name, best_l = distances[d], np.inf
            for name in ALL_DISTANCES:
                if name not in comp[d]["deltas"]:
                    continue
                trial = dict(distances); trial[d] = name
                l = ev(similarity_for_scheme(comp, trial, "unif"), idx, k_ref, beta)
                if l < best_l - 1e-12:
                    best_name, best_l = name, l
            distances[d] = best_name

    # ---- (d) contrast exponent gamma
    losses = {g: ev(similarity_for_scheme(comp, distances, "unif", gamma=g), idx, k_ref, beta) for g in GAMMA_GRID}
    gamma = min(losses, key=losses.get)

    # ---- (e) tau factor (reliability-aware scheme only)
    losses = {}
    for tf in TAU_FACTOR_GRID:
        comp_t = contextual_similarity_components(fitted, tf)
        losses[tf] = ev(similarity_for_scheme(comp_t, distances, "rel", gamma=gamma), idx, k_ref, beta)
    tau_factor = min(losses, key=losses.get)
    comp = contextual_similarity_components(fitted, tau_factor)

    # ---- (f) K neighbours for unif and rel
    k_best, val_loss = {}, {}
    for scheme in ("unif", "rel"):
        S = similarity_for_scheme(comp, distances, scheme, gamma=gamma)
        ls = {k: ev(S, idx, k, beta) for k in K_NEIGHBORS_GRID}
        k_best[scheme] = min(ls, key=ls.get); val_loss[scheme] = ls[k_best[scheme]]

    # ---- (g) prediction-aware alpha* through the complete pipeline
    uniform = 1.0 / len(dims); best = (np.inf, None, None)
    for alpha in simplex_grid(dims, ALPHA_GRID_STEP):
        S = similarity_for_scheme(comp, distances, "pred", alpha, gamma)
        pen = LAMBDA_ALPHA * sum((alpha[d] - uniform) ** 2 for d in dims)
        for k in K_NEIGHBORS_GRID:
            obj = ev(S, idx, k, beta) + pen
            if obj < best[0]:
                best = (obj, alpha, k)
    k_best["pred"], alpha_star, val_loss["pred"] = best[2], best[1], best[0]

    # ---- (h) baselines: Pearson KNN, context pre-filtering KNN, MF
    S_p = pearson_user_similarity(idx)

    # ---- (i) hybrid schemes: lambda and K per weighting scheme; alpha* through the hybrid pipeline
    hybrid = {}
    for scheme in (HYBRID_SCHEMES if ENABLE_HYBRID_KNN else []):
        S_ctx = similarity_for_scheme(comp, distances, scheme, alpha_star if scheme == "pred" else None, gamma)
        best_h = (np.inf, None, None, None)
        for lam in HYBRID_LAMBDA_GRID:
            S_h = hybrid_similarity(S_ctx, S_p, lam)
            for k in K_NEIGHBORS_GRID:
                for b in SHRINK_GRID:
                    l = ev(S_h, idx, k, b)
                    if l < best_h[0]:
                        best_h = (l, lam, k, b)
        hybrid[scheme] = {"lambda": best_h[1], "k": best_h[2], "shrink": best_h[3], "alpha": alpha_star if scheme == "pred" else None}
        val_loss[f"hybrid_{scheme}"] = best_h[0]
    if ENABLE_HYBRID_KNN and hybrid["pred"]["lambda"] is not None and hybrid["pred"]["lambda"] < 1.0:
        h = hybrid["pred"]; best_a = (np.inf, alpha_star)
        for alpha in simplex_grid(dims, ALPHA_GRID_STEP):
            S_h = hybrid_similarity(similarity_for_scheme(comp, distances, "pred", alpha, gamma), S_p, h["lambda"])
            pen = LAMBDA_ALPHA * sum((alpha[d] - uniform) ** 2 for d in dims)
            obj = ev(S_h, idx, h["k"], h["shrink"]) + pen
            if obj < best_a[0]:
                best_a = (obj, alpha)
        hybrid["pred"]["alpha"] = best_a[1]; val_loss["hybrid_pred"] = best_a[0]
    bl = {}
    for name, ctx in (("user_knn_pearson", False), ("context_prefilter_knn", True)):
        best_b = (np.inf, None, None, None)
        for br in BIAS_REG_GRID:
            idx_b = RatingIndex(itrain, br, PREFILTER_CONTEXT_VARS)
            for b in SHRINK_GRID:
                for k in K_NEIGHBORS_GRID:
                    l = ev(S_p, idx_b, k, b, True, ctx)
                    if l < best_b[0]:
                        best_b = (l, br, b, k)
        bl[name] = {"bias_reg": best_b[1], "shrink": best_b[2], "k": best_b[3]}; val_loss[name] = best_b[0]
    ls = {reg: loss_value(score_predictions(ival, biased_mf_predict(itrain, ival, reg, seed))) for reg in MF_REG_GRID}
    bl["biased_mf"] = {"reg": min(ls, key=ls.get)}; val_loss["biased_mf"] = ls[bl["biased_mf"]["reg"]]

    # ---- (j) similarity-regularised MF: beta per scheme and context-bias variant; alpha* through the MF pipeline
    mf_reg = bl["biased_mf"]["reg"]
    users_all = pd.Index(sorted(itrain["userID"].unique()))
    rho_u = pd.Series(user_reliability_vector(comp, users_all), index=users_all)
    ctx_tr, _ = context_membership_blocks(fitted, itrain); ctx_va, _ = context_membership_blocks(fitted, ival)
    srmf = {}
    for ctx_bias in (CONTEXT_BIAS_VARIANTS if ENABLE_SRMF else []):
        ct, cv = (ctx_tr, ctx_va) if ctx_bias else (None, None)
        # plain MF with context biases (no graph)
        if ctx_bias:
            m = fit_srmf(itrain, None, mf_reg, 0.0, seed, None, ct)
            val_loss[srmf_method_name(None, True)] = loss_value(score_predictions(ival, predict_srmf(m, ival, cv)))
        for scheme in WEIGHTING_SCHEMES:
            S_q = similarity_for_scheme(comp, distances, scheme, alpha_star if scheme == "pred" else None, gamma)
            best_b = (np.inf, 0.0)
            for b in SRMF_BETA_GRID:
                m = fit_srmf(itrain, S_q, mf_reg, b, seed, rho_u, ct)
                l = loss_value(score_predictions(ival, predict_srmf(m, ival, cv)))
                if l < best_b[0]:
                    best_b = (l, b)
            entry = {"beta": best_b[1], "reg": mf_reg, "alpha": alpha_star if scheme == "pred" else None}
            if scheme == "pred" and best_b[1] > 0:
                best_a = (best_b[0], alpha_star)
                for alpha in simplex_grid(dims, SRMF_ALPHA_GRID_STEP):
                    S_a = similarity_for_scheme(comp, distances, "pred", alpha, gamma)
                    m = fit_srmf(itrain, S_a, mf_reg, best_b[1], seed, rho_u, ct)
                    obj = loss_value(score_predictions(ival, predict_srmf(m, ival, cv))) + LAMBDA_ALPHA * sum((alpha[d] - uniform) ** 2 for d in dims)
                    if obj < best_a[0]:
                        best_a = (obj, alpha)
                entry["alpha"] = best_a[1]; best_b = (best_a[0], best_b[1])
            srmf[srmf_method_name(scheme, ctx_bias)] = entry
            val_loss[srmf_method_name(scheme, ctx_bias)] = best_b[0]

    # ---- (k) contextual factorisation on fuzzy clusters: kappa (rel) and alpha (pred) through the CMF pipeline
    cmf = {}
    if ENABLE_CMF:
        ctx_tr_c, layout = context_membership_blocks(fitted, itrain); ctx_va_c, _ = context_membership_blocks(fitted, ival)
        rho_ud = user_reliability_by_dimension(comp, users_all)
        m = fit_cmf(itrain, ctx_tr_c, layout, None, None, 0.0, mf_reg, seed)
        cmf["unif"] = {"reg": mf_reg, "kappa": 0.0, "alpha": None}
        val_loss["cmf_unif"] = loss_value(score_predictions(ival, predict_cmf(m, ival, ctx_va_c)))
        best_k = (np.inf, CMF_KAPPA_GRID[0])
        for kap in CMF_KAPPA_GRID:
            m = fit_cmf(itrain, ctx_tr_c, layout, None, rho_ud, kap, mf_reg, seed)
            l = loss_value(score_predictions(ival, predict_cmf(m, ival, ctx_va_c)))
            if l < best_k[0]:
                best_k = (l, kap)
        cmf["rel"] = {"reg": mf_reg, "kappa": best_k[1], "alpha": None}; val_loss["cmf_rel"] = best_k[0]
        m = fit_cmf(itrain, ctx_tr_c, layout, None, rho_ud, 0.0, mf_reg, seed, per_user_weights=True)
        cmf["relw"] = {"reg": mf_reg, "kappa": 0.0, "alpha": None}
        val_loss["cmf_relw"] = loss_value(score_predictions(ival, predict_cmf(m, ival, ctx_va_c)))
        cands = simplex_grid(dims, CMF_ALPHA_GRID_STEP)
        objs = parallel_map(_cmf_alpha_objective, [(itrain, ctx_tr_c, layout, alpha, mf_reg, seed, ival, ctx_va_c, uniform, dims) for alpha in cands],
                            n_jobs=1 if N_JOBS_INNER == 1 else N_JOBS_INNER)
        best_i = int(np.argmin(objs))
        cmf["pred"] = {"reg": mf_reg, "kappa": 0.0, "alpha": cands[best_i]}; val_loss["cmf_pred"] = float(objs[best_i])

    return {"dimensions": dims, "k_clusters": k_per_dim, "distances": distances, "gamma": gamma, "tau_factor": tau_factor,
            "bias_reg": bias_reg, "shrink": beta, "k_neighbors": k_best, "alpha_star": alpha_star, "hybrid": hybrid, "srmf": srmf, "cmf": cmf,
            "baselines": bl, "validation_loss": val_loss, "n_inner_train": int(len(itrain)), "n_inner_val": int(len(ival))}


# =============================================================================
# PHASE 3 - CROSS-VALIDATED EVALUATION
# =============================================================================

METHOD_NAMES = {"context_knn_unif": "Context-driven KNN (uniform)", "context_knn_rel": "Context-driven KNN (reliability-aware)",
                "context_knn_pred": "Context-driven KNN (prediction-aware)",
                "hybrid_knn_unif": "Hybrid KNN (uniform)", "hybrid_knn_rel": "Hybrid KNN (reliability-aware)",
                "hybrid_knn_pred": "Hybrid KNN (prediction-aware)",
                "cmf_unif": "Contextual MF on fuzzy clusters (uniform)", "cmf_rel": "Contextual MF on fuzzy clusters (reliability-aware)",
                "cmf_pred": "Contextual MF on fuzzy clusters (prediction-aware)",
                "cmf_relw": "Contextual MF on fuzzy clusters (reliability-aware, per-user dimension weights)",
                "srmf_unif": "SR-MF (uniform)", "srmf_rel": "SR-MF (reliability-aware)", "srmf_pred": "SR-MF (prediction-aware)",
                "biased_mf_cb": "MF + context-cluster biases", "srmf_unif_cb": "SR-MF + ctx biases (uniform)",
                "srmf_rel_cb": "SR-MF + ctx biases (reliability-aware)", "srmf_pred_cb": "SR-MF + ctx biases (prediction-aware)",
                "user_knn_pearson": "User-based KNN (Pearson)",
                "context_prefilter_knn": "Contextual pre-filtering KNN", "biased_mf": "Biased matrix factorisation",
                "baseline_bias": "Baseline predictor (mu + b_u + b_i)", "user_mean": "User mean", "item_mean": "Item mean",
                "global_mean": "Global mean"}
METHOD_ORDER = ["cmf_unif", "cmf_rel", "cmf_relw", "cmf_pred", "biased_mf_cb", "biased_mf",
                "context_knn_unif", "context_knn_rel", "context_knn_pred", "hybrid_knn_unif", "hybrid_knn_rel", "hybrid_knn_pred",
                "srmf_unif", "srmf_rel", "srmf_pred", "biased_mf_cb", "srmf_unif_cb", "srmf_rel_cb", "srmf_pred_cb",
                "user_knn_pearson", "context_prefilter_knn",
                "biased_mf", "baseline_bias", "user_mean", "item_mean", "global_mean"]


def paired_tests(pooled: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of methods a and b on per-user MAE over pooled test
    predictions (Wilcoxon signed-rank and paired t-test), plus the
    observation-level paired difference of absolute errors."""
    from scipy import stats
    per_user = pooled.groupby("userID").apply(lambda g: pd.Series({"a": np.mean(np.abs(g[a] - g["rating"])), "b": np.mean(np.abs(g[b] - g["rating"]))}))
    diff = per_user["a"] - per_user["b"]
    out = {"n_users": int(len(diff)), "mean_user_mae_diff": float(diff.mean()), "share_users_improved": float((diff < 0).mean())}
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(per_user["a"], per_user["b"]).pvalue)
        out["paired_t_p"] = float(stats.ttest_rel(per_user["a"], per_user["b"]).pvalue)
    except Exception:
        out["wilcoxon_p"] = np.nan; out["paired_t_p"] = np.nan
    ea, eb = np.abs(pooled[a] - pooled["rating"]), np.abs(pooled[b] - pooled["rating"])
    out["obs_level_paired_t_p"] = float(stats.ttest_rel(ea, eb).pvalue)
    return out


def evaluate_fold(args):
    """Complete evaluation of one (regime, repeat, fold): inner validation,
    re-fitting of the contextual models on the training part, predictions of
    every method, metrics on all subsets. Independent across folds."""
    df, selected, regime, rep, f, folds = args
    import time as _time
    context_methods = [f"context_knn_{s}" for s in WEIGHTING_SCHEMES]
    fold_rows, fold_details, pooled = [], [], []
    t0 = _time.time()
    train, test = df.loc[folds != f].copy(), df.loc[folds == f].copy()
    is_cold = np.zeros(len(test), dtype=bool)
    if regime == "cold_start":
        train, test, is_cold = apply_cold_start(train, test, FOLD_SEED + 500 * rep + f)
    print(f"\n[{regime}] REPEAT {rep + 1}/{N_REPEATS} FOLD {f + 1}/{N_FOLDS}: train={len(train)} test={len(test)}"
          + (f" (cold-user test ratings={int(is_cold.sum())})" if regime == "cold_start" else ""))
    sel = inner_validation(train, selected, FOLD_SEED + 1000 * rep + f, regime)
    print(f"  K_d={sel['k_clusters']} distances={sel['distances']} gamma={sel['gamma']} tau_factor={sel['tau_factor']} bias_reg={sel['bias_reg']} shrink={sel['shrink']}")
    print(f"  K neighbours={sel['k_neighbors']} alpha*={ {d: round(v, 2) for d, v in (sel['alpha_star'] or {}).items()} }")
    print(f"  hybrid: " + " | ".join(f"{s}: lambda={h['lambda']} K={h['k']} beta={h['shrink']}" + (f" alpha={ {d: round(v, 2) for d, v in h['alpha'].items()} }" if h['alpha'] else "") for s, h in sel["hybrid"].items()))

    fitted = fit_contextual_models(train, selected, sel["k_clusters"], FOLD_CLUSTERING_MODE, FOLD_SEED + 100 + f)
    if regime == "limited":
        fitted = limit_contextual_evidence(fitted, FOLD_SEED + 900 + f)
    comp = contextual_similarity_components(fitted, sel["tau_factor"])
    diag = weighting_diagnostics(comp, sel["distances"])
    heterogeneity = user_evidence_heterogeneity(comp)
    cutoff = heterogeneity.quantile(1.0 - HETEROGENEITY_SHARE)
    hetero_users = set(heterogeneity[heterogeneity >= cutoff].index)
    is_hetero = test["userID"].isin(hetero_users).to_numpy()
    idx = RatingIndex(train, sel["bias_reg"], PREFILTER_CONTEXT_VARS)

    preds = {}
    for scheme in WEIGHTING_SCHEMES:
        S = similarity_for_scheme(comp, sel["distances"], scheme, sel["alpha_star"] if scheme == "pred" else None, sel["gamma"])
        preds[f"context_knn_{scheme}"] = predict_with_similarity(test, idx, S, sel["k_neighbors"][scheme], sel["shrink"])
    S_p = pearson_user_similarity(idx)
    for scheme in (HYBRID_SCHEMES if ENABLE_HYBRID_KNN else []):
        h = sel["hybrid"][scheme]
        S_ctx = similarity_for_scheme(comp, sel["distances"], scheme, h["alpha"] if scheme == "pred" else None, sel["gamma"])
        preds[f"hybrid_knn_{scheme}"] = predict_with_similarity(test, idx, hybrid_similarity(S_ctx, S_p, h["lambda"]), h["k"], h["shrink"])
    for name, ctx in (("user_knn_pearson", False), ("context_prefilter_knn", True)):
        b = sel["baselines"][name]
        preds[name] = predict_with_similarity(test, RatingIndex(train, b["bias_reg"], PREFILTER_CONTEXT_VARS), S_p, b["k"], b["shrink"], True, ctx)
    preds["biased_mf"] = biased_mf_predict(train, test, sel["baselines"]["biased_mf"]["reg"], FOLD_SEED + f)
    users_all = pd.Index(sorted(train["userID"].unique()))
    rho_u = pd.Series(user_reliability_vector(comp, users_all), index=users_all)
    ctx_tr, _ = context_membership_blocks(fitted, train); ctx_te, _ = context_membership_blocks(fitted, test)
    layout_ctx = context_membership_blocks(fitted, train)[1]
    if ENABLE_CMF:
        rho_ud = user_reliability_by_dimension(comp, users_all)
        for scheme in CMF_SCHEMES:
            c = sel["cmf"][scheme]
            m = fit_cmf(train, ctx_tr, layout_ctx, c["alpha"], rho_ud if scheme in ("rel", "relw") else None, c["kappa"], c["reg"],
                        FOLD_SEED + f, per_user_weights=(scheme == "relw"))
            preds[f"cmf_{scheme}"] = predict_cmf(m, test, ctx_te)
        print("  cmf: " + " | ".join(f"{s}: kappa={c['kappa']}" + (f" alpha={ {d: round(v, 2) for d, v in c['alpha'].items()} }" if c['alpha'] else "") for s, c in sel["cmf"].items()))
    for ctx_bias in CONTEXT_BIAS_VARIANTS:
        ct, cte = (ctx_tr, ctx_te) if ctx_bias else (None, None)
        if ctx_bias:
            m = fit_srmf(train, None, sel["baselines"]["biased_mf"]["reg"], 0.0, FOLD_SEED + f, None, ct)
            preds["biased_mf_cb"] = predict_srmf(m, test, cte)
        if not ENABLE_SRMF:
            continue
        for scheme in WEIGHTING_SCHEMES:
            name = srmf_method_name(scheme, ctx_bias); cfg = sel["srmf"][name]
            S_q = similarity_for_scheme(comp, sel["distances"], scheme, cfg["alpha"] if scheme == "pred" else None, sel["gamma"])
            m = fit_srmf(train, S_q, cfg["reg"], cfg["beta"], FOLD_SEED + f, rho_u, ct)
            preds[name] = predict_srmf(m, test, cte)
    if ENABLE_SRMF:
        print("  srmf: " + " | ".join(f"{n}: beta={c['beta']}" + (f" alpha={ {d: round(v, 2) for d, v in c['alpha'].items()} }" if c['alpha'] else "") for n, c in sel["srmf"].items()))
    for b in ("global_mean", "item_mean", "user_mean", "baseline_bias"):
        preds[b] = predict_mean_baseline(test, idx, b)

    nb_all = np.all([(preds[m]["source"] == "neighbours").to_numpy() for m in context_methods + ["user_knn_pearson"] if m in preds], axis=0)
    for m, p in preds.items():
        overall = score_predictions(test, p)
        own_nb = score_predictions(test, p, (p["source"] == "neighbours").to_numpy())
        common = score_predictions(test, p, nb_all)
        hetero = score_predictions(test, p, is_hetero)
        cold = score_predictions(test, p, is_cold) if regime == "cold_start" else {"mae": np.nan, "rmse": np.nan, "n": 0}
        rank = ranking_metrics(test, p)
        fold_rows.append({"regime": regime, "repeat": rep + 1, "fold": f + 1, "method": m, **overall,
                          "mae_neighbour_subset": own_nb["mae"], "rmse_neighbour_subset": own_nb["rmse"],
                          "mae_common_subset": common["mae"], "rmse_common_subset": common["rmse"], "n_common_subset": common["n"],
                          "mae_hetero_subset": hetero["mae"], "rmse_hetero_subset": hetero["rmse"], "n_hetero_subset": hetero["n"],
                          "mae_cold_subset": cold["mae"], "rmse_cold_subset": cold["rmse"], "n_cold_subset": cold["n"],
                          **rank})
        extra = f" | cold RMSE={cold['rmse']:.4f}" if regime == "cold_start" else ""
        print(f"  {m:<24} MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} cov={overall['coverage']:.3f} "
              f"| hetero RMSE={hetero['rmse']:.4f} | NDCG@{RANKING_K}={rank['ndcg_at_k']:.3f}{extra}")
    pool = test[["userID", "itemID", TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "rating"}).copy()
    pool["repeat"] = rep + 1; pool["fold"] = f + 1; pool["is_hetero"] = is_hetero; pool["is_cold"] = is_cold
    for m, p in preds.items():
        pool[m] = p["prediction"].to_numpy()
    pooled.append(pool)
    fold_details.append({"regime": regime, "repeat": rep + 1, "fold": f + 1, "n_train": int(len(train)), "n_test": int(len(test)),
                         "runtime_seconds": round(_time.time() - t0, 1), "clustering_mode": FOLD_CLUSTERING_MODE,
                         "fitted_k": {d: v["k"] for d, v in fitted.items()}, "fitted_family": {d: v["family"] for d, v in fitted.items()},
                         "tau_d": {d: c["tau"] for d, c in comp.items()}, "weighting_diagnostics": diag,
                         "heterogeneity_cutoff": float(cutoff), "n_hetero_users": len(hetero_users),
                         **{k: v for k, v in sel.items()}})
    print(f"  weight diagnostics: std(w_rel)={ {d: round(v, 3) for d, v in diag['weights_rel_std_across_pairs'].items()} } "
          f"std(delta across dims)={diag['delta_std_across_dimensions_mean']:.3f} std(S_unif)={diag['similarity_unif_std']:.3f} corr(S_unif,S_rel)={diag['corr_similarity_unif_rel']:.3f}")

    return fold_rows, fold_details, pooled



def run_regime(df: pd.DataFrame, selected: dict, regime: str, out_dir: Path) -> dict:
    import time as _time
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    context_methods = [f"context_knn_{s}" for s in WEIGHTING_SCHEMES]
    fold_rows, fold_details, pooled = [], [], []

    tasks = []
    for rep in range(N_REPEATS):
        folds = assign_folds(df, N_FOLDS, FOLD_SEED + 1000 * rep)
        for f in range(N_FOLDS):
            tasks.append((df, selected, regime, rep, f, folds))
    with timed("phase3.regime", regime=regime, n_folds=len(tasks), n_users=int(df["userID"].nunique()), n_obs=int(len(df))):
        outputs = parallel_map(evaluate_fold, tasks)
    for rows_f, details_f, pooled_f in outputs:
        fold_rows.extend(rows_f); fold_details.extend(details_f); pooled.extend(pooled_f)

    per_fold = pd.DataFrame(fold_rows); per_fold.to_csv(out_dir / "results_per_fold.csv", index=False)
    cols = {"mae_mean": ("mae", "mean"), "mae_std": ("mae", "std"), "rmse_mean": ("rmse", "mean"), "rmse_std": ("rmse", "std"),
            "coverage_mean": ("coverage", "mean"), "rmse_common_subset": ("rmse_common_subset", "mean"),
            "rmse_hetero_subset": ("rmse_hetero_subset", "mean"), "mae_hetero_subset": ("mae_hetero_subset", "mean"),
            "rmse_cold_subset": ("rmse_cold_subset", "mean"), "mae_cold_subset": ("mae_cold_subset", "mean"),
            "precision_at_k": ("precision_at_k", "mean"), "recall_at_k": ("recall_at_k", "mean"), "ndcg_at_k": ("ndcg_at_k", "mean")}
    summary = per_fold.groupby("method").agg(**cols).reset_index()
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    summary = summary.sort_values("order").drop(columns="order").reset_index(drop=True)
    summary.insert(0, "regime", regime)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    (out_dir / "fold_details.json").write_text(json.dumps(fold_details, indent=2, default=float), encoding="utf-8")

    pooled_df = pd.concat(pooled, ignore_index=True); pooled_df.to_csv(out_dir / "pooled_test_predictions.csv", index=False)
    pairs = [("cmf_rel", "cmf_unif"), ("cmf_relw", "cmf_unif"), ("cmf_relw", "cmf_rel"), ("cmf_pred", "cmf_unif"), ("cmf_pred", "cmf_rel"), ("cmf_unif", "biased_mf_cb"), ("cmf_unif", "biased_mf"),
             ("cmf_pred", "biased_mf"), ("cmf_rel", "biased_mf"), ("cmf_pred", "user_knn_pearson"), ("cmf_pred", "context_prefilter_knn"),
             ("srmf_unif", "biased_mf"), ("srmf_rel", "srmf_unif"), ("srmf_pred", "srmf_unif"), ("srmf_pred", "srmf_rel"),
             ("biased_mf_cb", "biased_mf"), ("srmf_unif_cb", "biased_mf_cb"), ("srmf_rel_cb", "srmf_unif_cb"), ("srmf_pred_cb", "srmf_unif_cb"),
             ("srmf_pred_cb", "biased_mf"), ("srmf_rel_cb", "user_knn_pearson"),
             ("hybrid_knn_rel", "hybrid_knn_unif"), ("hybrid_knn_pred", "hybrid_knn_unif"), ("hybrid_knn_pred", "hybrid_knn_rel"),
             ("hybrid_knn_unif", "user_knn_pearson"), ("hybrid_knn_rel", "user_knn_pearson"), ("hybrid_knn_pred", "user_knn_pearson"),
             ("hybrid_knn_pred", "context_prefilter_knn"), ("hybrid_knn_pred", "biased_mf"), ("hybrid_knn_rel", "biased_mf"),
             ("context_knn_rel", "context_knn_unif"), ("context_knn_pred", "context_knn_unif"), ("context_knn_pred", "context_knn_rel"),
             ("context_knn_rel", "user_knn_pearson"), ("context_knn_pred", "user_knn_pearson"), ("context_knn_unif", "user_knn_pearson"),
             ("context_knn_pred", "context_prefilter_knn"), ("context_knn_pred", "biased_mf"), ("context_knn_rel", "biased_mf"),
             ("context_knn_pred", "baseline_bias"), ("context_knn_rel", "baseline_bias")]
    pairs = [(a, b) for a, b in pairs if a in pooled_df.columns and b in pooled_df.columns]
    tests = {"all": {f"{a} vs {b}": paired_tests(pooled_df, a, b) for a, b in pairs},
             "heterogeneous_evidence_users": {f"{a} vs {b}": paired_tests(pooled_df.loc[pooled_df["is_hetero"]], a, b) for a, b in pairs}}
    if regime == "cold_start":
        tests["cold_users"] = {f"{a} vs {b}": paired_tests(pooled_df.loc[pooled_df["is_cold"]], a, b) for a, b in pairs}
    (out_dir / "paired_tests.json").write_text(json.dumps(tests, indent=2, default=float), encoding="utf-8")

    alphas = pd.DataFrame([d["alpha_star"] for d in fold_details if d["alpha_star"]])
    if not alphas.empty:
        alphas.index = [f"rep{d['repeat']}_fold{d['fold']}" for d in fold_details if d["alpha_star"]]
        alphas.to_csv(out_dir / "alpha_star_per_fold.csv")
        if ENABLE_CMF:
            pd.DataFrame([{"fold": f"rep{d['repeat']}_fold{d['fold']}", "kappa_rel": d["cmf"]["rel"]["kappa"],
                           **{f"alpha_cmf_pred_{k}": v for k, v in (d["cmf"]["pred"]["alpha"] or {}).items()}} for d in fold_details]
                         ).to_csv(out_dir / "cmf_parameters_per_fold.csv", index=False)
        if ENABLE_HYBRID_KNN:
          pd.DataFrame([{"fold": f"rep{d['repeat']}_fold{d['fold']}", **{f"lambda_{s}": d["hybrid"][s]["lambda"] for s in HYBRID_SCHEMES},
                       **{f"alpha_hybrid_pred_{k}": v for k, v in (d["hybrid"]["pred"]["alpha"] or {}).items()}} for d in fold_details]
                     ).to_csv(out_dir / "hybrid_parameters_per_fold.csv", index=False)
        if ENABLE_SRMF:
          pd.DataFrame([{"fold": f"rep{d['repeat']}_fold{d['fold']}", **{f"beta_{n}": c["beta"] for n, c in d["srmf"].items()},
                       **{f"alpha_srmf_pred_{k}": v for k, v in (d["srmf"]["srmf_pred"]["alpha"] or {}).items()}} for d in fold_details]
                     ).to_csv(out_dir / "srmf_parameters_per_fold.csv", index=False)
        fig, ax = plt.subplots(figsize=(5, 3))
        alphas.plot(kind="bar", ax=ax, width=0.8); ax.set_ylabel("Prediction-aware weight"); ax.set_ylim(0, 1)
        ax.set_title(f"Prediction-aware coefficients per fold ({regime})"); ax.legend(frameon=False, fontsize=7)
        save_figure(fig, fig_dir / "prediction_aware_alpha")

    print("\n" + "=" * 110 + f"\nPHASE 3 - RESULTS, REGIME = {regime}\n" + "=" * 110)
    show = ["method", "mae_mean", "rmse_mean", "coverage_mean", "rmse_common_subset", "rmse_hetero_subset", "ndcg_at_k"] + (["rmse_cold_subset"] if regime == "cold_start" else [])
    print(summary[show].to_string(index=False))
    for scope, t in tests.items():
        print(f"\nPaired tests ({scope}):")
        for k, v in t.items():
            print(f"  {k:<48} diff={v['mean_user_mae_diff']:+.4f} improved={v['share_users_improved']:.2f} wilcoxon p={v['wilcoxon_p']:.3g}")
    return {"summary": summary, "fold_details": fold_details, "paired_tests": tests}


def run_phase3(script_dir: Path, clustering_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    clean_path = script_dir / CLEAN_FILENAME
    if not clean_path.exists():
        raise FileNotFoundError(f"{CLEAN_FILENAME} not found; run phase 1 first.")
    df = pd.read_csv(clean_path, sep=";")
    for c in EXPECTED_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.reset_index(drop=True)

    selected = {}
    for dim in CONTEXT_DIMENSIONS:
        p = clustering_dir / dim / "selected_configuration.json"
        if p.exists():
            selected[dim] = json.loads(p.read_text(encoding="utf-8"))
    if FOLD_CLUSTERING_MODE == "refit" and len(selected) < len(CONTEXT_DIMENSIONS):
        raise RuntimeError("Missing selected_configuration.json for some dimensions; run phase 1 first (the dimension set may have changed).")

    results = {}
    for regime in EVALUATION_REGIMES:
        results[regime] = run_regime(df, selected, regime, out_dir / regime)
    save_timings(out_dir / "timings_phase3.json")

    all_summary = pd.concat([r["summary"] for r in results.values()], ignore_index=True)
    all_summary.to_csv(out_dir / "results_summary_all_regimes.csv", index=False)

    # ---- paper table: one block per regime
    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{Rating-prediction and ranking accuracy under three evaluation regimes (%d$\times$%d-fold cross-validation). "
             r"Hetero.: test ratings of users with heterogeneous contextual evidence. Cold: test ratings of cold-start users.}" % (N_REPEATS, N_FOLDS),
             r"\label{tab:prediction-results}", r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Regime & Method & MAE & RMSE & Coverage & RMSE (hetero.) & NDCG@%d & RMSE (cold) \\" % RANKING_K, r"\midrule"]
    for regime, r in results.items():
        for _, row in r["summary"].iterrows():
            cold = f"{row['rmse_cold_subset']:.4f}" if regime == "cold_start" else "--"
            lines.append(f"{regime.replace('_', ' ')} & {METHOD_NAMES.get(row['method'], row['method'])} & {row['mae_mean']:.4f} & {row['rmse_mean']:.4f} $\\pm$ {row['rmse_std']:.4f} & "
                         f"{row['coverage_mean']:.3f} & {row['rmse_hetero_subset']:.4f} & {row['ndcg_at_k']:.3f} & {cold} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"; lines += [r"\end{tabular}", r"\end{table*}"]
    (out_dir / "paper_table_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- figure: weighting schemes vs baselines across regimes (RMSE) and NDCG
    methods = [m for m in [f"cmf_{s}" for s in CMF_SCHEMES] + ["biased_mf_cb", "biased_mf", "user_knn_pearson", "context_prefilter_knn", "context_knn_unif"]
               if all(m in r["summary"]["method"].values for r in results.values())]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, lab in zip(axes, ("rmse_mean", "ndcg_at_k"), ("RMSE", f"NDCG@{RANKING_K}")):
        xs = np.arange(len(results)); w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            vals = [r["summary"].set_index("method").loc[m, metric] for r in results.values()]
            ax.bar(xs + (j - len(methods) / 2 + 0.5) * w, vals, width=w, label=METHOD_NAMES.get(m, m),
                   color=None if m.startswith("cmf") else "#999999", alpha=1.0 if m.startswith("cmf") else 0.5 + 0.12 * (j % 4))
        ax.set_xticks(xs); ax.set_xticklabels([r.replace("_", " ") for r in results.keys()]); ax.set_ylabel(lab); ax.set_title(f"{lab} by evaluation regime")
        vals_all = [r["summary"].set_index("method").loc[m, metric] for r in results.values() for m in methods]
        ax.set_ylim(max(0, min(vals_all) - 0.05), max(vals_all) + 0.05)
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    save_figure(fig, fig_dir / "regimes_comparison")

    print("\n" + "=" * 110 + "\nPHASE 3 - ALL REGIMES\n" + "=" * 110)
    print(all_summary[["regime", "method", "mae_mean", "rmse_mean", "coverage_mean", "rmse_hetero_subset", "ndcg_at_k", "rmse_cold_subset"]].to_string(index=False))
    return results


# =============================================================================
# DATASET CONFIGURATION AND ORCHESTRATION
# =============================================================================







# =============================================================================
# DATASET-SPECIFIC CONFIGURATION
# =============================================================================

DATASET_NAME = "Frappe"
DEFAULT_INPUT = "frappe.csv"
PHASE1_SAMPLE_SIZE = 4000
RATING_SATURATION_QUANTILE = 0.99

CONTEXT_CODES = {
    "daytime": {"sunrise": 1, "morning": 2, "noon": 3, "afternoon": 4, "sunset": 5, "evening": 6, "night": 7},
    "weekday": {"monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4, "friday": 5, "saturday": 6, "sunday": 7},
    "homework": {"home": 1, "work": 2},                       # unknown -> missing
    "weather": {"sunny": 1, "cloudy": 2, "foggy": 3, "rainy": 4, "drizzle": 5, "stormy": 6, "snowy": 7, "sleet": 8},  # unknown -> missing
    "cost": {"free": 1, "paid": 2},
}
CONTEXT_COLUMNS = list(CONTEXT_CODES.keys())

CLEAN_FILENAME = f"{DATASET_NAME}_clean.csv"
EXPECTED_COLUMNS = ["userID", "itemID", "rating"] + CONTEXT_COLUMNS
ID_COLUMNS = ["userID", "itemID"]; TARGET_COLUMN = "rating"
USER_STATIC_COLUMNS = []; ALL_CONTEXT_COLUMNS = CONTEXT_COLUMNS; UNUSED_CONTEXT_COLUMNS = []
ITEM_METADATA_COLUMNS = []; NUMERIC_FEATURE_COLUMNS = []; CATEGORICAL_FEATURE_COLUMNS = CONTEXT_COLUMNS

CONTEXT_DIMENSIONS = {
    "temporal": {"features": ["daytime", "weekday"], "merge_rare_modalities": False,
                 "description": "Temporal context (time of day x day of week)."},
    "spatial_environmental": {"features": ["homework", "weather"], "merge_rare_modalities": True,
                              "description": "Spatial-environmental context (home/work x weather); unknown treated as missing."},
    "situational": {"features": ["cost"], "merge_rare_modalities": False,
                    "description": "Situational context (free / paid app)."},
}
K_CANDIDATES = {"temporal": [2, 3, 4, 5, 6], "spatial_environmental": [2, 3, 4, 5, 6], "situational": [2]}
FEATURE_TYPES = {c: "nominal" for c in CONTEXT_COLUMNS}
MODALITY_LABELS = {f: {v: k for k, v in codes.items()} for f, codes in CONTEXT_CODES.items()}
DIMENSION_DISPLAY_NAMES = {"temporal": "Temporal", "spatial_environmental": "Spatial-environmental", "situational": "Situational"}
DISTANCE_PER_DIMENSION = {"temporal": "hellinger", "spatial_environmental": "hellinger", "situational": "hellinger"}
PREFILTER_CONTEXT_VARS = ["daytime", "homework"]

# O(n) families only (the pairwise-distance families would need the n x n Gower matrix on ~77k training rows)
ALGORITHM_FAMILIES = ["fuzzy_k_modes", "latent_class_analysis", "modality_identity"]
COMPUTE_WASSERSTEIN = False
SELECT_DISTANCE_ON_VALIDATION = False
SELECT_K_CLUSTERS_ON_VALIDATION = False
ENABLE_HYBRID_KNN = False
ENABLE_SRMF = False
ENABLE_CMF = True


# =============================================================================
# READING AND CLEANING
# =============================================================================

def read_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path, sep="\t")
    raw.columns = [c.strip() for c in raw.columns]
    needed = ["user", "item", "cnt"] + CONTEXT_COLUMNS
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    report = {"rows_initial": int(len(raw))}
    df = raw[needed].copy()
    df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce")
    df = df.loc[df["cnt"].notna() & (df["cnt"] > 0)].copy()

    sat = float(np.log1p(df["cnt"].quantile(RATING_SATURATION_QUANTILE)))
    df["rating"] = 1.0 + 4.0 * np.minimum(1.0, np.log1p(df["cnt"]) / sat)
    report["pseudo_rating"] = {"formula": "1 + 4 * min(1, log(1+cnt)/log(1+cnt_p99))", "cnt_p99": float(np.expm1(sat)),
                               "rating_quantiles": {str(q): round(float(df["rating"].quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                               "share_rating_ge_4": float((df["rating"] >= 4).mean())}
    for col, codes in CONTEXT_CODES.items():
        v = df[col].astype("string").str.strip().str.lower()
        unknown = sorted(set(v.dropna().unique()) - set(codes) - {"unknown", "nan", ""})
        if unknown:
            raise ValueError(f"Unknown modality for {col}: {unknown}")
        df[col] = v.map(codes)

    df["userID"] = pd.to_numeric(df["user"]).astype(int)
    df["itemID"] = pd.to_numeric(df["item"]).astype(int) + 1
    df = df[["userID", "itemID", "rating"] + CONTEXT_COLUMNS].reset_index(drop=True)
    df.insert(0, "observationID", np.arange(len(df), dtype=int))
    for c in CONTEXT_COLUMNS:
        df[c] = pd.array(df[c], dtype="Int64")
    miss = df[CONTEXT_COLUMNS].isna()
    report.update({
        "rows_final": int(len(df)), "users": int(df["userID"].nunique()), "items": int(df["itemID"].nunique()),
        "missing_by_context": {c: int(v) for c, v in miss.sum().items()},
        "repeated_user_item_pairs": int(df.duplicated(["userID", "itemID"]).sum()),
        "records_per_user": {"min": int(df.groupby("userID").size().min()), "median": float(df.groupby("userID").size().median()),
                             "max": int(df.groupby("userID").size().max())},
        "records_per_item": {"median": float(df.groupby("itemID").size().median()), "max": int(df.groupby("itemID").size().max())},
        "unique_patterns": {d: int(df[c["features"]].dropna().drop_duplicates().shape[0]) for d, c in CONTEXT_DIMENSIONS.items()},
    })
    return df, report


# =============================================================================
# PHASE 1 AT SCALE: EXPLORATORY SEARCH ON A SAMPLE, REFIT ON ALL OBSERVATIONS
# =============================================================================

def run_phase1(script_dir: Path, input_path: Path) -> Path:
    t0 = time.time()
    out_dir = script_dir / "clustering_results"; out_dir.mkdir(exist_ok=True)
    df, report = read_and_clean(input_path)
    df.to_csv(script_dir / CLEAN_FILENAME, sep=";", index=False, na_rep="")
    (script_dir / f"{DATASET_NAME}_cleaning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 72 + f"\n{DATASET_NAME} - PHASE 1\n" + "=" * 72)
    print(f"Rows: {report['rows_initial']} -> {report['rows_final']} | users={report['users']} items={report['items']} | "
          f"missing={report['missing_by_context']} | patterns={report['unique_patterns']}")

    rng = np.random.default_rng(RANDOM_STATE)
    sample = df.iloc[np.sort(rng.choice(len(df), size=min(PHASE1_SAMPLE_SIZE, len(df)), replace=False))].reset_index(drop=True)
    datasets_s, reports_s = prepare_context_dimensions(sample)
    print(f"Exploratory search on a sample of {len(sample)} observations")
    run_all_clusterings(datasets_s, reports_s, out_dir)

    # refit the selected configuration on ALL observations of each dimension (O(n) families)
    datasets, reports = prepare_context_dimensions(df)
    (script_dir / f"{DATASET_NAME}_dimension_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    for dim, ddf in datasets.items():
        cfg_path = out_dir / dim / "selected_configuration.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        features = list(CONTEXT_DIMENSIONS[dim]["features"])
        entry = None
        for k_try in [cfg["selected_k"]] + [v for s in range(1, 6) for v in (cfg["selected_k"] + s, cfg["selected_k"] - s) if 2 <= v <= 8]:
            entry = fit_dimension_model(dim, ddf, cfg["selected_family"], k_try, cfg["selected_hyperparams"], RANDOM_STATE)
            if entry is not None:
                break
        if entry is None:
            raise RuntimeError(f"Could not refit {dim} on all observations")
        entry["memberships"].merge(ddf[["observationID", "userID", "itemID", "rating"] + features], on=["observationID", "userID", "itemID"]) \
            .assign(cluster=lambda x: entry["model"]["labels"] + 1).to_csv(out_dir / dim / "soft_memberships.csv", sep=";", index=False)
        if entry["cost"] is not None:
            names = [f"cluster_{i + 1}" for i in range(entry["cost"].shape[0])]
            pd.DataFrame(entry["cost"], index=names, columns=names).to_csv(out_dir / dim / "cluster_representative_distance_matrix.csv")
        cfg["selected_k"] = int(entry["k"]); cfg["refitted_on_all_observations"] = int(len(ddf)); cfg["exploratory_sample_size"] = int(len(sample))
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {dim:<24} refitted on {len(ddf)} observations: {cfg['selected_family']} K={entry['k']}")
    print(f"Phase 1 runtime: {time.time() - t0:.0f} s")
    return out_dir


def main() -> None:
    global EVALUATION_REGIMES, N_FOLDS, N_REPEATS
    parser = argparse.ArgumentParser(description=f"{DATASET_NAME} adapter for the context-driven CF pipeline.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--phase", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--regimes", default=",".join(EVALUATION_REGIMES))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = script_dir / input_path
    EVALUATION_REGIMES = [r.strip() for r in args.regimes.split(",") if r.strip()]
    N_FOLDS = args.folds; N_REPEATS = args.repeats
    clustering_dir, similarity_dir, evaluation_dir = script_dir / "clustering_results", script_dir / "similarity_results", script_dir / "evaluation_results"

    if args.phase in ("1", "all"):
        run_phase1(script_dir, input_path)
    if args.phase in ("2", "all"):
        t0 = time.time(); print("\n" + "=" * 72 + f"\n{DATASET_NAME} - PHASE 1 CLOSURE AND PHASE 2\n" + "=" * 72)
        run_phase2(clustering_dir, similarity_dir); print(f"Phase 2 runtime: {time.time() - t0:.0f} s")
    if args.phase in ("3", "all"):
        t0 = time.time(); print("\n" + "=" * 72 + f"\n{DATASET_NAME} - PHASE 3\n" + "=" * 72)
        run_phase3(script_dir, clustering_dir, evaluation_dir); print(f"Phase 3 runtime: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
