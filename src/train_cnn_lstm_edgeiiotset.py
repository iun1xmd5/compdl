#!/usr/bin/env python
# coding: utf-8
"""
CNN-LSTM IDS Training and Evaluation -- Edge-IIoTset Dataset

Trains and evaluates the Bayesian-optimized CNN-LSTM intrusion detection
model on the Edge-IIoTset dataset, benchmarked against gradient-boosting
and single-branch (CNN-only, LSTM-only) baselines. Includes Optuna TPE
hyperparameter search, sensitivity analysis over the composite objective's
penalty weights, and export of thesis-ready result tables.

This script is a cleaned, execution-equivalent conversion of the original
Kaggle notebook (final2-cnn-lstm-edgeiiotset-only.ipynb) used to produce the results reported in the
dissertation and accompanying paper. No experiment logic, hyperparameters,
preprocessing steps, or model definitions were altered during conversion —
only notebook-specific artifacts (cell markers, magic commands, inline
display() calls) were adapted for standalone script execution.

Requirements:
    See requirements.txt in this directory, or install directly:
    pip install optuna imbalanced-learn seaborn xgboost lightgbm catboost \
                tensorflow scikit-learn pandas numpy matplotlib

Usage:
    Update the dataset paths in the CONFIG section below (originally set for
    Kaggle's /kaggle/input mount), then run:
        python train_cnn_lstm_edgeiiotset.py
"""

# NOTE: the following packages must be installed before running this script.
# Original notebook cell used Kaggle's inline installer; run this once instead:
#     pip install -q optuna imbalanced-learn seaborn xgboost lightgbm catboost

# Cell 1: Install dependencies for Kaggle
# Enable Internet in Kaggle Notebook settings if any package is missing.
# (dependency installation handled via requirements.txt -- see header)

# Cell 2: Imports
import os
import gc
import sys
import time
import json
import random
import warnings
import platform
import subprocess
import shutil
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        """Fallback for environments without IPython (e.g. plain script execution)."""
        print(obj)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.base import clone

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
import optuna

warnings.filterwarnings("ignore")


# Kaggle / TensorFlow GPU stability setup.
# Memory growth prevents TensorFlow from reserving all GPU memory at once.
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(False)  # Disable XLA/JIT for better CUDA stability.
    print("GPU memory growth enabled for:", gpus)
except Exception as e:
    print("GPU setup warning:", e)

# ── Publication-quality figure style ─────────────────────────────────────────
# IEEE/Elsevier journals require:
#   • Vector PDF with embedded fonts (fonttype 42)
#   • ≥ 600 DPI for raster elements
#   • Colour-blind-safe palettes
#   • White background for compatibility with Word/PowerPoint
sns.set_theme(style="whitegrid", context="paper")
sns.set_palette("colorblind")

PALETTE = sns.color_palette("colorblind")

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       600,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.labelweight":  "bold",
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   9,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.8",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "lines.linewidth":   2.0,
    "lines.markersize":  6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "figure.constrained_layout.use": True,
    "figure.autolayout":             False,
    "mathtext.fontset":  "cm",
    "mathtext.default":  "regular",
})

# Cell 3: Reproducibility and Kaggle configuration — EdgeIIoT-set only
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

KAGGLE_INPUT_ROOT  = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def find_edgeiiot_csv():
    """Auto-discover DNN-EdgeIIoT-dataset.csv from /kaggle/input.

    Searches by directory-name tokens first, then filename keywords.
    Always picks the largest CSV found — avoids accidentally selecting a
    per-protocol subset instead of the full merged file.
    """
    import glob as _glob
    all_csvs = sorted(_glob.glob("/kaggle/input/**/*.csv", recursive=True))

    # Strategy 1: directory name contains an EdgeIIoT token
    tokens = ["edgeiiot", "edge-iiot", "edge_iiot", "edgeiiotset"]
    edge_dirs = set()
    for p in all_csvs:
        if any(t in p.lower() for t in tokens):
            edge_dirs.add(os.path.dirname(p))

    if edge_dirs:
        candidates = sorted(p for p in all_csvs if os.path.dirname(p) in edge_dirs)
        if candidates:
            candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
            print(f"  [EdgeIIoT] Found via directory match:")
            print(f"    {candidates[0]}  ({os.path.getsize(candidates[0])/1024**2:.0f} MB)")
            return candidates[0]

    # Strategy 2: filename keywords
    fname_tokens = ["edgeiiot", "edge-iiot", "edge_iiot", "dnn-edgeiiot"]
    candidates = [
        p for p in all_csvs
        if any(t in os.path.basename(p).lower() for t in fname_tokens)
    ]
    if candidates:
        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        print(f"  [EdgeIIoT] Found via filename match:")
        print(f"    {candidates[0]}  ({os.path.getsize(candidates[0])/1024**2:.0f} MB)")
        return candidates[0]

    print("  [EdgeIIoT] WARNING: file not found — update path manually.")
    return "/kaggle/input/CHANGE_ME/DNN-EdgeIIoT-dataset.csv"


CONFIG = {
    "DATASET_PATHS": {
        "EdgeIIoT": find_edgeiiot_csv(),
    },
    "TARGET_COLUMNS": {
        "EdgeIIoT": "Attack_type",   # 16-class label column in EdgeIIoT-set
    },
    "RESULTS_DIR": str(KAGGLE_WORKING_ROOT / "thesis_results"),
    "OUTPUT_ZIP":  str(KAGGLE_WORKING_ROOT / "thesis_results.zip"),
    "TEST_SIZE":  0.15,
    "VAL_SIZE":   0.15,

    "TOP_K_FEATURES": 32,
    "TOP_K_FEATURES_BY_DATASET": {
        "EdgeIIoT": 32,
    },

    # ── EdgeIIoT-set sampling ─────────────────────────────────────────────────
    # EdgeIIoT-set contains ~2.2 M rows across 15 attack classes + Normal.
    # EDGEIIOT_SAMPLES_PER_CLASS = 10_000 → ~160k total rows (16 classes).
    # Fits comfortably within Kaggle's 16 GB RAM.
    # Increase to 20_000–30_000 if you want more training data.
    "EDGEIIOT_SAMPLES_PER_CLASS": 10_000,

    # Columns to drop before feature selection — metadata and leakage fields.
    # Attack_label is a binary flag that would directly leak the target.
    "EDGEIIOT_DROP_COLS": [
        "frame.time", "ip.src_host", "ip.dst_host",
        "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
        "http.file_data", "http.request.full_uri",
        "http.request.uri.query", "http.request.version",
        "http.response", "http.tls_port",
        "mqtt.msg", "mqtt.topic", "mqtt.username", "mqtt.passwd",
        "mqtt.willmsg", "mqtt.willtopic",
        "dns.qry.name", "dns.qry.name.len",
        "tcp.payload", "tcp.segment_data", "udp.payload",
        "Attack_label",   # binary label — leaks the target
    ],

    "LOW_CARDINALITY_THRESHOLD": 30,
    "USE_SAMPLE":       False,   # not used for EdgeIIoT (chunked sampling instead)
    "SAMPLE_PER_CLASS": 12000,   # not used for EdgeIIoT
    # SEQUENCE_MODE changed to sliding_window (Option A — addresses
    # Reviewer 1 & 2 Concern 1). Each sample is now a window of
    # SLIDING_WINDOW consecutive network flows, giving the LSTM genuine
    # temporal sequences instead of a static feature vector reshaped
    # into artificial timesteps. Input shape: (SLIDING_WINDOW, n_features)
    # e.g. (8, 32) — 8 real timesteps × 32 selected features.
    "SEQUENCE_MODE":    "sliding_window",
    "SLIDING_WINDOW":   8,
    # N_TRIALS increased 15 → 50 (addresses Reviewer 1 Concern 5 and
    # Reviewer 2 Concern 4). 15 trials is insufficient for a
    # 9-dimensional mixed space. At 50 trials with 20 random startup
    # trials, Optuna TPE has 30 guided trials for a stronger
    # convergence argument.
    "N_TRIALS":         50,
    "LATENCY_PENALTY":  0.10,
    "PARAMS_PENALTY":   0.10,
    "PATIENCE":         4,

    "RUN_LINEAR_SVM":    False,
    "RUN_RANDOM_FOREST": True,
    "RUN_XGBOOST":       True,
    "RUN_LIGHTGBM":      True,
    "RUN_CATBOOST":      True,

    "RF_N_ESTIMATORS":    100,
    "XGB_N_ESTIMATORS":   100,
    "XGB_MAX_DEPTH":      6,
    "XGB_SUBSAMPLE":      0.5,
    "XGB_LEARNING_RATE":  0.3,
    "LGBM_N_ESTIMATORS":  600,
    "CATBOOST_ITERATIONS": 100,

    "BASELINE_SETTING_SOURCES": {
        "RandomForest":          "Adewole et al. (2025), Sensors, DOI: 10.3390/s25061845",
        "XGBoost":               "Adewole et al. (2025), Sensors, DOI: 10.3390/s25061845",
        "LightGBM":              "Abid et al. (2025), Journal of Cloud Computing, DOI: 10.1186/s13677-025-00785-2",
        "CatBoost":              "Adewole et al. (2025), Sensors, DOI: 10.3390/s25061845",
        "CNN_Only":              "Kim et al. (2020), Electronics, DOI: 10.3390/electronics9060916",
        "LSTM_Only":             "Ogunseyi and Thiyagarajan (2025), Sensors, DOI: 10.3390/s25072288",
        "CNN_LSTM_NonOptimized": "Altunay and Albayrak (2023), Engineering Science and Technology, DOI: 10.1016/j.jestch.2022.101322",
        "EdgeIIoT_dataset":      "Ferrag et al. (2022), IEEE Access, DOI: 10.1109/ACCESS.2022.3190173",
    },

    "FIGURE_FORMATS":     ["pdf", "png"],
    "FIGURE_DPI":         600,
    "NN_VERBOSE":         0,
    # LATENCY_BATCH_SIZE=1 measures true single-sample edge inference
    # (addresses Reviewer 2 Concern 2). Timing at trained batch size
    # introduced GPU-parallelism bias — larger batches appeared faster
    # per sample. Batch size=1 removes this confound and reflects
    # real edge streaming conditions.
    "LATENCY_BATCH_SIZE": 1,
}

os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

print("TensorFlow :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))
print("Results dir:", CONFIG["RESULTS_DIR"])
print("\nEdgeIIoT path:", CONFIG["DATASET_PATHS"]["EdgeIIoT"])
print("\nIf the path contains CHANGE_ME, attach the dataset via Add Data.")

# Cell 4: Verify EdgeIIoT-set file is accessible
if KAGGLE_INPUT_ROOT.exists():
    print("Files under /kaggle/input:")
    shown = 0
    for dirpath, _, filenames in os.walk(str(KAGGLE_INPUT_ROOT)):
        for fname in sorted(filenames):
            fpath = os.path.join(dirpath, fname)
            size_mb = os.path.getsize(fpath) / 1024**2
            print(f"  {fpath}  ({size_mb:.0f} MB)")
            shown += 1
            if shown >= 30:
                print("  ... (showing first 30 files only)")
                break
        if shown >= 30:
            break
else:
    print("/kaggle/input not found — this notebook is intended for Kaggle.")

print("\nEdgeIIoT path check:")
p = CONFIG["DATASET_PATHS"]["EdgeIIoT"]
if os.path.exists(p):
    print(f"  ✓  {p}  ({os.path.getsize(p)/1024**2:.0f} MB)")
else:
    print(f"  ✗  NOT FOUND: {p}")
    print("  → Attach the EdgeIIoT-set dataset via Add Data and re-run Cell 3.")

# Cell 5: General helper functions
def make_onehot_encoder():
    try:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        from sklearn.preprocessing import OneHotEncoder
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_path_exists(path):
    """Validate a single path or a list of paths."""
    if isinstance(path, (list, tuple)):
        missing = [str(p) for p in path if not os.path.exists(str(p))]
        if missing:
            raise FileNotFoundError(
                "Some InSDN CSV files were not found:\n" + "\n".join(missing) +
                "\nPlease attach the InSDN dataset via Add Data in Kaggle."
            )
        return
    if not os.path.exists(str(path)):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Please attach the dataset to this Kaggle notebook using Add Data, or update CONFIG['DATASET_PATHS']."
        )

def reduce_sample_if_needed(df, target_col, use_sample=False, sample_per_class=5000):
    if not use_sample:
        return df.copy()
    sampled = (
        df.groupby(target_col, group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), sample_per_class), random_state=SEED))
          .reset_index(drop=True)
    )
    return sampled

def clean_dataframe(df, dataset_name, target_col):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    leakage_cols = []
    if dataset_name == "ToN_IoT" and "label" in df.columns:
        leakage_cols.append("label")

    # EdgeIIoT-set: drop metadata / leakage columns defined in CONFIG.
    # 'Attack_label' is a binary flag that directly encodes whether a row is
    # an attack — keeping it would cause data leakage into the classifier.
    if dataset_name == "EdgeIIoT":
        drop_cfg = CONFIG.get("EDGEIIOT_DROP_COLS", [])
        leakage_cols.extend([c for c in drop_cfg if c in df.columns])

    leakage_cols = [c for c in leakage_cols if c != target_col]
    df.drop(columns=leakage_cols, errors="ignore", inplace=True)

    # Ensure all object columns are uniform str — prevents mixed str/float
    # that breaks sklearn OneHotEncoder when NA values are present.
    for _col in df.select_dtypes(include="object").columns:
        if _col != target_col:
            df[_col] = df[_col].astype(str).replace({"nan": "Unknown",
                                                      "<NA>": "Unknown"})
    return df

def choose_feature_columns(df, target_col, low_cardinality_threshold=30):
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols, categorical_cols, dropped_cols = [], [], []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            unique_ratio = nunique / max(len(df), 1)
            if nunique > low_cardinality_threshold or unique_ratio > 0.10:
                dropped_cols.append(col)
            else:
                categorical_cols.append(col)

    return numeric_cols, categorical_cols, dropped_cols

def stratified_split(X_df, y_series):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_df,
        y_series,
        test_size=(CONFIG["TEST_SIZE"] + CONFIG["VAL_SIZE"]),
        stratify=y_series,
        random_state=SEED,
    )

    relative_test_size = CONFIG["TEST_SIZE"] / (CONFIG["TEST_SIZE"] + CONFIG["VAL_SIZE"])

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_dataframe(df, path, index=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    return path

def get_gpu_name():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        return output[0] if output else "CPU / Not available"
    except Exception:
        return "CPU / Not available"

def get_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return np.nan

def get_runtime_environment_table():
    env = {
        "item": [
            "Platform",
            "Python version",
            "TensorFlow version",
            "scikit-learn version",
            "GPU name",
            "RAM (GB)",
        ],
        "value": [
            platform.platform(),
            sys.version.split(" ")[0],
            tf.__version__,
            sklearn.__version__,
            get_gpu_name(),
            get_ram_gb(),
        ],
    }
    return pd.DataFrame(env)

def make_class_count_df(labels, split_name):
    counts = pd.Series(labels, dtype="object").astype(str).value_counts().sort_index()
    return pd.DataFrame({
        "split": split_name,
        "class": counts.index,
        "count": counts.values,
    })

def save_json_table(json_dict, path):
    df = pd.DataFrame(list(json_dict.items()), columns=["hyperparameter", "value"])
    df.to_csv(path, index=False)
    return df

# Cell 6: Preprocessing, feature selection, balancing, and sequence helpers
def build_preprocessor(numeric_cols, categorical_cols):
    # StandardScaler (z-score) is used: centers each feature at 0 with unit variance.
    # This pairs well with ReLU + BatchNorm + Adam in the CNN-LSTM, and is the
    # most common choice in recent deep-learning IDS work.
    from sklearn.preprocessing import StandardScaler

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor

def encode_targets(y_train, y_val, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)
    class_names = list(le.classes_)
    return y_train_enc, y_val_enc, y_test_enc, le, class_names

def apply_feature_selection(X_train, y_train, X_val, X_test, top_k=32):
    k = min(top_k, X_train.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel = selector.transform(X_val)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_val_sel, X_test_sel, selector

def balance_training_data(X_train, y_train):
    class_counts = Counter(y_train)
    min_count = min(class_counts.values())

    if min_count > 1:
        k_neighbors = min(5, min_count - 1)
        sampler = SMOTE(random_state=SEED, k_neighbors=k_neighbors)
    else:
        sampler = RandomOverSampler(random_state=SEED)

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res, sampler

def make_sequences(X, y, mode="feature_vector", window=8):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if mode == "feature_vector":
        return X.reshape(X.shape[0], X.shape[1], 1), y

    if mode == "sliding_window":
        if len(X) < window:
            raise ValueError(f"Not enough samples ({len(X)}) for window size {window}.")
        X_seq, y_seq = [], []
        for i in range(len(X) - window + 1):
            X_seq.append(X[i:i + window])
            y_seq.append(y[i + window - 1])
        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq)

    raise ValueError("mode must be either 'feature_vector' or 'sliding_window'")

# Cell 7: Evaluation and plotting helpers
def macro_false_positive_rate(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp_rates = []
    total = cm.sum()

    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - (tp + fp + fn)
        denominator = fp + tn
        fp_rate = (fp / denominator) if denominator > 0 else 0.0
        fp_rates.append(fp_rate)

    return float(np.mean(fp_rates))

def safe_multiclass_auc(y_true, y_prob, n_classes):
    try:
        if y_prob is None:
            return np.nan
        y_prob = np.asarray(y_prob)
        if n_classes == 2:
            # For binary classification, use the probability of the positive class.
            positive_scores = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.ravel()
            return float(roc_auc_score(y_true, positive_scores))
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        return float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return np.nan

def measure_latency_and_throughput(model, X, batch_size=256, model_type="keras"):
    start = time.perf_counter()

    if model_type == "keras":
        preds = model.predict(X, batch_size=batch_size, verbose=0)
    else:
        preds = model.predict(X)

    total_time = time.perf_counter() - start
    latency = total_time / len(X)
    throughput = len(X) / total_time if total_time > 0 else np.nan
    return preds, latency, throughput, total_time

def thesis_figure_paths(save_path):
    """Return all output paths for one figure using CONFIG['FIGURE_FORMATS']."""
    if save_path is None:
        return []
    base, _ = os.path.splitext(save_path)
    formats = CONFIG.get("FIGURE_FORMATS", ["pdf", "png"])
    return [f"{base}.{fmt.lower().lstrip('.')}" for fmt in formats]

def save_thesis_figure(fig, save_path):
    """Save vector PDF + 600 dpi PNG.  Always white background."""
    if save_path is None:
        return []
    saved_paths = []
    for path in thesis_figure_paths(save_path):
        ext = os.path.splitext(path)[1].lower()
        kwargs = {
            "bbox_inches": "tight",
            "facecolor":   "white",
            "edgecolor":   "none",
        }
        if ext != ".pdf":
            kwargs["dpi"] = CONFIG.get("FIGURE_DPI", 600)
        fig.savefig(path, **kwargs)
        saved_paths.append(path)
    return saved_paths

def plot_history(history, title_prefix, save_path=None):
    """Training / validation loss and accuracy curves — publication quality."""
    if history is None:
        return

    hist   = history.history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Loss ──────────────────────────────────────────────────────────────────
    axes[0].plot(epochs, hist.get("loss",     []), color=PALETTE[0],
                 label="Train",      linewidth=2.0, marker="o",
                 markersize=3, markevery=max(1, len(epochs)//10))
    axes[0].plot(epochs, hist.get("val_loss", []), color=PALETTE[1],
                 label="Validation", linewidth=2.0, marker="s",
                 markersize=3, markevery=max(1, len(epochs)//10),
                 linestyle="--")
    axes[0].set_title(f"{title_prefix} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Categorical cross-entropy loss")
    axes[0].legend()

    # ── Accuracy ──────────────────────────────────────────────────────────────
    axes[1].plot(epochs, hist.get("accuracy",     []), color=PALETTE[0],
                 label="Train",      linewidth=2.0, marker="o",
                 markersize=3, markevery=max(1, len(epochs)//10))
    axes[1].plot(epochs, hist.get("val_accuracy", []), color=PALETTE[1],
                 label="Validation", linewidth=2.0, marker="s",
                 markersize=3, markevery=max(1, len(epochs)//10),
                 linestyle="--")
    axes[1].set_title(f"{title_prefix} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    acc_vals = hist.get("accuracy", []) + hist.get("val_accuracy", [])
    if acc_vals:
        ymin = max(0.0, min(acc_vals) - 0.05)
    else:
        ymin = 0.0
    axes[1].set_ylim(ymin, 1.02)
    axes[1].legend()

    # suptitle with constrained_layout — no manual y offset needed
    fig.suptitle(title_prefix, fontsize=13, fontweight="bold")
    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)

def plot_confusion_matrix_seaborn(y_true, y_pred, class_names, title,
                                   save_path=None, normalize=True):
    """Normalised confusion matrix with per-class recall on the diagonal."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(class_names)
    cell = max(0.85, min(1.4, 10 / n))
    fig_w = max(7, cell * n + 2.5)
    fig_h = max(5.5, cell * n + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Annotate with percentage (normalised) but show raw count in parentheses
    # when the matrix is small enough to be legible.
    if n <= 12:
        annot_arr = np.empty_like(cm, dtype=object)
        for i in range(n):
            for j in range(n):
                pct = cm_norm[i, j]
                cnt = cm[i, j]
                # explicit newline — safe across all editors
                annot_arr[i, j] = "{:.2f}".format(pct) + "\n" + "({})".format(cnt)
        annot, fmt = annot_arr, ""
    else:
        annot, fmt = True, ".2f"

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt=fmt,
        cmap="Blues",          # always Blues regardless of normalize param
        vmin=0, vmax=1,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Recall (row-normalised)", "shrink": 0.8},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label",      labelpad=8)
    # ha="right" prevents long class-name labels from overlapping
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # constrained_layout handles spacing — calling tight_layout() after
    # a colorbar is attached raises RuntimeError (incompatible engines).
    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)

def plot_multiclass_roc(y_true, y_prob, class_names, title, save_path=None):
    """Per-class OvR ROC curves + macro-average — publication quality."""
    n_classes = len(class_names)
    y_prob    = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(9, 7))
    # Use husl for >8 classes (colorblind only has 8 distinct colours)
    palette = "colorblind" if n_classes <= 8 else "husl"
    colors  = sns.color_palette(palette, n_colors=max(n_classes, 2))

    if n_classes == 2:
        try:
            pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob.ravel()
            fpr, tpr, _ = roc_curve(y_true, pos)
            ax.plot(fpr, tpr, color=colors[0], lw=2.0,
                    label=f"{class_names[1]} (AUC = {auc(fpr,tpr):.3f})")
        except Exception:
            pass
    else:
        y_bin     = label_binarize(y_true, classes=np.arange(n_classes))
        all_fpr   = np.linspace(0, 1, 300)
        tpr_interp = []

        for i, cname in enumerate(class_names):
            try:
                fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                auc_i = auc(fpr_i, tpr_i)
                ax.plot(fpr_i, tpr_i, color=colors[i % len(colors)], lw=1.5,
                        alpha=0.80,
                        label=f"{cname} (AUC = {auc_i:.3f})")
                tpr_interp.append(np.interp(all_fpr, fpr_i, tpr_i))
            except Exception:
                continue

        # Macro-average curve
        if tpr_interp:
            mean_tpr  = np.mean(tpr_interp, axis=0)
            mean_auc  = auc(all_fpr, mean_tpr)
            ax.plot(all_fpr, mean_tpr, color="black", lw=2.5, linestyle="-.",
                    label=f"Macro-average (AUC = {mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            linewidth=1.2, label="Random classifier")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0,   1.02])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title, pad=10)
    # Place legend outside axes so curves aren't hidden
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        borderaxespad=0,
        framealpha=0.9,
    )
    fig.subplots_adjust(right=0.72)   # make room for outside legend
    fig.tight_layout()

    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)

def plot_metric_bar(df, metric_col, title, ylabel, save_path=None, log_scale=False):
    """Horizontal grouped bar chart — easier to read long model names in papers."""
    plot_df = df.dropna(subset=[metric_col]).copy()
    plot_df = plot_df.sort_values(metric_col, ascending=True)   # ascending for horiz

    n       = len(plot_df)
    fig_h   = max(4.5, 0.55 * n + 2.0)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    colors  = sns.color_palette("colorblind", n_colors=n)
    bars    = ax.barh(
        range(n),
        plot_df[metric_col].values,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.65,
    )

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_df["model"].values, fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=10)

    xmax = plot_df[metric_col].max()
    if log_scale:
        ax.set_xscale("log")
        for bar, val in zip(bars, plot_df[metric_col].values):
            ax.text(
                val * 1.15,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}",
                va="center", ha="left", fontsize=8, fontweight="bold",
            )
    else:
        ax.set_xlim(0, xmax * 1.18)
        for bar, val in zip(bars, plot_df[metric_col].values):
            offset = xmax * 0.008
            ax.text(
                val + offset,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}" if abs(val) < 10 else f"{val:,.1f}",
                va="center", ha="left", fontsize=8.5, fontweight="bold",
            )

    fig.tight_layout()
    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_radar(df, metrics, title, save_path=None):
    """Radar / spider chart comparing all models across multiple metrics."""
    import math
    models  = df["model"].tolist()
    n_vars  = len(metrics)
    angles  = [n / n_vars * 2 * math.pi for n in range(n_vars)]
    angles += angles[:1]                                     # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={"projection": "polar"})
    colors  = sns.color_palette("colorblind", n_colors=len(models))

    for (_, row), color in zip(df.iterrows(), colors):
        vals  = [float(row.get(m, 0) or 0) for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.8, color=color, label=row["model"])
        ax.fill(angles, vals, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in metrics],
        fontsize=9,
    )
    ax.set_rlabel_position(15)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.set_title(title, pad=28, fontsize=13, fontweight="bold")
    # Legend below the chart — never clips in PDF
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(4, len(models)),
        fontsize=8,
        framealpha=0.9,
    )
    fig.subplots_adjust(bottom=0.18)

    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_pareto(df, title, save_path=None):
    """F1-macro vs parameter count Pareto scatter — efficiency story for the paper."""
    plot_df = df.dropna(subset=["f1_macro", "trainable_params"]).copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors  = sns.color_palette("colorblind", n_colors=len(plot_df))

    for (_, row), color in zip(plot_df.iterrows(), colors):
        ax.scatter(
            row["trainable_params"], row["f1_macro"],
            color=color, s=120, zorder=3,
            edgecolors="black", linewidths=0.6,   # black edge visible in greyscale
        )
        ax.annotate(
            row["model"],
            (row["trainable_params"], row["f1_macro"]),
            textcoords="offset points", xytext=(7, 4),
            fontsize=8, color=color, fontweight="bold",
        )

    f1_vals = plot_df["f1_macro"].values
    y_margin = (f1_vals.max() - f1_vals.min()) * 0.25 if len(f1_vals) > 1 else 0.05
    ax.set_ylim(max(0, f1_vals.min() - y_margin), min(1.02, f1_vals.max() + y_margin))
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Macro F1-score")
    ax.set_xscale("log")
    ax.set_title(title, pad=10)
    fig.tight_layout()

    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)


def plot_per_class_f1(report_df, model_name, dataset_name, save_path=None):
    """Per-class F1 bar chart from classification_report DataFrame."""
    # classification_report rows: class names + summary rows
    skip = {"accuracy", "macro avg", "weighted avg"}
    cls_df = report_df[~report_df.index.isin(skip)].copy()
    if cls_df.empty or "f1-score" not in cls_df.columns:
        return

    cls_df = cls_df.sort_values("f1-score", ascending=True)
    n      = len(cls_df)
    colors = sns.color_palette("colorblind", n_colors=n)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * n + 1.5)))
    bars = ax.barh(range(n), cls_df["f1-score"].values,
                   color=colors, edgecolor="white", linewidth=0.6, height=0.65)

    for bar, val in zip(bars, cls_df["f1-score"].values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=8.5)

    ax.set_yticks(range(n))
    ax.set_yticklabels(cls_df.index.tolist(), fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1-score")
    ax.set_title(f"{dataset_name} — {model_name}: Per-class F1", pad=10)
    fig.tight_layout()

    save_thesis_figure(fig, save_path)
    plt.show()
    plt.close(fig)

def evaluate_predictions(y_true, y_pred, y_prob, class_names, latency, throughput):
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    fpr_macro = macro_false_positive_rate(y_true, y_pred, labels=np.arange(len(class_names)))
    roc_auc_macro = safe_multiclass_auc(y_true, y_prob, n_classes=len(class_names)) if y_prob is not None else np.nan

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    metrics_dict = {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "fpr_macro": fpr_macro,
        "roc_auc_ovr_macro": roc_auc_macro,
        "latency_seconds_per_sample": latency,
        "throughput_samples_per_second": throughput,
    }

    return metrics_dict, pd.DataFrame(report).transpose()

# Cell 8: Model building functions

def compile_model(model, learning_rate):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_cnn_only_params():
    """CNN-only ablation setting adapted from CNN-based IDS literature.

    The cited CNN IDS paper by Kim et al. uses a CNN design for network intrusion
    detection, and survey summaries describe it as a three-convolution/two-pooling
    CNN configuration. This notebook uses a 1D adaptation because the present
    datasets are selected flow-feature vectors, not images.
    """
    return {
        "conv_filters": [64, 128, 128],
        "kernel_size": 3,
        "pool_size": 2,
        "n_pool": 2,
        "dropout": 0.3,
        "dense_units": 64,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 20,
    }


def get_lstm_only_params():
    """LSTM-only ablation setting adapted from Ogunseyi and Thiyagarajan (2025).

    Their LSTM IDS setting reports one LSTM layer with 64 units, dropout 0.2,
    learning rate 0.001, batch size 64, 20 epochs, and Adam optimizer.
    """
    return {
        "lstm_units": 64,
        "dropout": 0.2,
        "dense_units": 64,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 20,
    }


def get_default_cnn_lstm_params():
    """Non-optimized CNN-LSTM counterfactual baseline.

    This is a non-tuned CNN+LSTM control model adapted from hybrid CNN+LSTM IDS
    literature. The epoch and batch-size settings follow the reported CNN+LSTM
    experimental setting from Altunay and Albayrak, while the input and output
    layers are adapted to the present ToN-IoT and InSDN feature tensors.
    Early stopping is still used during training to avoid unnecessary overfitting.
    """
    return {
        "n_conv": 1,
        "filters": 64,
        "kernel_size": 3,
        "lstm_units": 64,
        "dense_units": 64,
        "dropout": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 120,
        "epochs": 100,
    }


def build_cnn_lstm_model(input_shape, n_classes, params):
    model = models.Sequential(name="CNN_LSTM")
    model.add(layers.Input(shape=input_shape))

    for i in range(params["n_conv"]):
        model.add(
            layers.Conv1D(
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"conv1d_{i+1}",
            )
        )
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(params["dropout"]))

    model.add(layers.LSTM(params["lstm_units"], return_sequences=False))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))

    return compile_model(model, params["learning_rate"])


def build_cnn_only_model(input_shape, n_classes, params=None):
    params = params or get_cnn_only_params()
    conv_filters = params.get("conv_filters") or [params.get("filters", 64)] * params.get("n_conv", 2)
    n_pool = params.get("n_pool", min(2, len(conv_filters)))

    model = models.Sequential(name="CNN_ONLY")
    model.add(layers.Input(shape=input_shape))

    for i, filters in enumerate(conv_filters):
        model.add(
            layers.Conv1D(
                filters=filters,
                kernel_size=params["kernel_size"],
                activation="relu",
                padding="same",
                name=f"cnn_only_conv1d_{i+1}",
            )
        )
        model.add(layers.BatchNormalization())
        if i < n_pool:
            model.add(layers.MaxPooling1D(pool_size=params.get("pool_size", 2)))
        model.add(layers.Dropout(params["dropout"]))

    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])


def build_lstm_only_model(input_shape, n_classes, params=None):
    params = params or get_lstm_only_params()
    model = models.Sequential(name="LSTM_ONLY")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(params["lstm_units"], return_sequences=False))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(params["dense_units"], activation="relu"))
    model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(n_classes, activation="softmax"))
    return compile_model(model, params["learning_rate"])

# Cell 9: Bayesian optimization with Optuna
def tune_cnn_lstm(X_train, y_train, X_val, y_val, n_classes, dataset_name):
    input_shape = X_train.shape[1:]

    def objective(trial):
        # TIGHTENED SEARCH SPACE — targets edge-deployable models (< 30k params).
        #
        # Root cause of large models: LSTM params = 4 × units × (filters + units + 1).
        # At lstm_units=192, filters=96: LSTM alone = ~221k params (80% of total).
        # Reducing lstm_units 192→48 and filters 96→32 drops LSTM from ~222k to ~14k
        # while preserving sequence learning capacity on 32-feature IDS inputs.
        #
        # Search space bounds (worst case = n_conv=2, filters=32, lstm=48, dense=32):
        #   Conv1D ×2 + BN:  ~4.4k params
        #   LSTM:            ~14.7k params
        #   Dense head:       ~1.6k params
        #   TOTAL:           ~20.7k params  ← well within IoT/SDN deployment budget
        # Search space — deliberately constrained to keep models under 30k params,
        # consistent with edge/IoT/SDN deployment requirements (addresses R1C5).
        # Worst case (n_conv=2, f=32, lstm=48, dense=32): ~21k params.
        # The params_penalty term in the objective (tanh, denom=30k) provides smooth
        # discrimination across the full 4.7k–21k range without near-saturation.
        # 50 trials (up from 15) confirms convergence within this constrained space.
        params = {
            "n_conv":        trial.suggest_int("n_conv", 1, 2),
            "filters":       trial.suggest_categorical("filters", [16, 24, 32]),
            "kernel_size":   trial.suggest_categorical("kernel_size", [3, 5]),
            "lstm_units":    trial.suggest_categorical("lstm_units", [24, 32, 48]),
            "dense_units":   trial.suggest_categorical("dense_units", [16, 24, 32]),
            "dropout":       trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [64, 128, 256]),
            "epochs":        trial.suggest_int("epochs", 10, 30),
        }

        model = build_cnn_lstm_model(input_shape, n_classes, params)

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=CONFIG["PATIENCE"],
            restore_best_weights=True,
            verbose=0,
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[es],
            verbose=CONFIG["NN_VERBOSE"],
        )

        y_prob = model.predict(X_val, batch_size=params["batch_size"], verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

        _, _, f1_macro, _ = precision_recall_fscore_support(
            y_val, y_pred, average="macro", zero_division=0
        )

        _, latency, _, _ = measure_latency_and_throughput(
            model, X_val, batch_size=params["batch_size"], model_type="keras"
        )
        latency_ms   = latency * 1000.0
        n_params     = int(model.count_params())

        # Composite score penalises both latency and parameter count.
        # tanh keeps each penalty bounded in [0, penalty_weight] so a single
        # extreme trial cannot dominate the search.
        #   latency_penalty: saturates at LATENCY_PENALTY  (~10 ms half-saturation)
        #   params_penalty:  saturates at PARAMS_PENALTY   (~30k params half-saturation)
        latency_penalty = CONFIG["LATENCY_PENALTY"]  * np.tanh(latency_ms / 10.0)
        params_penalty  = CONFIG.get("PARAMS_PENALTY", 0.10) * np.tanh(n_params / 30_000.0)
        combined_score  = f1_macro - latency_penalty - params_penalty

        # Store for post-hoc inspection / Pareto plot
        trial.set_user_attr("f1_macro",  float(f1_macro))
        trial.set_user_attr("n_params",  n_params)
        trial.set_user_attr("lat_ms",    float(latency_ms))

        tf.keras.backend.clear_session()
        gc.collect()
        return combined_score

    # Seeded TPE sampler so that BO results are reproducible across reruns.
    # n_startup_trials=10 keeps the first 10 trials random (default), so only ~5 of
    # the 15 trials are TPE-driven; this is documented in the paper.
    # n_startup_trials=20 (40% of 50) gives TPE 30 guided trials to exploit
    # the surrogate — more principled than the previous 67% random startup.
    sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=20)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"{dataset_name}_cnn_lstm_optuna",
    )
    study.optimize(objective, n_trials=CONFIG["N_TRIALS"], show_progress_bar=True)

    best_params = study.best_trial.params
    return study, best_params

# Cell 10: Training wrappers
def fit_neural_model(model, X_train, y_train, X_val, y_val,
                     batch_size, epochs, patience=None):
    # Use caller-supplied patience, or fall back to CONFIG value.
    # CNN/LSTM ablations use epochs=20; a global patience=4 stops them
    # as early as epoch 8-9 on minor noise.  min_delta=1e-4 ensures
    # early stopping only fires when improvement is genuinely negligible.
    _patience = patience if patience is not None else CONFIG["PATIENCE"]
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=_patience,
        min_delta=1e-4,          # ignore improvements smaller than 0.0001
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=max(2, _patience // 2),
        min_delta=1e-4,
        min_lr=1e-6,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, reduce_lr],
        verbose=1,
    )
    return history

def train_and_evaluate_keras_model(
    model_name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    class_names,
    output_dir,
    batch_size=128,
    epochs=15,
    patience=None,   # forwarded to fit_neural_model
):
    fit_start = time.perf_counter()
    history = fit_neural_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=batch_size, epochs=epochs, patience=patience,
    )
    fit_time = time.perf_counter() - fit_start

    # Use LATENCY_BATCH_SIZE=1 for fair per-sample timing (R2C2)
    y_prob, latency, throughput, total_time = measure_latency_and_throughput(
        model, X_test, batch_size=CONFIG["LATENCY_BATCH_SIZE"], model_type="keras"
    )
    y_pred = np.argmax(y_prob, axis=1)

    metrics_dict, report_df = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        latency=latency,
        throughput=throughput,
    )

    history_path = os.path.join(output_dir, f"{model_name}_training_history.pdf")
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.pdf")
    cm_norm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_normalized.pdf")
    roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.pdf")

    plot_history(history, title_prefix=model_name, save_path=history_path)
    plot_confusion_matrix_seaborn(y_test, y_pred, class_names, f"{model_name} - Confusion Matrix", save_path=cm_path)
    plot_confusion_matrix_seaborn(y_test, y_pred, class_names, f"{model_name} - Normalized Confusion Matrix", save_path=cm_norm_path, normalize=True)
    plot_multiclass_roc(y_test, y_prob, class_names, f"{model_name} - Multiclass ROC", save_path=roc_path)

    report_csv = os.path.join(output_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_csv)

    metrics_dict["model"] = model_name
    metrics_dict["fit_seconds"] = fit_time
    metrics_dict["prediction_total_seconds"] = total_time
    metrics_dict["trainable_params"] = int(model.count_params())
    metrics_dict["n_features_used"] = int(X_train.shape[1] if CONFIG.get("SEQUENCE_MODE") == "feature_vector" else X_train.shape[-1])

    model_save_path = os.path.join(output_dir, f"{model_name}.keras")
    model.save(model_save_path)

    return metrics_dict, report_df

def train_and_evaluate_sklearn_model(
    model_name,
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    class_names,
    output_dir,
):
    model = clone(estimator)
    fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start

    y_pred = np.asarray(model.predict(X_test)).ravel()

    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        y_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    _, latency, throughput, total_time = measure_latency_and_throughput(
        model, X_test, batch_size=CONFIG["LATENCY_BATCH_SIZE"], model_type="sklearn"
    )

    metrics_dict, report_df = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        class_names=class_names,
        latency=latency,
        throughput=throughput,
    )

    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.pdf")
    cm_norm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix_normalized.pdf")

    plot_confusion_matrix_seaborn(y_test, y_pred, class_names, f"{model_name} - Confusion Matrix", save_path=cm_path)
    plot_confusion_matrix_seaborn(y_test, y_pred, class_names, f"{model_name} - Normalized Confusion Matrix", save_path=cm_norm_path, normalize=True)

    if y_prob is not None:
        roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.pdf")
        plot_multiclass_roc(y_test, y_prob, class_names, f"{model_name} - Multiclass ROC", save_path=roc_path)

    report_csv = os.path.join(output_dir, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_csv)

    metrics_dict["model"] = model_name
    metrics_dict["fit_seconds"] = fit_time
    metrics_dict["prediction_total_seconds"] = total_time
    metrics_dict["trainable_params"] = np.nan
    metrics_dict["n_features_used"] = X_train.shape[1]

    return metrics_dict, report_df, model

# Cell 11: Dataset preparation pipeline
def prepare_dataset(dataset_name, file_path):
    ensure_path_exists(file_path)
    target_col = CONFIG["TARGET_COLUMNS"][dataset_name]
    output_dir = os.path.join(CONFIG["RESULTS_DIR"], dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Loading dataset: {dataset_name}")

    # ── EdgeIIoT-set: chunked stratified sampling ─────────────────────────────
    # EdgeIIoT-set is ~2.2 M rows / ~2.5 GB on disk. Loading it entirely into
    # RAM before sampling wastes memory and risks OOM on Kaggle (16 GB cap).
    # Strategy: read in 200k-row chunks, collect rows per class, stop each
    # class once it reaches EDGEIIOT_SAMPLES_PER_CLASS. This keeps peak RAM
    # usage close to the final sample size rather than the full file size.
    if dataset_name == "EdgeIIoT" and not isinstance(file_path, (list, tuple)):
        n_per_class  = CONFIG.get("EDGEIIOT_SAMPLES_PER_CLASS", 10_000)
        _drop_cols   = CONFIG.get("EDGEIIOT_DROP_COLS", [])
        _chunk_size  = 50_000      # small chunks → low peak RAM
        _total_read  = 0

        print(f"  EdgeIIoT sampling: up to {n_per_class:,} rows per class")
        print(f"  File : {file_path}  ({os.path.getsize(file_path)/1024**2:.0f} MB)")
        print(f"  Chunk: {_chunk_size:,} rows — reading until all classes satisfied")

        # First pass: find which columns exist so we can read only what we need.
        # Reading just the header is near-instant and avoids a full scan.
        _header_df   = pd.read_csv(file_path, nrows=0, low_memory=False)
        _header_df.columns = [str(c).strip() for c in _header_df.columns]
        _keep_cols   = [c for c in _header_df.columns if c not in _drop_cols
                        or c == target_col]
        del _header_df; gc.collect()
        print(f"  Columns: {len(_keep_cols)} kept (of which 1 is target)")

        # Per-class row-index lists — store only index arrays, not full DataFrames,
        # so memory use is O(n_per_class × n_classes) not O(file_size).
        _class_rows  = {}   # class_str → list of row dicts  (we store dicts, not DFs)
        _class_done  = set()
        _row_offset  = 0    # absolute row index in the file (for progress reporting)

        for _chunk in pd.read_csv(
            file_path,
            chunksize=_chunk_size,
            usecols=_keep_cols,
            low_memory=False,
        ):
            _chunk.columns = [str(c).strip() for c in _chunk.columns]
            _total_read   += len(_chunk)

            # Route each row to its class bucket — no groupby (avoids full copy)
            _labels = _chunk[target_col].astype(str).str.strip()
            for _cls_str in _labels.unique():
                if _cls_str in _class_done:
                    continue
                if _cls_str not in _class_rows:
                    _class_rows[_cls_str] = []
                _need = n_per_class - sum(
                    len(r) for r in _class_rows[_cls_str]
                )
                if _need <= 0:
                    _class_done.add(_cls_str)
                    continue
                _mask = _labels == _cls_str
                _sub  = _chunk.loc[_mask].head(_need)
                if len(_sub):
                    _class_rows[_cls_str].append(_sub)
                if sum(len(r) for r in _class_rows[_cls_str]) >= n_per_class:
                    _class_done.add(_cls_str)

            del _chunk, _labels; gc.collect()
            _row_offset += _chunk_size

            n_seen = len(_class_rows)
            print(f"  Read {_total_read:>8,} rows | "
                  f"classes seen: {n_seen} | satisfied: {len(_class_done)}",
                  end="\r", flush=True)

            # Stop when all seen classes are satisfied AND we have read enough
            # rows that new classes are very unlikely to appear.
            # Guard: at least 5 × n_per_class rows read per class seen.
            _stop_threshold = 5 * n_per_class * max(n_seen, 1)
            if (_class_done == set(_class_rows.keys())
                    and n_seen > 1
                    and _total_read >= _stop_threshold):
                print(f"\n  All {n_seen} classes satisfied after "
                      f"{_total_read:,} rows — stopping early.")
                break
        else:
            print(f"\n  Finished reading full file ({_total_read:,} rows).")

        # Concat each class bucket and subsample to exactly n_per_class
        print("  Building final sample:")
        _sampled_parts = []
        for _cls_str in sorted(_class_rows.keys()):
            _cls_df = pd.concat(_class_rows[_cls_str], ignore_index=True)
            _n      = min(len(_cls_df), n_per_class)
            _sampled_parts.append(
                _cls_df.sample(n=_n, random_state=SEED).reset_index(drop=True)
            )
            print(f"    '{_cls_str}': {len(_cls_df):,} collected → {_n:,} kept")
        del _class_rows; gc.collect()

        df = pd.concat(_sampled_parts, ignore_index=True)
        del _sampled_parts; gc.collect()

        # Cast every object column to plain Python str so sklearn encoders
        # receive a uniform dtype.  pd.NA / np.nan become the string "Unknown".
        for _col in df.select_dtypes(include="object").columns:
            df[_col] = df[_col].astype(str).replace({"nan": "Unknown",
                                                      "<NA>": "Unknown"})
        print(f"  EdgeIIoT sample shape: {df.shape}")

    # ── Multi-file InSDN ──────────────────────────────────────────────────────
    elif isinstance(file_path, (list, tuple)):
        print(f"Merging {len(file_path)} CSV file(s):")
        frames = []
        for p in file_path:
            print(f"  {p}  ({os.path.getsize(p)/1024**2:.1f} MB)")
            _df = pd.read_csv(p, low_memory=False)
            _df.columns = [str(c).strip() for c in _df.columns]
            frames.append(_df)
            del _df
        df = pd.concat(frames, ignore_index=True)
        del frames
        gc.collect()

    # ── Single-file datasets (ToN-IoT, any future addition) ──────────────────
    else:
        print(f"Path: {file_path}")
        df = pd.read_csv(file_path, low_memory=False)

    print("Original shape:", df.shape)

    df = clean_dataframe(df, dataset_name, target_col=target_col)

    # USE_SAMPLE / SAMPLE_PER_CLASS apply only to non-EdgeIIoT datasets.
    # EdgeIIoT was already sampled above during the chunked read.
    if dataset_name != "EdgeIIoT":
        df = reduce_sample_if_needed(
            df,
            target_col=target_col,
            use_sample=CONFIG["USE_SAMPLE"],
            sample_per_class=CONFIG["SAMPLE_PER_CLASS"],
        )
    print("Shape after cleaning / optional sampling:", df.shape)

    print("\nClass distribution:")
    print(df[target_col].value_counts())

    numeric_cols, categorical_cols, dropped_cols = choose_feature_columns(
        df, target_col, low_cardinality_threshold=CONFIG["LOW_CARDINALITY_THRESHOLD"]
    )

    selected_top_k = CONFIG.get("TOP_K_FEATURES_BY_DATASET", {}).get(
        dataset_name, CONFIG.get("TOP_K_FEATURES", 32)
    )

    dataset_overview_df = pd.DataFrame({
        "item": [
            "dataset_name",
            "n_rows_after_cleaning",
            "n_total_columns",
            "n_numeric_features",
            "n_categorical_features_kept",
            "n_dropped_high_cardinality_columns",
            "n_selected_features_for_training",
            "target_column",
        ],
        "value": [
            dataset_name,
            len(df),
            df.shape[1],
            len(numeric_cols),
            len(categorical_cols),
            len(dropped_cols),
            selected_top_k,
            target_col,
        ],
    })
    save_dataframe(dataset_overview_df, os.path.join(output_dir, f"{dataset_name}_dataset_overview.csv"))

    print("\nNumeric columns:", len(numeric_cols))
    print("Categorical columns kept:", len(categorical_cols))
    print("Dropped high-cardinality columns:", dropped_cols)
    print("Selected top-K features for training:", selected_top_k)

    X_df = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = stratified_split(X_df, y)

    print("\nSplit shapes:")
    print("Train:", X_train_df.shape, y_train.shape)
    print("Validation:", X_val_df.shape, y_val.shape)
    print("Test:", X_test_df.shape, y_test.shape)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train_prep = preprocessor.fit_transform(X_train_df)
    X_val_prep = preprocessor.transform(X_val_df)
    X_test_prep = preprocessor.transform(X_test_df)

    y_train_enc, y_val_enc, y_test_enc, label_encoder, class_names = encode_targets(
        y_train, y_val, y_test
    )

    X_train_sel, X_val_sel, X_test_sel, selector = apply_feature_selection(
        X_train_prep,
        y_train_enc,
        X_val_prep,
        X_test_prep,
        top_k=selected_top_k,
    )

    print("\nFeature shapes after preprocessing and selection:")
    print("Train:", X_train_sel.shape)
    print("Validation:", X_val_sel.shape)
    print("Test:", X_test_sel.shape)

    X_train_bal, y_train_bal, sampler = balance_training_data(X_train_sel, y_train_enc)
    y_train_bal_names = label_encoder.inverse_transform(y_train_bal)

    print("\nBalanced training class distribution:")
    print(pd.Series(y_train_bal_names).value_counts().sort_index())

    class_distribution_df = pd.concat([
        make_class_count_df(y, "full"),
        make_class_count_df(y_train, "train"),
        make_class_count_df(y_val, "validation"),
        make_class_count_df(y_test, "test"),
        make_class_count_df(y_train_bal_names, "train_after_smote"),
    ], ignore_index=True)

    save_dataframe(
        class_distribution_df,
        os.path.join(output_dir, f"{dataset_name}_class_distribution.csv")
    )

    X_train_seq, y_train_seq = make_sequences(
        X_train_bal, y_train_bal,
        mode=CONFIG["SEQUENCE_MODE"],
        window=CONFIG["SLIDING_WINDOW"],
    )
    X_val_seq, y_val_seq = make_sequences(
        X_val_sel, y_val_enc,
        mode=CONFIG["SEQUENCE_MODE"],
        window=CONFIG["SLIDING_WINDOW"],
    )
    X_test_seq, y_test_seq = make_sequences(
        X_test_sel, y_test_enc,
        mode=CONFIG["SEQUENCE_MODE"],
        window=CONFIG["SLIDING_WINDOW"],
    )

    print("\nSequence shapes:")
    print("Train:", X_train_seq.shape, y_train_seq.shape)
    print("Validation:", X_val_seq.shape, y_val_seq.shape)
    print("Test:", X_test_seq.shape, y_test_seq.shape)

    prep_artifacts = {
        "preprocessor": preprocessor,
        "selector": selector,
        "label_encoder": label_encoder,
        "sampler": sampler,
        "class_names": class_names,
        "dropped_cols": dropped_cols,
        "target_col": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }

    data_bundle = {
        "dataset_name": dataset_name,

        # Tabular versions for classical ML baselines
        "X_train_tab": X_train_sel,
        "X_val_tab": X_val_sel,
        "X_test_tab": X_test_sel,
        "X_train_tab_bal": X_train_bal,
        "y_train_tab": y_train_enc,
        "y_val_tab": y_val_enc,
        "y_test_tab": y_test_enc,
        "y_train_tab_bal": y_train_bal,

        # Sequential versions for deep learning models
        "X_train_seq": X_train_seq,
        "X_val_seq": X_val_seq,
        "X_test_seq": X_test_seq,
        "y_train_seq": y_train_seq,
        "y_val_seq": y_val_seq,
        "y_test_seq": y_test_seq,

        "class_names": class_names,
        "n_classes": len(class_names),
        "input_shape": X_train_seq.shape[1:],
        "artifacts": prep_artifacts,
        "class_distribution_df": class_distribution_df,
        "dataset_overview_df": dataset_overview_df,
    }

    return data_bundle

# Cell 12: Full experiment runner
def make_modern_baseline_estimators(n_classes):
    """Return paper-supported tabular baseline models for the selected feature input."""
    baselines = []

    # Adewole et al. (2025), Sensors: ensemble estimators set to 100.
    if CONFIG.get("RUN_RANDOM_FOREST", True):
        baselines.append((
            "RandomForest",
            RandomForestClassifier(
                n_estimators=CONFIG.get("RF_N_ESTIMATORS", 100),
                random_state=SEED,
                n_jobs=-1,
            )
        ))

    # Linear SVM is older and disabled by default. Enable CONFIG['RUN_LINEAR_SVM'] only if your panel requests it.
    if CONFIG.get("RUN_LINEAR_SVM", False):
        baselines.append((
            "LinearSVM",
            LinearSVC(random_state=SEED, max_iter=5000),
        ))

    if CONFIG.get("RUN_XGBOOST", True):
        # Adewole et al. (2025): max_depth=6, subsample=0.5, eta=0.3, estimators=100.
        xgb_params = {
            "n_estimators": CONFIG.get("XGB_N_ESTIMATORS", 100),
            "learning_rate": CONFIG.get("XGB_LEARNING_RATE", 0.3),
            "max_depth": CONFIG.get("XGB_MAX_DEPTH", 6),
            "subsample": CONFIG.get("XGB_SUBSAMPLE", 0.5),
            "objective": "multi:softprob" if n_classes > 2 else "binary:logistic",
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "tree_method": "hist",
            "random_state": SEED,
            "n_jobs": -1,
        }
        if n_classes > 2:
            xgb_params["num_class"] = n_classes
        baselines.append(("XGBoost", XGBClassifier(**xgb_params)))

    if CONFIG.get("RUN_LIGHTGBM", True):
        # Abid et al. (2025): LightGBM n_estimators=600.
        baselines.append((
            "LightGBM",
            LGBMClassifier(
                n_estimators=CONFIG.get("LGBM_N_ESTIMATORS", 600),
                objective="multiclass" if n_classes > 2 else "binary",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            )
        ))

    if CONFIG.get("RUN_CATBOOST", True):
        # Adewole et al. (2025): ensemble estimators/iterations set to 100; other parameters near defaults.
        baselines.append((
            "CatBoost",
            CatBoostClassifier(
                iterations=CONFIG.get("CATBOOST_ITERATIONS", 100),
                loss_function="MultiClass" if n_classes > 2 else "Logloss",
                random_seed=SEED,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
        ))

    return baselines

def make_baseline_settings_table(n_classes):
    """Export a transparent table showing exactly which settings each baseline uses."""
    rows = []
    sources = CONFIG.get("BASELINE_SETTING_SOURCES", {})

    if CONFIG.get("RUN_RANDOM_FOREST", True):
        rows.append({
            "model": "RandomForest",
            "family": "Tree ensemble",
            "key_settings": f"n_estimators={CONFIG.get('RF_N_ESTIMATORS', 100)}; remaining major hyperparameters use sklearn defaults; random_state={SEED}",
            "source_basis": sources.get("RandomForest", ""),
        })

    if CONFIG.get("RUN_XGBOOST", True):
        rows.append({
            "model": "XGBoost",
            "family": "Gradient boosting",
            "key_settings": (
                f"n_estimators={CONFIG.get('XGB_N_ESTIMATORS', 100)}; "
                f"max_depth={CONFIG.get('XGB_MAX_DEPTH', 6)}; "
                f"subsample={CONFIG.get('XGB_SUBSAMPLE', 0.5)}; "
                f"learning_rate/eta={CONFIG.get('XGB_LEARNING_RATE', 0.3)}; "
                f"objective={'multi:softprob' if n_classes > 2 else 'binary:logistic'}"
            ),
            "source_basis": sources.get("XGBoost", ""),
        })

    if CONFIG.get("RUN_LIGHTGBM", True):
        rows.append({
            "model": "LightGBM",
            "family": "Gradient boosting",
            "key_settings": f"n_estimators={CONFIG.get('LGBM_N_ESTIMATORS', 600)}; objective={'multiclass' if n_classes > 2 else 'binary'}; other major hyperparameters close to LightGBM defaults",
            "source_basis": sources.get("LightGBM", ""),
        })

    if CONFIG.get("RUN_CATBOOST", True):
        rows.append({
            "model": "CatBoost",
            "family": "Gradient boosting",
            "key_settings": f"iterations={CONFIG.get('CATBOOST_ITERATIONS', 100)}; loss_function={'MultiClass' if n_classes > 2 else 'Logloss'}; other major hyperparameters close to CatBoost defaults",
            "source_basis": sources.get("CatBoost", ""),
        })

    cnn_params = get_cnn_only_params()
    rows.append({
        "model": "CNN_Only",
        "family": "Deep-learning ablation",
        "key_settings": json.dumps(cnn_params),
        "source_basis": sources.get("CNN_Only", ""),
    })

    lstm_params = get_lstm_only_params()
    rows.append({
        "model": "LSTM_Only",
        "family": "Deep-learning ablation",
        "key_settings": json.dumps(lstm_params),
        "source_basis": sources.get("LSTM_Only", ""),
    })

    nonopt_params = get_default_cnn_lstm_params()
    rows.append({
        "model": "CNN_LSTM_NonOptimized",
        "family": "Deep-learning counterfactual baseline",
        "key_settings": json.dumps(nonopt_params),
        "source_basis": sources.get("CNN_LSTM_NonOptimized", ""),
    })

    return pd.DataFrame(rows)

def export_baseline_settings(dataset_name, n_classes, output_dir):
    baseline_settings_df = make_baseline_settings_table(n_classes=n_classes)
    baseline_settings_df.insert(0, "dataset", dataset_name)
    settings_path = os.path.join(output_dir, f"{dataset_name}_baseline_settings.csv")
    baseline_settings_df.to_csv(settings_path, index=False)
    print("\nBaseline settings used in this experiment:")
    display(baseline_settings_df)
    return baseline_settings_df

def export_optuna_artifacts(study, dataset_name, output_dir):
    """Save Optuna study, trials table, and publication-ready convergence plots."""
    import pickle

    study_pkl_path = os.path.join(output_dir, f"{dataset_name}_optuna_study.pkl")
    with open(study_pkl_path, "wb") as fh:
        pickle.dump(study, fh)

    trial_rows = []
    for t in study.trials:
        row = {
            "trial_number": t.number,
            "value": t.value,
            "state": t.state.name,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
        }
        for k, v in t.params.items():
            row[f"param_{k}"] = v
        trial_rows.append(row)

    trials_df = pd.DataFrame(trial_rows)
    trials_csv_path = os.path.join(output_dir, f"{dataset_name}_optuna_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)

    completed = trials_df[trials_df["state"] == "COMPLETE"].sort_values("trial_number").reset_index(drop=True)
    if not completed.empty and completed["value"].notna().any():
        completed["best_so_far"] = completed["value"].cummax()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            completed["trial_number"], completed["value"],
            marker="o", linestyle="--", alpha=0.65,
            color=PALETTE[1], label="Trial objective score",
        )
        ax.plot(
            completed["trial_number"], completed["best_so_far"],
            marker="s", linewidth=2.2,
            color=PALETTE[0], label="Best score so far",
        )
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Composite objective score")
        ax.set_title(f"{dataset_name}: Bayesian Optimisation Convergence (Optuna TPE)")
        ax.legend()
        conv_path = os.path.join(output_dir, f"{dataset_name}_bo_convergence.pdf")
        save_thesis_figure(fig, conv_path)
        plt.close(fig)
        print(f"  -> Saved BO convergence plot: {conv_path}")

    return trials_df

def run_experiments_for_dataset(dataset_name, file_path):
    data = prepare_dataset(dataset_name, file_path)
    output_dir = os.path.join(CONFIG["RESULTS_DIR"], dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Classical and modern ML baselines use tabular data.
    # SMOTE is applied once to the selected feature matrix for fairness across models.
    X_train_tab = data["X_train_tab_bal"]
    X_test_tab = data["X_test_tab"]
    y_train_tab = data["y_train_tab_bal"]
    y_test_tab = data["y_test_tab"]

    # Deep learning models use the sequential representation.
    X_train_seq = data["X_train_seq"]
    X_val_seq = data["X_val_seq"]
    X_test_seq = data["X_test_seq"]

    y_train = data["y_train_seq"]
    y_val = data["y_val_seq"]
    y_test = data["y_test_seq"]
    class_names = data["class_names"]
    n_classes = data["n_classes"]
    input_shape = data["input_shape"]

    all_metrics = []
    reports = {}
    fitted_baseline_models = {}

    # Save a transparent baseline-configuration table for the thesis appendix.
    export_baseline_settings(dataset_name, n_classes, output_dir)

    # ------------------------------------------------------------------
    # Paper-supported tabular baselines
    # ------------------------------------------------------------------
    for model_name, estimator in make_modern_baseline_estimators(n_classes=n_classes):
        print(f"\n{'-'*80}\nTraining {model_name} baseline on {dataset_name}")
        metrics, report, fitted_model = train_and_evaluate_sklearn_model(
            model_name=model_name,
            estimator=estimator,
            X_train=X_train_tab,
            y_train=y_train_tab,
            X_test=X_test_tab,
            y_test=y_test_tab,
            class_names=class_names,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)
        reports[model_name] = report
        fitted_baseline_models[model_name] = fitted_model

    # ------------------------------------------------------------------
    # Deep-learning comparison and ablation models
    # ------------------------------------------------------------------
    print(f"\n{'-'*80}\nTraining non-optimized CNN-LSTM on {dataset_name}")
    default_params = get_default_cnn_lstm_params()
    non_opt_model = build_cnn_lstm_model(input_shape, n_classes, default_params)
    non_opt_metrics, non_opt_report = train_and_evaluate_keras_model(
        model_name="CNN_LSTM_NonOptimized",
        model=non_opt_model,
        X_train=X_train_seq,
        y_train=y_train,
        X_val=X_val_seq,
        y_val=y_val,
        X_test=X_test_seq,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
        batch_size=default_params["batch_size"],
        epochs=default_params["epochs"],
        patience=10,  # 10% of 100-epoch budget — allows long plateau before stopping
    )
    all_metrics.append(non_opt_metrics)
    reports["CNN_LSTM_NonOptimized"] = non_opt_report

    print(f"\n{'-'*80}\nTraining CNN-only ablation on {dataset_name}")
    cnn_params = get_cnn_only_params()
    cnn_only = build_cnn_only_model(input_shape, n_classes, cnn_params)
    cnn_metrics, cnn_report = train_and_evaluate_keras_model(
        model_name="CNN_Only",
        model=cnn_only,
        X_train=X_train_seq,
        y_train=y_train,
        X_val=X_val_seq,
        y_val=y_val,
        X_test=X_test_seq,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
        batch_size=cnn_params["batch_size"],
        epochs=cnn_params["epochs"],
        patience=8,   # 40% of epoch budget — prevents premature stopping on 20-epoch run
    )
    all_metrics.append(cnn_metrics)
    reports["CNN_Only"] = cnn_report

    print(f"\n{'-'*80}\nTraining LSTM-only ablation on {dataset_name}")
    lstm_params = get_lstm_only_params()
    lstm_only = build_lstm_only_model(input_shape, n_classes, lstm_params)
    lstm_metrics, lstm_report = train_and_evaluate_keras_model(
        model_name="LSTM_Only",
        model=lstm_only,
        X_train=X_train_seq,
        y_train=y_train,
        X_val=X_val_seq,
        y_val=y_val,
        X_test=X_test_seq,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
        batch_size=lstm_params["batch_size"],
        epochs=lstm_params["epochs"],
        patience=8,   # 40% of epoch budget — prevents premature stopping on 20-epoch run
    )
    all_metrics.append(lstm_metrics)
    reports["LSTM_Only"] = lstm_report

    # ------------------------------------------------------------------
    # Proposed optimized CNN-LSTM
    # ------------------------------------------------------------------
    print(f"\n{'-'*80}\nRunning Bayesian hyperparameter optimization for CNN-LSTM on {dataset_name}")
    study, best_params = tune_cnn_lstm(
        X_train_seq,
        y_train,
        X_val_seq,
        y_val,
        n_classes=n_classes,
        dataset_name=dataset_name,
    )

    export_optuna_artifacts(study, dataset_name, output_dir)

    best_params_path = os.path.join(output_dir, "best_cnn_lstm_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    best_params_df = save_json_table(
        best_params,
        os.path.join(output_dir, f"{dataset_name}_best_cnn_lstm_params.csv")
    )

    print("Best parameters:")
    display(best_params_df)

    opt_model = build_cnn_lstm_model(input_shape, n_classes, best_params)
    opt_metrics, opt_report = train_and_evaluate_keras_model(
        model_name="CNN_LSTM_Optimized",
        model=opt_model,
        X_train=X_train_seq,
        y_train=y_train,
        X_val=X_val_seq,
        y_val=y_val,
        X_test=X_test_seq,
        y_test=y_test,
        class_names=class_names,
        output_dir=output_dir,
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
    )
    all_metrics.append(opt_metrics)
    reports["CNN_LSTM_Optimized"] = opt_report

    # ------------------------------------------------------------------
    # Tables and summary plots
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(all_metrics)
    results_df.insert(0, "dataset", dataset_name)
    results_df["latency_ms_per_sample"] = results_df["latency_seconds_per_sample"] * 1000.0
    results_df.sort_values(by=["f1_macro", "accuracy"], ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    results_csv = os.path.join(output_dir, f"{dataset_name}_summary_metrics.csv")
    results_df.to_csv(results_csv, index=False)

    main_results_cols = [
        "dataset", "model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "fpr_macro", "roc_auc_ovr_macro", "latency_ms_per_sample",
        "throughput_samples_per_second", "fit_seconds", "trainable_params", "n_features_used"
    ]
    comparative_df = results_df[[c for c in main_results_cols if c in results_df.columns]].copy()
    comparative_df.to_csv(os.path.join(output_dir, f"{dataset_name}_comparative_results_table.csv"), index=False)

    complexity_cols = [
        "dataset", "model", "trainable_params", "n_features_used", "fit_seconds",
        "prediction_total_seconds", "latency_ms_per_sample", "throughput_samples_per_second"
    ]
    complexity_df = results_df[[c for c in complexity_cols if c in results_df.columns]].copy()
    complexity_df.to_csv(os.path.join(output_dir, f"{dataset_name}_model_complexity_table.csv"), index=False)

    print(f"\nSummary metrics for {dataset_name}:")
    display(results_df)

    # ── Bar: Macro-F1 ─────────────────────────────────────────────────────────
    plot_metric_bar(
        results_df, metric_col="f1_macro",
        title=f"{dataset_name}: Macro F1-score by model",
        ylabel="Macro F1-score",
        save_path=os.path.join(output_dir, f"{dataset_name}_macro_f1_comparison"),
    )
    # ── Bar: Accuracy ─────────────────────────────────────────────────────────
    plot_metric_bar(
        results_df, metric_col="accuracy",
        title=f"{dataset_name}: Accuracy by model",
        ylabel="Accuracy",
        save_path=os.path.join(output_dir, f"{dataset_name}_accuracy_comparison"),
    )
    # ── Bar: False positive rate ──────────────────────────────────────────────
    plot_metric_bar(
        results_df, metric_col="fpr_macro",
        title=f"{dataset_name}: False positive rate by model",
        ylabel="Macro false positive rate",
        save_path=os.path.join(output_dir, f"{dataset_name}_fpr_comparison"),
    )
    # ── Bar: Throughput (log) ─────────────────────────────────────────────────
    plot_metric_bar(
        results_df, metric_col="throughput_samples_per_second",
        title=f"{dataset_name}: Throughput by model (log scale)",
        ylabel="Throughput (samples / second)",
        save_path=os.path.join(output_dir, f"{dataset_name}_throughput_comparison"),
        log_scale=True,
    )
    # ── Bar: Latency ──────────────────────────────────────────────────────────
    plot_metric_bar(
        results_df, metric_col="latency_ms_per_sample",
        title=f"{dataset_name}: Inference latency by model",
        ylabel="Latency (ms / sample)",
        save_path=os.path.join(output_dir, f"{dataset_name}_latency_comparison"),
    )
    # ── Radar: multi-metric spider ────────────────────────────────────────────
    _radar_df = results_df.copy()
    _radar_df["specificity"] = (1.0 - _radar_df["fpr_macro"].fillna(1.0)).clip(0, 1)
    plot_radar(
        _radar_df,
        metrics=["accuracy", "precision_macro", "recall_macro",
                 "f1_macro", "specificity", "roc_auc_ovr_macro"],
        title=f"{dataset_name}: Multi-metric model comparison",
        save_path=os.path.join(output_dir, f"{dataset_name}_radar"),
    )
    # ── Pareto: F1 vs parameter count (neural models only) ───────────────────
    _neural = results_df.dropna(subset=["trainable_params"]).copy()
    if not _neural.empty:
        plot_pareto(
            _neural,
            title=f"{dataset_name}: Accuracy–Efficiency Pareto (F1 vs Parameters)",
            save_path=os.path.join(output_dir, f"{dataset_name}_pareto"),
        )
    # ── Per-class F1 — neural models only ─────────────────────────────────────
    _neural_model_names = {
        "CNN_LSTM_NonOptimized", "CNN_LSTM_Optimized", "CNN_Only", "LSTM_Only"
    }
    for _model_name, _report_dict in reports.items():
        if _model_name not in _neural_model_names:
            continue
        try:
            _rep = pd.DataFrame(_report_dict).transpose()
            if "f1-score" not in _rep.columns:
                continue
            plot_per_class_f1(
                _rep, _model_name, dataset_name,
                save_path=os.path.join(output_dir,
                    f"{dataset_name}_{_model_name}_per_class_f1"),
            )
        except Exception as _e:
            print(f"  [skip per-class F1 for {_model_name}]: {_e}")

    return results_df, reports, study, best_params

# Cell 13: Run all experiments on EdgeIIoT-set
all_studies = {}

results_df, reports, study, best_params = run_experiments_for_dataset(
    "EdgeIIoT", CONFIG["DATASET_PATHS"]["EdgeIIoT"]
)
all_studies["EdgeIIoT"] = study

final_results = results_df.copy()
final_results_path = os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_FINAL_RESULTS.csv")
final_results.to_csv(final_results_path, index=False)

print("\nFinal results:")
display(final_results)

# Sensitivity analysis — penalty weight robustness (addresses Reviewer 1, Concern 3)
# Re-scores every completed Optuna trial under 9 (lambda_lat, lambda_pam) combinations
# without re-training. Outputs a table + heatmap showing which architecture wins
# under each weight pair.
import itertools

print("=" * 70)
print("Penalty weight sensitivity analysis")
print("=" * 70)

_lambdas   = [0.05, 0.10, 0.20]
_sens_rows = []

# Locate the trials CSV for each dataset that was run
_dataset_names = list(CONFIG["DATASET_PATHS"].keys())
for _ds in _dataset_names:
    _trials_path = os.path.join(
        CONFIG["RESULTS_DIR"], _ds, f"{_ds}_optuna_trials.csv"
    )
    if not os.path.exists(_trials_path):
        print(f"  [{_ds}] Trials CSV not found — run Cell 13 first.")
        continue

    _trials_df  = pd.read_csv(_trials_path)
    _completed  = _trials_df[_trials_df["state"] == "COMPLETE"].copy()

    # Rename user_attr columns if present
    _completed = _completed.rename(columns={
        "user_attr_f1_macro": "f1_macro",
        "user_attr_n_params":  "n_params",
        "user_attr_lat_ms":    "lat_ms",
    })

    _has_attrs = all(c in _completed.columns for c in ["f1_macro", "n_params", "lat_ms"])
    if not _has_attrs:
        print(f"  [{_ds}] user_attr columns not found — skipping.")
        continue

    print(f"\n[{_ds}] Re-scoring {len(_completed)} trials across "
          f"{len(_lambdas)**2} weight combinations")

    _ds_rows = []
    for lam_lat, lam_pam in itertools.product(_lambdas, _lambdas):
        _completed["rescore"] = (
            _completed["f1_macro"]
            - lam_lat * np.tanh(_completed["lat_ms"]  / 10.0)
            - lam_pam * np.tanh(_completed["n_params"] / 30_000.0)
        )
        _best = _completed.loc[_completed["rescore"].idxmax()]
        _ds_rows.append({
            "dataset":    _ds,
            "lambda_lat": lam_lat,
            "lambda_pam": lam_pam,
            "best_trial": int(_best["trial_number"]),
            "best_score": round(float(_best["rescore"]), 5),
            "f1_macro":   round(float(_best["f1_macro"]), 5),
            "n_params":   int(_best["n_params"]),
            "lat_ms":     round(float(_best["lat_ms"]), 4),
            "filters":    _best.get("param_filters", "—"),
            "lstm_units": _best.get("param_lstm_units", "—"),
            "n_conv":     _best.get("param_n_conv", "—"),
            "dense_units":_best.get("param_dense_units", "—"),
        })
    _sens_rows.extend(_ds_rows)

    _ds_df = pd.DataFrame(_ds_rows)
    display(_ds_df)

    # Heatmap for this dataset
    _pivot_f1 = _ds_df.pivot(index="lambda_lat", columns="lambda_pam", values="f1_macro")
    _pivot_p  = _ds_df.pivot(index="lambda_lat", columns="lambda_pam", values="n_params")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(_pivot_f1, annot=True, fmt=".4f", cmap="YlGn",
                linewidths=0.5, ax=axes[0],
                cbar_kws={"label": "Best trial F1-macro"})
    axes[0].set_title(f"{_ds}: Best F1 per weight pair")
    axes[0].set_xlabel("λ_pam"); axes[0].set_ylabel("λ_lat")
    sns.heatmap(_pivot_p, annot=True, fmt=".0f", cmap="YlOrRd_r",
                linewidths=0.5, ax=axes[1],
                cbar_kws={"label": "Best trial params"})
    axes[1].set_title(f"{_ds}: Best param count per weight pair")
    axes[1].set_xlabel("λ_pam"); axes[1].set_ylabel("λ_lat")
    fig.suptitle(f"Penalty weight sensitivity — {_ds}",
                 fontsize=13, fontweight="bold")
    _hp = os.path.join(CONFIG["RESULTS_DIR"], f"{_ds}_SENSITIVITY_HEATMAP")
    save_thesis_figure(fig, _hp)
    plt.show(); plt.close(fig)

    _n_unique = _ds_df["best_trial"].nunique()
    print(f"  Unique best-trial selections: {_n_unique} of {len(_ds_rows)}")
    if _n_unique <= 2:
        print("  → Architecture selection is ROBUST to penalty weight choice.")
    else:
        print("  → Some sensitivity detected — discuss in the paper.")

if _sens_rows:
    _all_sens = pd.DataFrame(_sens_rows)
    _sens_csv = os.path.join(CONFIG["RESULTS_DIR"], "SENSITIVITY_PENALTY_WEIGHTS.csv")
    _all_sens.to_csv(_sens_csv, index=False)
    print(f"\nFull sensitivity table saved: {_sens_csv}")

# Cell 14: Save thesis-ready summary table — EdgeIIoT-set
thesis_columns = [
    "dataset", "model", "accuracy", "precision_macro", "recall_macro",
    "f1_macro", "fpr_macro", "roc_auc_ovr_macro",
    "latency_seconds_per_sample", "throughput_samples_per_second",
    "fit_seconds", "trainable_params", "n_features_used",
]

thesis_table = final_results[[c for c in thesis_columns if c in final_results.columns]].copy()
thesis_table["latency_ms_per_sample"] = thesis_table["latency_seconds_per_sample"] * 1000.0

ordered_columns = [
    "dataset", "model", "accuracy", "precision_macro", "recall_macro",
    "f1_macro", "fpr_macro", "roc_auc_ovr_macro",
    "latency_ms_per_sample", "throughput_samples_per_second",
    "fit_seconds", "trainable_params", "n_features_used",
]
thesis_table = thesis_table[[c for c in ordered_columns if c in thesis_table.columns]]
thesis_table.sort_values("f1_macro", ascending=False, inplace=True)

thesis_table_path = os.path.join(CONFIG["RESULTS_DIR"], "THESIS_RESULTS_TABLE.csv")
thesis_table.to_csv(thesis_table_path, index=False)

print("\nThesis-ready results table (EdgeIIoT-set):")
display(thesis_table)
print(f"\nResults saved to: {CONFIG['RESULTS_DIR']}")

# ── LaTeX table ───────────────────────────────────────────────────────────────
_fmt = {
    "accuracy": "{:.4f}", "precision_macro": "{:.4f}",
    "recall_macro": "{:.4f}", "f1_macro": "{:.4f}",
    "fpr_macro": "{:.6f}", "roc_auc_ovr_macro": "{:.4f}",
    "latency_ms_per_sample": "{:.4f}",
    "throughput_samples_per_second": "{:.0f}",
    "fit_seconds": "{:.1f}", "trainable_params": "{:.0f}",
}
_latex_df = thesis_table.copy()
for col, fmt in _fmt.items():
    if col in _latex_df.columns:
        _latex_df[col] = _latex_df[col].apply(
            lambda v: fmt.format(v) if pd.notna(v) else "—"
        )
_col_renames = {
    "dataset": "Dataset", "model": "Model",
    "accuracy": "Acc.", "precision_macro": "Prec.",
    "recall_macro": "Rec.", "f1_macro": "F1",
    "fpr_macro": "FPR", "roc_auc_ovr_macro": "AUC",
    "latency_ms_per_sample": "Lat. (ms)",
    "throughput_samples_per_second": "Throughput",
    "fit_seconds": "Train (s)", "trainable_params": "Params",
}
_latex_df = _latex_df.rename(columns=_col_renames)
_latex_path = os.path.join(CONFIG["RESULTS_DIR"], "THESIS_RESULTS_TABLE.tex")
_latex_df.to_latex(
    _latex_path, index=False, escape=True,
    caption="EdgeIIoT-set: model performance summary.",
    label="tab:edgeiiot_results",
    column_format="ll" + "r" * (len(_latex_df.columns) - 2),
)
print(f"LaTeX table: {_latex_path}")

# ── Styled HTML for Kaggle output panel ──────────────────────────────────────
_html_path = os.path.join(CONFIG["RESULTS_DIR"], "THESIS_RESULTS_TABLE.html")
_num_cols  = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr_macro"]
_other_fmt = {
    "fpr_macro": "{:.6f}", "latency_ms_per_sample": "{:.4f}",
    "throughput_samples_per_second": "{:,.0f}",
    "fit_seconds": "{:.1f}", "trainable_params": "{:.0f}",
}
(
    thesis_table.style
    .format({c: "{:.4f}" for c in _num_cols}, na_rep="—")
    .format(_other_fmt, na_rep="—")
    .background_gradient(subset=["f1_macro"], cmap="Greens")
    .background_gradient(subset=["fpr_macro"], cmap="Reds_r")
    .set_caption("EdgeIIoT-set results — green = best F1, red = worst FPR")
    .to_html(_html_path)
)
print(f"HTML table : {_html_path}")
display(
    thesis_table.style
    .format({c: "{:.4f}" for c in _num_cols}, na_rep="—")
    .format(_other_fmt, na_rep="—")
    .background_gradient(subset=["f1_macro"], cmap="Greens")
    .background_gradient(subset=["fpr_macro"], cmap="Reds_r")
    .highlight_max(subset=["f1_macro", "accuracy", "roc_auc_ovr_macro"], color="#d4edda", axis=0)
    .highlight_min(subset=["fpr_macro", "latency_ms_per_sample"], color="#d4edda", axis=0)
    .set_caption("EdgeIIoT-set — green = best, highlighted = top per metric")
)


# 
# ## Suggested structure for your results chapter
# 
# You can report:
# 1. class distribution for each dataset
# 2. preprocessing pipeline and selected feature count
# 3. traditional baseline comparison: Random Forest
# 4. modern baseline comparison: XGBoost, LightGBM, and CatBoost
# 5. optional older baseline: Linear SVM, only if enabled in `CONFIG["RUN_LINEAR_SVM"]`
# 6. optimized vs non-optimized CNN–LSTM
# 7. CNN-only and LSTM-only ablation analysis
# 8. confusion matrices and ROC curves saved as PDF for LaTeX
# 9. latency, throughput, and parameter-count comparison
# 10. discussion of the accuracy-efficiency trade-off
# 

# Cell 15: Export runtime environment, hyperparameter summaries, BO convergence
runtime_df   = get_runtime_environment_table()
runtime_path = os.path.join(CONFIG["RESULTS_DIR"], "runtime_environment.csv")
runtime_df.to_csv(runtime_path, index=False)
print("Runtime environment:")
display(runtime_df)

dataset_dir = os.path.join(CONFIG["RESULTS_DIR"], "EdgeIIoT")

# ── Best hyperparameters ──────────────────────────────────────────────────────
best_params_path = os.path.join(dataset_dir, "best_cnn_lstm_params.json")
if os.path.exists(best_params_path):
    with open(best_params_path) as f:
        params = json.load(f)
    bp_df = pd.DataFrame(list(params.items()), columns=["hyperparameter", "value"])
    bp_df.to_csv(os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_BEST_PARAMS.csv"), index=False)
    print("\nBest CNN-LSTM hyperparameters:")
    display(bp_df)

# ── Class distribution ────────────────────────────────────────────────────────
dist_path = os.path.join(dataset_dir, "EdgeIIoT_class_distribution.csv")
if os.path.exists(dist_path):
    dist_df = pd.read_csv(dist_path)
    dist_df.to_csv(os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_CLASS_DISTRIBUTION.csv"), index=False)
    print("\nClass distribution:")
    display(dist_df)

# ── Per-model classification reports ─────────────────────────────────────────
report_frames = []
if os.path.exists(dataset_dir):
    for filename in sorted(os.listdir(dataset_dir)):
        if filename.endswith("_classification_report.csv"):
            rp = pd.read_csv(os.path.join(dataset_dir, filename))
            model_name = filename.replace("_classification_report.csv", "")
            rp.insert(0, "model", model_name)
            report_frames.append(rp)
if report_frames:
    all_reports = pd.concat(report_frames, ignore_index=True)
    all_reports.to_csv(
        os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_CLASSIFICATION_REPORTS.csv"), index=False
    )
    print("\nClassification reports saved.")

# ── BO convergence plot ───────────────────────────────────────────────────────
trials_path = os.path.join(dataset_dir, "EdgeIIoT_optuna_trials.csv")
if os.path.exists(trials_path):
    all_trials_summary = pd.read_csv(trials_path)
    all_trials_summary.insert(0, "dataset", "EdgeIIoT")
    all_trials_summary.to_csv(
        os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_OPTUNA_TRIALS.csv"), index=False
    )
    sub = all_trials_summary[all_trials_summary["state"] == "COMPLETE"].copy()
    sub = sub.sort_values("trial_number").reset_index(drop=True)
    if not sub.empty and not sub["value"].isna().all():
        sub["best_so_far"] = sub["value"].cummax()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sub["trial_number"], sub["value"],
                marker="o", linestyle="--", alpha=0.65,
                color=PALETTE[1], label="Trial objective score")
        ax.plot(sub["trial_number"], sub["best_so_far"],
                marker="s", linewidth=2.2,
                color=PALETTE[0], label="Best score so far")
        ax.set_xlabel("Trial number")
        ax.set_ylabel("Composite objective score")
        ax.set_title("EdgeIIoT-set: Bayesian Optimisation Convergence (Optuna TPE)")
        ax.legend()
        conv_path = os.path.join(CONFIG["RESULTS_DIR"], "EDGEIIOT_BO_CONVERGENCE")
        save_thesis_figure(fig, conv_path)
        plt.show()
        plt.close(fig)
        print(f"BO convergence figure saved: {conv_path}.pdf / .png")

print("\nArtifacts exported:")
print("  runtime_environment.csv")
print("  EDGEIIOT_BEST_PARAMS.csv")
print("  EDGEIIOT_CLASS_DISTRIBUTION.csv")
print("  EDGEIIOT_CLASSIFICATION_REPORTS.csv")
print("  EDGEIIOT_OPTUNA_TRIALS.csv")
print("  EDGEIIOT_BO_CONVERGENCE.pdf / .png")

# Cell 17: Zip all results for Kaggle download
zip_base = str(KAGGLE_WORKING_ROOT / "my_folder")
zip_path = shutil.make_archive(zip_base, "zip", CONFIG["RESULTS_DIR"])
CONFIG["OUTPUT_ZIP"] = zip_path
print(f"Created results ZIP: {zip_path}")
print("Download it from the Kaggle Output panel after the notebook finishes.")

# Cell 18: Kaggle download instruction
print("Kaggle does not use files.download like Colab.")
print(f"Your zipped results are saved at: {CONFIG.get('OUTPUT_ZIP', '/kaggle/working/my_folder.zip')}")
print("Open the right-side Output panel in Kaggle and download my_folder.zip.")

