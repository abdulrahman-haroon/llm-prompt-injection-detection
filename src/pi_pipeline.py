"""Data loading, configuration and metric helpers for the prompt-injection
detection experiment."""
from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# project paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "Datasets")
FIG_DIR = os.path.join(ROOT, "figures")
RES_DIR = os.path.join(ROOT, "results")
MODEL_DIR = os.path.join(ROOT, "models")
for _d in (FIG_DIR, RES_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

RANDOM_STATE = 42


@dataclass
class Config:
    """Experiment settings."""
    ahsan_sample: int = 12000          # subsample size for the large corpus
    test_size: float = 0.20            # internal hold-out proportion
    cv_folds: int = 5
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    transformer_model: str = "distilbert-base-uncased"  # or microsoft/deberta-v3-base
    transformer_sample: int = 6000     # subsample size for transformer fine-tuning
    transformer_epochs: int = 2
    transformer_batch: int = 16
    transformer_max_len: int = 128
    latency_n: int = 300               # number of single requests timed
    run_transformer: bool = True
    run_shap: bool = True
    random_state: int = RANDOM_STATE


def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_datasets(cfg: Config):
    """Load the three corpora into one table with columns text, label, dataset.
    deepset is returned separately as the external test set."""
    # ahsanayub/malicious-prompts (primary training corpus)
    ahsan = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, "ahsanayub-malicious-prompts", "train.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "ahsanayub-malicious-prompts", "test.csv")),
    ], ignore_index=True)[["text", "label"]]
    ahsan["dataset"] = "ahsan"

    # stratified subsample of the large corpus
    if cfg.ahsan_sample and len(ahsan) > cfg.ahsan_sample:
        frac_pos = ahsan["label"].mean()
        n_pos = int(round(cfg.ahsan_sample * frac_pos))
        n_neg = cfg.ahsan_sample - n_pos
        pos = ahsan[ahsan.label == 1].sample(n=n_pos, random_state=cfg.random_state)
        neg = ahsan[ahsan.label == 0].sample(n=n_neg, random_state=cfg.random_state)
        ahsan = pd.concat([pos, neg], ignore_index=True)

    # reshabhs/SPML_Chatbot_Prompt_Injection
    spml = pd.read_csv(os.path.join(DATA_DIR, "reshabhs-SPML_Chatbot_Prompt_Injection",
                                    "spml_prompt_injection.csv"))
    spml = spml.rename(columns={"User Prompt": "text", "Prompt injection": "label"})
    spml = spml[["text", "label"]]
    spml["dataset"] = "spml"

    # deepset/prompt-injections (external test set)
    deepset = pd.concat([
        pd.read_csv(os.path.join(DATA_DIR, "deepset-prompt-injections",
                                 "train-00000-of-00001-9564e8b05b4757ab.csv")),
        pd.read_csv(os.path.join(DATA_DIR, "deepset-prompt-injections",
                                 "test-00000-of-00001-701d16158af87368.csv")),
    ], ignore_index=True)[["text", "label"]]
    deepset["dataset"] = "deepset"

    # clean and de-duplicate
    def _prep(df):
        df = df.copy()
        df["text"] = df["text"].map(_clean_text)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        df = df[df["text"].str.len() > 0]
        df = df.drop_duplicates(subset=["text"])
        return df.reset_index(drop=True)

    ahsan, spml, deepset = _prep(ahsan), _prep(spml), _prep(deepset)

    train_pool = pd.concat([ahsan, spml], ignore_index=True)
    # drop training rows that also appear in the external test set
    train_pool = train_pool[~train_pool["text"].isin(set(deepset["text"]))]
    train_pool = train_pool.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)

    return train_pool, deepset


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            m["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            m["roc_auc"] = float("nan")
    return m


def measure_latency(predict_fn, samples, n=300) -> float:
    """Mean per-request latency in milliseconds."""
    xs = list(samples)[:n]
    if not xs:
        return float("nan")
    predict_fn(xs[0])  # warm-up
    t0 = time.perf_counter()
    for x in xs:
        predict_fn(x)
    dt = time.perf_counter() - t0
    return (dt / len(xs)) * 1000.0
