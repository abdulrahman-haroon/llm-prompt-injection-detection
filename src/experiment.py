"""Runs the full experiment: loads data, trains the models, evaluates them,
and saves the figures and metrics. run_pipeline(cfg) does all of this."""
from __future__ import annotations

import os
import json
import time

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix

from .pi_pipeline import (Config, RES_DIR, MODEL_DIR, load_datasets,
                          compute_metrics, measure_latency)
from . import models as M
from . import plots as P


def run_pipeline(cfg: Config, verbose: bool = True) -> dict:
    def log(*a):
        if verbose:
            print(*a, flush=True)

    t_start = time.time()

    # load and summarise the datasets
    log("[1/8] Loading & unifying datasets ...")
    train_pool, external = load_datasets(cfg)
    summary = (pd.concat([train_pool, external])
               .groupby("dataset")
               .agg(n=("label", "size"),
                    injection=("label", "sum"),
                    injection_rate=("label", "mean"))
               .reset_index())
    summary["injection_rate"] = (summary["injection_rate"] * 100).round(1)
    summary.to_csv(os.path.join(RES_DIR, "dataset_summary.csv"), index=False)
    log(summary.to_string(index=False))

    X = train_pool["text"].values
    y = train_pool["label"].values
    X_ext, y_ext = external["text"].values, external["label"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state)
    log(f"    internal train={len(X_tr)}  internal test={len(X_te)}  "
        f"external={len(X_ext)}")

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                          random_state=cfg.random_state)

    results, roc_ext, roc_int, cms_ext, latency, cv_scores = [], {}, {}, {}, {}, {}

    def record(name, split, y_true, y_score):
        y_pred = (np.asarray(y_score) >= 0.5).astype(int)
        m = compute_metrics(y_true, y_pred, y_score)
        m.update({"model": name, "split": split})
        results.append(m)
        return y_pred

    # exploratory figures
    log("[2/8] Exploratory figures ...")
    P.fig_class_distribution(train_pool, external)
    P.fig_length_distribution(train_pool)

    # method 1: TF-IDF baselines
    log("[3/8] Training baseline models (TF-IDF + LR / Linear-SVM) ...")
    baselines, tuning = M.train_baselines(X_tr, y_tr, cfg)
    for name, mdl in baselines.items():
        cv = cross_val_score(mdl, X_tr, y_tr, cv=skf, scoring="f1", n_jobs=-1)
        cv_scores[name] = {"f1_mean": float(cv.mean()), "f1_std": float(cv.std())}
        log(f"    {name}: CV F1 = {cv.mean():.3f} ± {cv.std():.3f}")
        s_int = mdl.predict_proba(X_te)[:, 1]
        s_ext = mdl.predict_proba(X_ext)[:, 1]
        record(name, "internal", y_te, s_int)
        yp = record(name, "external", y_ext, s_ext)
        roc_int[name], roc_ext[name] = (y_te, s_int), (y_ext, s_ext)
        cms_ext[name] = confusion_matrix(y_ext, yp)
        latency[name] = measure_latency(lambda t: mdl.predict([t]), X_ext, cfg.latency_n)

    # method 2: MiniLM embeddings + tree models
    log("[4/8] Encoding MiniLM embeddings ...")
    encoder = M.EmbeddingEncoder(cfg.embed_model)
    Xtr_emb = encoder.encode(X_tr, show=False)
    Xte_emb = encoder.encode(X_te, show=False)
    Xext_emb = encoder.encode(X_ext, show=False)

    log("[5/8] Training embedding classifiers (RandomForest / XGBoost) ...")
    emb_models = M.train_embedding_models(Xtr_emb, y_tr, cfg)
    for name, mdl in emb_models.items():
        cv = cross_val_score(mdl, Xtr_emb, y_tr, cv=skf, scoring="f1", n_jobs=-1)
        cv_scores[name] = {"f1_mean": float(cv.mean()), "f1_std": float(cv.std())}
        log(f"    {name}: CV F1 = {cv.mean():.3f} ± {cv.std():.3f}")
        s_int = mdl.predict_proba(Xte_emb)[:, 1]
        s_ext = mdl.predict_proba(Xext_emb)[:, 1]
        record(name, "internal", y_te, s_int)
        yp = record(name, "external", y_ext, s_ext)
        roc_int[name], roc_ext[name] = (y_te, s_int), (y_ext, s_ext)
        cms_ext[name] = confusion_matrix(y_ext, yp)
        latency[name] = measure_latency(
            lambda t: mdl.predict(encoder.encode([t])), X_ext, cfg.latency_n)

    # method 3: fine-tuned transformer
    if cfg.run_transformer:
        log(f"[6/8] Fine-tuning transformer ({cfg.transformer_model}) ...")
        n = min(cfg.transformer_sample, len(X_tr))
        sub = pd.DataFrame({"text": X_tr, "label": y_tr})
        if n < len(sub):
            sub, _ = train_test_split(sub, train_size=n, stratify=sub["label"],
                                      random_state=cfg.random_state)
        tclf = M.TransformerClassifier(cfg)
        tclf.fit(sub["text"].tolist(), sub["label"].tolist())
        name = f"Transformer ({cfg.transformer_model.split('/')[-1]})"
        s_int = tclf.predict_proba(list(X_te))
        s_ext = tclf.predict_proba(list(X_ext))
        record(name, "internal", y_te, s_int)
        yp = record(name, "external", y_ext, s_ext)
        roc_int[name], roc_ext[name] = (y_te, s_int), (y_ext, s_ext)
        cms_ext[name] = confusion_matrix(y_ext, yp)
        latency[name] = measure_latency(lambda t: tclf.predict([t]), X_ext, cfg.latency_n)
    else:
        log("[6/8] Transformer skipped.")

    # figures
    log("[7/8] Generating result figures ...")
    results_df = pd.DataFrame(results)
    P.fig_roc_curves(roc_ext, "ROC curves (external / cross-dataset test)", "fig3_roc.png")
    P.fig_pr_curves(roc_ext, "Precision-Recall (external / cross-dataset test)", "fig4_pr.png")
    P.fig_confusion_matrices(cms_ext)
    P.fig_metric_comparison(results_df)
    P.fig_latency(latency)
    P.fig_generalization_gap(results_df)
    P.fig_lr_top_features(baselines["LogReg (TF-IDF)"])
    if cfg.run_shap:
        try:
            P.fig_shap_summary(emb_models["XGBoost (MiniLM)"], Xte_emb[:400])
        except Exception as e:
            log("    SHAP figure skipped:", e)

    # save results and models
    log("[8/8] Saving results ...")
    results_df.to_csv(os.path.join(RES_DIR, "results_table.csv"), index=False)
    for fname, obj in [("metrics.json", results),
                       ("cv_scores.json", cv_scores),
                       ("latency.json", latency)]:
        with open(os.path.join(RES_DIR, fname), "w") as f:
            json.dump(obj, f, indent=2)
    joblib.dump(baselines["LogReg (TF-IDF)"],
                os.path.join(MODEL_DIR, "baseline_model.joblib"))
    # save the best model as a raw-text classifier for the API
    best = M.EmbeddingTextClassifier(cfg.embed_model,
                                     emb_models["XGBoost (MiniLM)"])
    joblib.dump(best, os.path.join(MODEL_DIR, "best_model.joblib"))

    elapsed = time.time() - t_start
    log(f"\nTotal wall-clock: {elapsed:0.1f}s")
    return {
        "results_df": results_df, "cv_scores": cv_scores, "latency": latency,
        "tuning": tuning, "summary": summary, "roc_ext": roc_ext,
        "cms_ext": cms_ext, "elapsed": elapsed,
    }
