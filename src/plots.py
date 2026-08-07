"""Functions that build the figures used in the report and notebook."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .pi_pipeline import FIG_DIR

sns.set_theme(style="whitegrid", context="paper")
CB = sns.color_palette("colorblind")


def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_class_distribution(train_pool, external):
    df = pd.concat([train_pool, external], ignore_index=True)
    counts = (df.groupby(["dataset", "label"]).size()
              .rename("n").reset_index())
    counts["label"] = counts["label"].map({0: "Benign", 1: "Injection"})
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(data=counts, x="dataset", y="n", hue="label",
                palette=[CB[2], CB[3]], ax=ax)
    ax.set_title("Class distribution per dataset")
    ax.set_xlabel(""); ax.set_ylabel("Number of prompts")
    ax.set_yscale("log")
    for c in ax.containers:
        ax.bar_label(c, fmt="%d", fontsize=7, padding=2)
    ax.legend(title="Class", loc="upper right")
    return _save(fig, "fig1_class_distribution.png")


def fig_length_distribution(train_pool):
    d = train_pool.copy()
    d["len"] = d["text"].str.len().clip(upper=1500)
    d["label"] = d["label"].map({0: "Benign", 1: "Injection"})
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.kdeplot(data=d, x="len", hue="label", fill=True, common_norm=False,
                palette=[CB[2], CB[3]], ax=ax, clip=(0, 1500))
    ax.set_title("Prompt length distribution by class (training pool)")
    ax.set_xlabel("Prompt length (characters, capped at 1500)")
    return _save(fig, "fig2_length_distribution.png")


def fig_confusion_matrices(cms: dict, title="Confusion matrices (external test)"):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Benign", "Inj"], yticklabels=["Benign", "Inj"], ax=ax)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle(title, y=1.03, fontsize=10)
    return _save(fig, "fig5_confusion_matrices.png")


def fig_roc_curves(roc_data: dict, title="ROC curves (external test)", fname="fig3_roc.png"):
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, (y, s)) in enumerate(roc_data.items()):
        if s is None or np.all(np.isnan(s)):
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})",
                color=CB[i % len(CB)], lw=1.6)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    return _save(fig, fname)


def fig_pr_curves(roc_data: dict, title="Precision-Recall curves (external test)",
                  fname="fig4_pr.png"):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, (name, (y, s)) in enumerate(roc_data.items()):
        if s is None or np.all(np.isnan(s)):
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                color=CB[i % len(CB)], lw=1.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower left")
    return _save(fig, fname)


def fig_metric_comparison(results_df):
    """results_df: columns [model, split, accuracy, precision, recall, f1, roc_auc]."""
    d = results_df.melt(id_vars=["model", "split"],
                        value_vars=["accuracy", "precision", "recall", "f1"],
                        var_name="metric", value_name="score")
    g = sns.catplot(data=d, x="metric", y="score", hue="model", col="split",
                    kind="bar", height=3.6, aspect=1.15, palette="colorblind",
                    legend_out=True)
    g.set_axis_labels("", "Score")
    g.set_titles("{col_name} test")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1.0)
    g.figure.suptitle("Model performance comparison", y=1.05, fontsize=11)
    path = os.path.join(FIG_DIR, "fig6_metric_comparison.png")
    g.figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(g.figure)
    return path


def fig_latency(latency: dict):
    d = pd.DataFrame({"model": list(latency.keys()),
                      "ms": list(latency.values())}).sort_values("ms")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(data=d, y="model", x="ms", palette="viridis", ax=ax)
    ax.set_title("Mean per-request inference latency (single prompt, CPU)")
    ax.set_xlabel("Latency (ms/request)"); ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.1f", fontsize=8, padding=3)
    return _save(fig, "fig7_latency.png")


def fig_lr_top_features(lr_pipeline, top_n=18):
    """Most influential TF-IDF tokens for the Logistic Regression model."""
    feats = lr_pipeline.named_steps["feats"]
    clf = lr_pipeline.named_steps["clf"]
    names = feats.get_feature_names_out()
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    top_pos = order[-top_n:][::-1]
    top_neg = order[:top_n]
    idx = np.concatenate([top_neg, top_pos[::-1]])
    d = pd.DataFrame({"token": names[idx], "coef": coefs[idx]})
    d["direction"] = np.where(d["coef"] >= 0, "→ Injection", "→ Benign")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(data=d, y="token", x="coef", hue="direction",
                palette={"→ Injection": CB[3], "→ Benign": CB[2]},
                dodge=False, ax=ax)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_title("Most influential TF-IDF tokens (Logistic Regression)")
    ax.set_xlabel("Model coefficient"); ax.set_ylabel("")
    ax.legend(title="", fontsize=8)
    return _save(fig, "fig8_lr_features.png")


def fig_generalization_gap(results_df):
    """F1 drop from internal -> external test = cross-dataset generalization gap."""
    piv = results_df.pivot_table(index="model", columns="split", values="f1")
    piv = piv.dropna()
    piv["gap"] = piv.get("internal", np.nan) - piv.get("external", np.nan)
    piv = piv.sort_values("gap")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    sns.barplot(y=piv.index, x=piv["gap"], palette="rocket", ax=ax)
    ax.set_title("Cross-dataset generalization gap (internal F1 − external F1)")
    ax.set_xlabel("F1 drop (lower is better)"); ax.set_ylabel("")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, padding=3)
    return _save(fig, "fig9_generalization_gap.png")


def fig_shap_summary(xgb_model, X_emb_sample, fname="fig10_shap.png"):
    """SHAP explanation of the XGBoost-on-embeddings model (global importance)."""
    import shap
    expl = shap.TreeExplainer(xgb_model)
    sv = expl.shap_values(X_emb_sample)
    fig = plt.figure(figsize=(6, 4))
    shap.summary_plot(sv, X_emb_sample, show=False,
                      feature_names=[f"dim_{i}" for i in range(X_emb_sample.shape[1])],
                      max_display=15, plot_size=None)
    fig = plt.gcf()
    fig.suptitle("SHAP feature importance — XGBoost on MiniLM embeddings", fontsize=9)
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
