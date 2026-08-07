"""
Runs the full prompt-injection detection experiment and writes every figure to
figures/ and every metric to results/.

Usage:
    python run_experiments.py                 # full run (defaults)
    python run_experiments.py --quick         # tiny smoke test
    python run_experiments.py --no-transformer
"""
from __future__ import annotations

import argparse

from src.pi_pipeline import Config
from src.experiment import run_pipeline


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test run")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--ahsan-sample", type=int, default=None)
    ap.add_argument("--tf-sample", type=int, default=None)
    ap.add_argument("--tf-epochs", type=int, default=None)
    ap.add_argument("--tf-model", type=str, default=None)
    return ap.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    if args.quick:
        cfg.ahsan_sample = 1500
        cfg.transformer_sample = 400
        cfg.transformer_epochs = 1
        cfg.latency_n = 60
    if args.no_transformer:
        cfg.run_transformer = False
    if args.no_shap:
        cfg.run_shap = False
    if args.ahsan_sample is not None:
        cfg.ahsan_sample = args.ahsan_sample
    if args.tf_sample is not None:
        cfg.transformer_sample = args.tf_sample
    if args.tf_epochs is not None:
        cfg.transformer_epochs = args.tf_epochs
    if args.tf_model is not None:
        cfg.transformer_model = args.tf_model
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    print("=" * 70)
    print("CONFIG:", {k: getattr(cfg, k) for k in vars(cfg)})
    print("=" * 70)
    out = run_pipeline(cfg)
    ext = out["results_df"]
    ext = ext[ext.split == "external"].set_index("model")
    print("\n=== EXTERNAL (cross-dataset) RESULTS ===")
    print(ext[["accuracy", "precision", "recall", "f1", "roc_auc"]].round(3).to_string())


if __name__ == "__main__":
    main()
