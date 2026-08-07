"""Model builders for the baseline, embedding and transformer methods."""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

from .pi_pipeline import (Config, FIG_DIR, RES_DIR, compute_metrics,
                          measure_latency, RANDOM_STATE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
PALETTE = sns.color_palette("colorblind")


# Method 1: TF-IDF baseline (Logistic Regression / Linear SVM)
def build_tfidf_features():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           min_df=3, max_features=40000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=3, max_features=40000, sublinear_tf=True)
    return FeatureUnion([("word", word), ("char", char)])


def train_baselines(X_train, y_train, cfg: Config):
    """Fit the TF-IDF Logistic Regression and Linear SVM with C tuning."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True,
                          random_state=cfg.random_state)
    models = {}

    # Logistic Regression
    lr_pipe = Pipeline([("feats", build_tfidf_features()),
                        ("clf", LogisticRegression(max_iter=2000,
                                                   class_weight="balanced"))])
    lr_grid = GridSearchCV(lr_pipe, {"clf__C": [0.5, 1.0, 2.0, 4.0]},
                           scoring="f1", cv=skf, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    models["LogReg (TF-IDF)"] = lr_grid.best_estimator_

    # Linear SVM, calibrated so it can output probabilities
    svm_base = Pipeline([("feats", build_tfidf_features()),
                         ("clf", LinearSVC(class_weight="balanced"))])
    svm_grid = GridSearchCV(svm_base, {"clf__C": [0.5, 1.0, 2.0]},
                            scoring="f1", cv=skf, n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    from sklearn.svm import LinearSVC as _LSVC
    best_c = svm_grid.best_params_["clf__C"]
    svm_cal = Pipeline([("feats", build_tfidf_features()),
                        ("clf", CalibratedClassifierCV(
                            _LSVC(C=best_c, class_weight="balanced"),
                            cv=3, method="sigmoid"))])
    svm_cal.fit(X_train, y_train)
    models["Linear SVM (TF-IDF)"] = svm_cal

    tuning = {"LogReg (TF-IDF)": lr_grid.best_params_,
              "Linear SVM (TF-IDF)": {"clf__C": best_c}}
    return models, tuning


# Method 2: MiniLM sentence embeddings -> Random Forest / XGBoost
class EmbeddingEncoder:
    """Encodes text into MiniLM sentence-embedding vectors."""
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=64, show=False):
        return self.model.encode(list(texts), batch_size=batch_size,
                                 show_progress_bar=show,
                                 convert_to_numpy=True, normalize_embeddings=True)


class EmbeddingTextClassifier:
    """Takes raw text and returns an injection probability. Stores the embedding
    model name and the fitted classifier so it can be pickled and loaded by the
    API."""
    def __init__(self, embed_model_name: str, classifier):
        self.embed_model_name = embed_model_name
        self.classifier = classifier
        self._encoder = None

    def _enc(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embed_model_name)
        return self._encoder

    def predict_proba(self, texts):
        emb = self._enc().encode(list(texts), normalize_embeddings=True,
                                 convert_to_numpy=True)
        return self.classifier.predict_proba(emb)

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


def train_embedding_models(Xtr_emb, y_train, cfg: Config):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    models = {}
    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=cfg.random_state)
    rf.fit(Xtr_emb, y_train)
    models["RandomForest (MiniLM)"] = rf

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    spw = neg / max(pos, 1.0)
    xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                        subsample=0.9, colsample_bytree=0.9,
                        scale_pos_weight=spw, eval_metric="logloss",
                        tree_method="hist", n_jobs=-1,
                        random_state=cfg.random_state)
    xgb.fit(Xtr_emb, y_train)
    models["XGBoost (MiniLM)"] = xgb
    return models


# Method 3: fine-tuned transformer encoder
class TransformerClassifier:
    """Fine-tunes a transformer encoder for sequence classification."""

    def __init__(self, cfg: Config):
        import torch
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  logging as hf_logging)
        hf_logging.set_verbosity_error()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg.transformer_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.transformer_model, num_labels=2).to(self.device)

    def _encode(self, texts):
        import torch
        enc = self.tok(list(texts), truncation=True, padding=True,
                       max_length=self.cfg.transformer_max_len,
                       return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def fit(self, texts, labels):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        enc = self.tok(list(texts), truncation=True, padding="max_length",
                       max_length=self.cfg.transformer_max_len)
        ids = torch.tensor(enc["input_ids"])
        mask = torch.tensor(enc["attention_mask"])
        y = torch.tensor(list(labels))
        ds = TensorDataset(ids, mask, y)
        dl = DataLoader(ds, batch_size=self.cfg.transformer_batch, shuffle=True)

        # class weights for imbalance
        pos = float((np.array(labels) == 1).sum())
        neg = float((np.array(labels) == 0).sum())
        w = torch.tensor([1.0, neg / max(pos, 1.0)], dtype=torch.float).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.model.parameters(), lr=2e-5)

        self.model.train()
        for ep in range(self.cfg.transformer_epochs):
            tot = 0.0
            for bids, bmask, by in dl:
                bids, bmask, by = bids.to(self.device), bmask.to(self.device), by.to(self.device)
                opt.zero_grad()
                out = self.model(input_ids=bids, attention_mask=bmask)
                loss = loss_fn(out.logits, by)
                loss.backward()
                opt.step()
                tot += float(loss)
            print(f"    [transformer] epoch {ep+1}/{self.cfg.transformer_epochs} "
                  f"loss={tot/len(dl):.4f}")
        return self

    def predict_proba(self, texts, batch_size=32):
        import torch
        self.model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts)[i:i + batch_size]
                enc = self._encode(batch)
                out = self.model(**enc)
                p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
                probs.append(p)
        return np.concatenate(probs) if probs else np.array([])

    def predict(self, texts):
        return (self.predict_proba(texts) >= 0.5).astype(int)
