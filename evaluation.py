# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:01:08 2026

@author: rasheed
Evaluation pipeline ."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from feature_engineering import engineer_features, load_dataset

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"


def _apply_clustering_features(X: pd.DataFrame) -> pd.DataFrame:
    cluster_path = METADATA_DIR / "cluster_model.pkl"
    if not cluster_path.exists():
        return X
    bundle = joblib.load(cluster_path)
    scaler = bundle.get("scaler")
    clusterer = bundle.get("clusterer")
    pca = bundle.get("pca")
    features = bundle.get("features", [])
    if not features:
        return X
    work = X.copy()
    numeric_df = work.reindex(columns=features, fill_value=0).copy()
    scaled = scaler.transform(numeric_df)
    if pca is not None:
        scaled = pca.transform(scaled)
    if hasattr(clusterer, "fit_predict") and hasattr(clusterer, "labels_"):
        labels = clusterer.fit_predict(scaled)
    else:
        labels = clusterer.fit_predict(scaled) if hasattr(clusterer, "fit_predict") else clusterer.predict(scaled)
    work["Cluster_ID"] = labels
    return work


def evaluate_models(data_path: str) -> pd.DataFrame:
    test_split_path = METADATA_DIR / "test_split.csv"
    if test_split_path.exists():
        test_df = pd.read_csv(test_split_path)
        X = test_df.drop(columns=["Anomaly"])
        y = test_df["Anomaly"].astype(int)
    else:
        df = load_dataset(data_path)
        engineered, _ = engineer_features(df)
        X = engineered.drop(columns=["Anomaly"])
        y = engineered["Anomaly"].astype(int)

    X = _apply_clustering_features(X)

    selected_path = METADATA_DIR / "selected_features.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)["selected_features"].tolist()
        missing = [c for c in selected if c not in X.columns]
        if missing:
            X = X.reindex(columns=selected, fill_value=0)
        else:
            X = X[selected]

    results = []
    for model_path in MODELS_DIR.glob("*.pkl"):
        model = joblib.load(model_path)
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            roc_auc = roc_auc_score(y, proba)
        else:
            decision = model.decision_function(X)
            proba = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)
            roc_auc = roc_auc_score(y, proba)

        results.append({
            "model": model_path.stem,
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "f1_score": f1_score(y, preds, zero_division=0),
            "roc_auc": roc_auc,
        })

    results_df = pd.DataFrame(results).sort_values("f1_score", ascending=False)
    results_df.to_csv(METADATA_DIR / "model_test_metrics.csv", index=False)
    return results_df


if __name__ == "__main__":
    evaluate_models("logs_SPEDIA_annotated_en.csv")
