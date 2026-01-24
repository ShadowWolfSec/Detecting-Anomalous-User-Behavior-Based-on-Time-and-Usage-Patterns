# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:27:15 2026

@author: rasheed
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from config import RISK_BANDS, FEATURE_FIELDS
from data_loader import load_model, load_selected_features
from feature_engineering import engineer_features


def build_single_input(
    user: str,
    activity: str,
    action: str,
    decoder_name: str,
    level: str,
    timestamp: datetime,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Timestamp": timestamp.isoformat(),
                "User": user,
                "Activity": activity,
                "Action": action,
                "Decoder_name": decoder_name,
                "Level": level,
                "Anomaly": 0,
            }
        ]
    )


def get_risk_band(score: float) -> Dict[str, str]:
    for threshold, label, advice in RISK_BANDS:
        if score >= threshold:
            return {"label": label, "advice": advice}
    return {"label": "Low", "advice": "Normal behavior"}


def coerce_selected_features(X: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    if not selected_features:
        return X
    for col in selected_features:
        if col not in X.columns:
            X[col] = 0
    return X[selected_features]


def normalize_importance(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty or "importance_mean" not in work.columns:
        return work
    work = work[work["importance_mean"] > 0].copy()
    total = work["importance_mean"].sum()
    if total <= 0:
        return work
    work["impact_percent"] = (work["importance_mean"] / total) * 100
    return work.sort_values("impact_percent", ascending=False)


def prepare_metric_chart_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    chart_df = df.copy()
    if "model" not in chart_df.columns:
        chart_df = chart_df.rename(columns={"model": "model"})
    chart_df["model"] = chart_df["model"].astype(str)
    return chart_df.set_index("model")[[metric]].sort_values(metric, ascending=False)


def score_single_event(
    input_mode: str,
    raw_inputs: Dict[str, Any],
    derived_inputs: Dict[str, Any],
    model_path: Path,
) -> Tuple[float, Dict[str, str], pd.DataFrame]:
    """معالجة مدخلات الحدث الفردي وحساب درجة الخطورة"""
    selected_features = load_selected_features()

    if input_mode in {"Raw fields", "Both"}:
        timestamp = datetime.combine(raw_inputs["event_date"], raw_inputs["event_time"])
        payload = build_single_input(
            raw_inputs["user"],
            raw_inputs["activity"],
            raw_inputs["action"],
            raw_inputs["decoder_name"],
            raw_inputs["level"],
            timestamp,
        )
        engineered, _ = engineer_features(payload)
        X_raw = engineered.drop(columns=["Anomaly"])
    else:
        X_raw = pd.DataFrame([derived_inputs], columns=FEATURE_FIELDS)

    if input_mode == "Both":
        X = X_raw.copy()
        for key, value in derived_inputs.items():
            X[key] = value
    else:
        X = X_raw

    X = coerce_selected_features(X, selected_features)

    model = load_model(model_path)
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[:, 1][0])
    else:
        decision = model.decision_function(X)
        score = float((decision - decision.min()) / (decision.max() - decision.min() + 1e-9))[0]

    band = get_risk_band(score)
    return score, band, X


def score_batch_dataframe(batch_df: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """معالجة ملف CSV كامل وإرجاع جدول النتائج"""
    work_df = batch_df.copy()
    if "Anomaly" not in work_df.columns:
        work_df["Anomaly"] = 0
    work_df["Anomaly"] = (
        pd.to_numeric(work_df["Anomaly"], errors="coerce")
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .astype(int)
    )

    engineered_batch, _ = engineer_features(work_df)
    X_batch = engineered_batch.drop(columns=["Anomaly"])
    X_batch = coerce_selected_features(X_batch, load_selected_features())
    
    model = load_model(model_path)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_batch)[:, 1]
    else:
        decision = model.decision_function(X_batch)
        scores = (decision - decision.min()) / (decision.max() - decision.min() + 1e-9)

    risk_labels = [get_risk_band(float(s))["label"] for s in scores]
    results = work_df.copy()
    results["anomaly_probability_percent"] = (scores * 100).round(2)
    results["risk_level"] = risk_labels
    return results