# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:03:20 2026

@author: rasheed
"""

import joblib
import pandas as pd

from pathlib import Path

# تحميل النموذج (اختَر واحدًا)
model = joblib.load("artifacts/models/random_forest.pkl")

METADATA_DIR = Path("artifacts") / "metadata"


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
    numeric_df = X.reindex(columns=features, fill_value=0).copy()
    scaled = scaler.transform(numeric_df)
    if pca is not None:
        scaled = pca.transform(scaled)
    labels = clusterer.predict(scaled) if hasattr(clusterer, "predict") else clusterer.fit_predict(scaled)
    work = X.copy()
    work["Cluster_ID"] = labels
    return work

# إنشاء سجل واحد يدويًا
sample = pd.DataFrame([{
        "hour": 23,
        "weekday": 5,
        "is_weekend": 1,
        "is_off_hours": 1,
        "is_night": 1,
        "user_event_index": 80,
        "time_since_last_event": 0.1,
        "days_since_first_activity": 3.0,
        "activity_rate": 5.0,
        "is_new_user": 1,
        "is_first_activity": 0,
        "daily_event_index": 20,
        "hours_since_first_today": 0.5,
        "daily_activity_rate": 6.0,
        "activity_is_missing": 0,
        "action_is_missing": 0,
        "decoder_is_missing": 0,
        "level_is_high": 1,
}])

sample = _apply_clustering_features(sample)

# التنبؤ
prediction = model.predict(sample)
probability = model.predict_proba(sample)[0][1]

print("Prediction:", "Anomaly 🚨" if prediction[0] == 1 else "Normal ✅")
print("Anomaly probability:", round(probability, 3))
