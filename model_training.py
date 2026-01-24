# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:01:08 2026

@author: rasheed
Model training and feature selection for ML5.
Strict UBA focus with temporal/behavioral features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional
    XGBClassifier = None

from feature_engineering import FeatureDocumentation, engineer_features, load_dataset

RANDOM_STATE = 42
CLUSTER_METHOD = "compare"  # options: dbscan, kmeans, agglomerative, compare
CLUSTER_K = 3
CLUSTER_EPS = 1.2
CLUSTER_MIN_SAMPLES = 10
USE_PCA_BEFORE_CLUSTER = True
PCA_COMPONENTS = 5
TSNE_SAMPLE_SIZE = 2000
AGGLOMERATIVE_MAX_SAMPLES = 50000
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureSelectionReport:
    selected_features: List[str]
    dropped_features: Dict[str, str]
    statistical_scores: pd.DataFrame
    importance_scores: pd.DataFrame


@dataclass
class TrainReport:
    metrics: pd.DataFrame
    models: Dict[str, Pipeline]
    selected_features: List[str]


def _split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical = [c for c in df.columns if df[c].dtype == "object"]
    numeric = [c for c in df.columns if df[c].dtype != "object"]
    return categorical, numeric


def _prepare_encoded(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    categorical, numeric = _split_columns(df)
    work = df.copy()
    for col in numeric:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work[numeric] = work[numeric].fillna(work[numeric].median())
    if categorical:
        work[categorical] = work[categorical].fillna("missing")
        work = pd.get_dummies(work, columns=categorical, drop_first=False)
    return work, categorical, numeric


def _run_clustering(method: str, scaled: np.ndarray) -> Tuple[np.ndarray, object]:
    if method == "kmeans":
        clusterer = KMeans(n_clusters=CLUSTER_K, random_state=RANDOM_STATE, n_init=10)
        labels = clusterer.fit_predict(scaled)
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=CLUSTER_K)
        labels = clusterer.fit_predict(scaled)
    else:
        clusterer = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES)
        labels = clusterer.fit_predict(scaled)
    return labels, clusterer


def _add_clustering_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Add Cluster_ID as pseudo-label feature and return clustering report."""
    numeric_df = df.select_dtypes(exclude=["object"]).copy()
    if numeric_df.empty:
        df["Cluster_ID"] = -1
        return df, pd.DataFrame()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    if USE_PCA_BEFORE_CLUSTER and scaled.shape[1] > PCA_COMPONENTS:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
        scaled_for_cluster = pca.fit_transform(scaled)
    else:
        pca = None
        scaled_for_cluster = scaled

    methods = [CLUSTER_METHOD] if CLUSTER_METHOD != "compare" else ["dbscan", "kmeans", "agglomerative"]
    reports = []
    best_labels = None
    best_clusterer = None
    best_score = -np.inf

    for method in methods:
        if method == "agglomerative" and scaled_for_cluster.shape[0] > AGGLOMERATIVE_MAX_SAMPLES:
            reports.append({
                "cluster_method": method,
                "clusters": 0,
                "silhouette_score": np.nan,
                "davies_bouldin": np.nan,
                "note": f"skipped (n>{AGGLOMERATIVE_MAX_SAMPLES})",
            })
            continue
        labels, clusterer = _run_clustering(method, scaled_for_cluster)
        unique_labels = set(labels)
        metrics = {
            "cluster_method": method,
            "clusters": len(unique_labels),
        }
        if len(unique_labels) > 1 and not (len(unique_labels) == 1 and -1 in unique_labels):
            try:
                metrics["silhouette_score"] = silhouette_score(scaled_for_cluster, labels)
            except Exception:
                metrics["silhouette_score"] = np.nan
            try:
                metrics["davies_bouldin"] = davies_bouldin_score(scaled_for_cluster, labels)
            except Exception:
                metrics["davies_bouldin"] = np.nan
        else:
            metrics["silhouette_score"] = np.nan
            metrics["davies_bouldin"] = np.nan

        reports.append(metrics)
        score = metrics["silhouette_score"] if not np.isnan(metrics["silhouette_score"]) else -np.inf
        if score > best_score:
            best_score = score
            best_labels = labels
            best_clusterer = clusterer

    cluster_report = pd.DataFrame(reports)
    labels = best_labels if best_labels is not None else np.full(len(df), -1)

    if TSNE_SAMPLE_SIZE > 0:
        sample_size = min(TSNE_SAMPLE_SIZE, scaled_for_cluster.shape[0])
        sample_idx = np.random.RandomState(RANDOM_STATE).choice(scaled_for_cluster.shape[0], sample_size, replace=False)
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, init="pca", learning_rate="auto")
        tsne_coords = tsne.fit_transform(scaled_for_cluster[sample_idx])
        vis_df = pd.DataFrame({
            "tsne_1": tsne_coords[:, 0],
            "tsne_2": tsne_coords[:, 1],
            "Cluster_ID": labels[sample_idx],
        })
        vis_df.to_csv(METADATA_DIR / "cluster_tsne.csv", index=False)

    if pca is not None:
        pca_coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
        pd.DataFrame({
            "pca_1": pca_coords[:, 0],
            "pca_2": pca_coords[:, 1],
            "Cluster_ID": labels,
        }).to_csv(METADATA_DIR / "cluster_pca.csv", index=False)
    df = df.copy()
    df["Cluster_ID"] = labels

    joblib.dump(
        {
            "scaler": scaler,
            "clusterer": best_clusterer,
            "pca": pca,
            "features": list(numeric_df.columns),
        },
        METADATA_DIR / "cluster_model.pkl",
    )
    return df, cluster_report


def select_features(
    df: pd.DataFrame,
    target: pd.Series,
    mi_top_k: int = 20,
    corr_threshold: float = 0.9,
) -> FeatureSelectionReport:
    """Feature selection: domain filtering (done) -> MI -> model-based importance."""
    dropped: Dict[str, str] = {}

    encoded, _, _ = _prepare_encoded(df)
    mi_scores = mutual_info_classif(encoded, target, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": encoded.columns, "mi_score": mi_scores}).sort_values(
        "mi_score", ascending=False
    )

    top_features = mi_df.head(min(mi_top_k, len(mi_df)))["feature"].tolist()
    corr_matrix = encoded[top_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    kept_after_corr = [f for f in top_features if f not in corr_drop]
    for col in corr_drop:
        dropped[col] = f"Dropped due to correlation > {corr_threshold:.2f}"

    X_train, X_val, y_train, y_val = train_test_split(
        encoded[kept_after_corr],
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    rf = RandomForestClassifier(
        n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    perm = permutation_importance(
        rf, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_df = pd.DataFrame(
        {"feature": X_val.columns, "importance_mean": perm.importances_mean}
    ).sort_values("importance_mean", ascending=False)

    selected = importance_df[importance_df["importance_mean"] > 0]["feature"].tolist()
    if not selected:
        selected = importance_df.head(15)["feature"].tolist()

    for col in encoded.columns:
        if col not in selected and col not in dropped:
            dropped[col] = "Dropped after model-based importance"

    return FeatureSelectionReport(
        selected_features=selected,
        dropped_features=dropped,
        statistical_scores=mi_df,
        importance_scores=importance_df,
    )


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical, numeric = _split_columns(df)
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def train_models(df: pd.DataFrame, target: pd.Series) -> TrainReport:
    X_train, X_temp, y_train, y_temp = train_test_split(
        df,
        target,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1).to_csv(
        METADATA_DIR / "test_split.csv", index=False
    )

    preprocessor = _build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
# =============================================================================
#         "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
# =============================================================================
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )

    metrics_rows = []
    trained: Dict[str, Pipeline] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        val_preds = pipeline.predict(X_val)
        metrics_rows.append({
            "model": name,
            "validation_f1": f1_score(y_val, val_preds, zero_division=0),
        })
        trained[name] = pipeline
        joblib.dump(pipeline, MODELS_DIR / f"{name.lower().replace(' ', '_')}.pkl")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("validation_f1", ascending=False)
    metrics_df.to_csv(METADATA_DIR / "model_validation_metrics.csv", index=False)

    return TrainReport(metrics=metrics_df, models=trained, selected_features=list(df.columns))


def run_pipeline(data_path: str) -> None:
    df = load_dataset(data_path)
    engineered, documentation = engineer_features(df)

    X = engineered.drop(columns=["Anomaly"])
    X, cluster_report = _add_clustering_features(X)
    y = engineered["Anomaly"].astype(int)

    selection_report = select_features(X, y)

    X_selected = X.copy()
    # Keep raw features used to build selected one-hot columns
    selected_raw = set()
    for feature in selection_report.selected_features:
        if feature in X.columns:
            selected_raw.add(feature)
        else:
            for col in X.columns:
                if feature.startswith(f"{col}_"):
                    selected_raw.add(col)

    if not selected_raw:
        selected_raw = set(X.columns)

    X_selected = X[sorted(selected_raw)]

    train_report = train_models(X_selected, y)

    # Save documentation
    pd.DataFrame(
        [
            {"feature": f, "rationale": documentation.feature_rationales.get(f, "")}
            for f in documentation.selected_features
        ]
    ).to_csv(METADATA_DIR / "feature_rationales.csv", index=False)

    pd.DataFrame(
        list(documentation.dropped_features.items()), columns=["feature", "reason"]
    ).to_csv(METADATA_DIR / "dropped_features.csv", index=False)

    selection_report.statistical_scores.to_csv(
        METADATA_DIR / "mutual_information_scores.csv", index=False
    )
    selection_report.importance_scores.to_csv(
        METADATA_DIR / "permutation_importance_scores.csv", index=False
    )

    pd.DataFrame({"selected_features": list(X_selected.columns)}).to_csv(
        METADATA_DIR / "selected_features.csv", index=False
    )

    if not cluster_report.empty:
        cluster_report.to_csv(METADATA_DIR / "cluster_report.csv", index=False)
    X.assign(Anomaly=y).to_csv(METADATA_DIR / "cluster_assignments.csv", index=False)

    train_report.metrics.to_csv(METADATA_DIR / "model_validation_metrics.csv", index=False)


if __name__ == "__main__":
    run_pipeline("data/logs_SPEDIA_annotated_en.csv")
