# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:20:17 2026

@author: rasheed
"""
import base64
from pathlib import Path
from typing import Dict, List, Any

import joblib
import pandas as pd
import streamlit as st

from config import (
    MODELS_DIR,
    METADATA_DIR,
    DEFAULT_DERIVED_PRESETS,
    FEATURE_FIELDS,
)


def load_css(css_file_path: str) -> None:
    css_path = Path(css_file_path)
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_data
def load_logo_data(logo_file_path: str = "assets/image/shadow1.png") -> str:
    logo_path = Path(logo_file_path)
    if not logo_path.exists():
        return ""
    return base64.b64encode(logo_path.read_bytes()).decode("utf-8")


def get_available_models() -> List[Path]:
    return sorted(MODELS_DIR.glob("*.pkl"))


@st.cache_resource
def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


@st.cache_data
def load_selected_features() -> List[str]:
    selected_path = METADATA_DIR / "selected_features.csv"
    if selected_path.exists():
        return pd.read_csv(selected_path)["selected_features"].tolist()
    return []


@st.cache_data
def build_data_presets() -> Dict[str, Dict[str, float]]:
    test_split_path = METADATA_DIR / "test_split.csv"
    if not test_split_path.exists():
        return DEFAULT_DERIVED_PRESETS

    test_df = pd.read_csv(test_split_path)
    label_available = "Anomaly" in test_df.columns
    available = [c for c in FEATURE_FIELDS if c in test_df.columns]
    if not available:
        return DEFAULT_DERIVED_PRESETS

    def _row_from_df(df: pd.DataFrame, row_name: str = "50%") -> Dict[str, float]:
        stats = df[available].describe(percentiles=[0.5]).T
        return {
            col: float(stats.loc[col, row_name])
            if col in stats.index
            else DEFAULT_DERIVED_PRESETS["Normal"][col]
            for col in FEATURE_FIELDS
        }

    if label_available:
        normal_df = test_df[test_df["Anomaly"] == 0]
        high_df = test_df[test_df["Anomaly"] == 1]
        if not normal_df.empty and not high_df.empty:
            return {
                "Normal": _row_from_df(normal_df),
                "Medium": _row_from_df(test_df),
                "High": _row_from_df(high_df),
            }

    stats = test_df[available].describe(percentiles=[0.25, 0.5, 0.75]).T

    def _row(row_name: str) -> Dict[str, float]:
        return {
            col: float(stats.loc[col, row_name])
            if col in stats.index
            else DEFAULT_DERIVED_PRESETS["Normal"][col]
            for col in FEATURE_FIELDS
        }

    return {
        "Normal": _row("25%"),
        "Medium": _row("50%"),
        "High": _row("75%"),
    }


@st.cache_data
def load_reference_stats() -> Dict[str, pd.DataFrame]:
    stats = {}
    val_path = METADATA_DIR / "model_validation_metrics.csv"
    test_path = METADATA_DIR / "model_test_metrics.csv"
    if val_path.exists():
        stats["validation"] = pd.read_csv(val_path)
    if test_path.exists():
        stats["test"] = pd.read_csv(test_path)
    return stats


@st.cache_data
def load_importance_scores() -> pd.DataFrame:
    importance_path = METADATA_DIR / "permutation_importance_scores.csv"
    if importance_path.exists():
        return pd.read_csv(importance_path)
    return pd.DataFrame()


@st.cache_data
def load_cluster_report() -> pd.DataFrame:
    report_path = METADATA_DIR / "cluster_report.csv"
    if report_path.exists():
        return pd.read_csv(report_path)
    return pd.DataFrame()


@st.cache_data
def load_cluster_plot(path_name: str) -> pd.DataFrame:
    plot_path = METADATA_DIR / path_name
    if plot_path.exists():
        return pd.read_csv(plot_path)
    return pd.DataFrame()