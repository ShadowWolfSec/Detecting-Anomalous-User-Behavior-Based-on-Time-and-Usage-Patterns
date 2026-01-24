# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:01:08 2026

@author: rasheed
"""

import pandas as pd


REQUIRED_COLUMNS = ["Timestamp", "User", "Anomaly"]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load dataset and validate required columns.
    """
    df = pd.read_csv(csv_path, low_memory=False)
# =============================================================================
#     low_memory=False؟ # يمنع مشاكل:
# 
# اختلاف أنواع البيانات # أخطاء قراءة الملفات الكبيرة # يجعل القراءة أكثر أمانًا
# =============================================================================

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def explore_dataset(df: pd.DataFrame) -> None:
    """
    Lightweight exploration for understanding the dataset.
    """
    print("=" * 60)
    print("📊 Dataset Overview")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing Values (Top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))

    print("\nTarget Distribution (Anomaly):")
    print(df["Anomaly"].value_counts(normalize=True))


if __name__ == "__main__":
    df = load_dataset("data/logs_SPEDIA_annotated_en.csv")
    explore_dataset(df)
