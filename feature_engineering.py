# -*- coding: utf-8 -*-
"""

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

RAW_TEXT_COLUMNS = [
    "Full_log",
    "Description",
    "Content",
    "Url",
    "Path",
    "Command",
    "Filename",
]

IDENTIFIER_COLUMNS = ["id", "User", "Agent_name"]


@dataclass
class FeatureDocumentation:
    selected_features: List[str]
    dropped_features: Dict[str, str]
    feature_rationales: Dict[str, str]


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    required = ["Timestamp", "User", "Anomaly"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _safe_str(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _winsorize(series: pd.Series, upper_q: float = 0.99) -> pd.Series:
    """Clip extreme upper tail to reduce outlier impact (no target leakage)."""
    if series.empty:
        return series
    upper = series.quantile(upper_q)
    return series.clip(lower=0, upper=upper)


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureDocumentation]:
    """Engineer temporal + behavioral features without leakage."""
    df = df.copy()

    # Strict temporal sorting
    df["Timestamp"] = _safe_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Drop raw text columns (prohibited)
    dropped: Dict[str, str] = {}
    for col in RAW_TEXT_COLUMNS:
        if col in df.columns:
            dropped[col] = "Raw text prohibited; high leakage/PII risk"
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            dropped[col] = "Identifier not allowed as feature"

    # Temporal features
    df["hour"] = df["Timestamp"].dt.hour
    df["weekday"] = df["Timestamp"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_off_hours"] = df["hour"].apply(lambda x: 1 if (x < 7 or x > 19) else 0)
    df["is_night"] = df["hour"].apply(lambda x: 1 if (x >= 22 or x <= 5) else 0)

    # User behavioral features (past-only)
    df["User"] = df["User"].fillna("unknown_user")
    df = df.sort_values(["User", "Timestamp"]).reset_index(drop=True)

    df["user_event_index"] = df.groupby("User").cumcount()
    df["time_since_last_event"] = (
        df.groupby("User")["Timestamp"].diff().dt.total_seconds().div(3600)
    )
    df["time_since_last_event"] = df["time_since_last_event"].fillna(0)

    df["first_activity_time"] = df.groupby("User")["Timestamp"].transform("first")
    df["days_since_first_activity"] = (
        (df["Timestamp"] - df["first_activity_time"]).dt.total_seconds().div(86400)
    )
    df["activity_rate"] = df["user_event_index"] / (df["days_since_first_activity"] + 1)

    # Reduce extreme skew for temporal/rate features (winsorization)
    df["time_since_last_event"] = _winsorize(df["time_since_last_event"])
    df["days_since_first_activity"] = _winsorize(df["days_since_first_activity"])
    df["activity_rate"] = _winsorize(df["activity_rate"])

    df["is_new_user"] = (df["days_since_first_activity"] < 7).astype(int)
    df["is_first_activity"] = (df["user_event_index"] == 0).astype(int)

    # Daily behavior (past-only)
    df["date"] = df["Timestamp"].dt.date
    df["daily_event_index"] = df.groupby(["User", "date"]).cumcount()
    df["first_event_today"] = df.groupby(["User", "date"])["Timestamp"].transform("first")
    df["hours_since_first_today"] = (
        (df["Timestamp"] - df["first_event_today"]).dt.total_seconds().div(3600)
    )
    df["daily_activity_rate"] = df["daily_event_index"] / (df["hours_since_first_today"] + 0.1)
    df["hours_since_first_today"] = _winsorize(df["hours_since_first_today"])
    df["daily_activity_rate"] = _winsorize(df["daily_activity_rate"])

    # Lightweight behavioral proxies from allowed columns
    df["activity"] = _safe_str(df.get("Activity"))
    df["action"] = _safe_str(df.get("Action"))
    df["decoder_name"] = _safe_str(df.get("Decoder_name"))
    df["level"] = _safe_str(df.get("Level"))

    df["activity_is_missing"] = (df["activity"] == "").astype(int)
    df["action_is_missing"] = (df["action"] == "").astype(int)
    df["decoder_is_missing"] = (df["decoder_name"] == "").astype(int)
    df["level_is_high"] = df["level"].str.lower().isin(["high", "critical", "error"]).astype(int)

    # Feature list (domain-filtered, UBA-focused)
    features = [
        # سلوك المستخدم عبر الزمن
        "hour",
        "weekday",
        "is_weekend",
        "is_off_hours",
        "is_night",
        "user_event_index",
        "time_since_last_event",
        "days_since_first_activity",
        "activity_rate",
        "is_new_user",
        "is_first_activity",
    #    سلوك المستخدم اليومي
        "daily_event_index",
        "hours_since_first_today",
        "daily_activity_rate",
        "activity_is_missing",
        # إشارات غير مباشرة (ذكية)
        "action_is_missing",
        "decoder_is_missing",
        "level_is_high",
    ]

# =============================================================================
#     feature_rationales = {
#         "hour": "Behavior deviates at specific hours (temporal usage patterns)",
#         "weekday": "Weekly routine changes can signal abnormal usage",
#         "is_weekend": "Weekend access may indicate abnormal or risky behavior",
#         "is_off_hours": "Off-hours usage is a common insider risk indicator",
#         "is_night": "Night-time usage suggests deviation from normal schedules",
#         "user_event_index": "Cumulative event position captures user activity volume",
#         "time_since_last_event": "Short gaps can imply bursts or automation",
#         "days_since_first_activity": "Newer accounts often exhibit higher risk",
#         "activity_rate": "Long-term activity rate shows behavioral baselines",
#         "is_new_user": "Cold-start behavior can differ from established users",
#         "is_first_activity": "First events may have higher uncertainty",
#         "daily_event_index": "Within-day activity growth highlights bursts",
#         "hours_since_first_today": "Daily time progression contextualizes activity density",
#         "daily_activity_rate": "High daily rate indicates possible abuse",
#         "activity_is_missing": "Missing activity may reflect log anomalies",
#         "action_is_missing": "Missing action can indicate unusual logging behavior",
#         "decoder_is_missing": "Missing decoder may signal uncommon event types",
#         "level_is_high": "High severity levels align with risk escalation",
#     }
# =============================================================================
    feature_rationales = {
        
           "hour": "سلوك المستخدم يختلف حسب الساعة، والانحراف عن ساعات العمل الطبيعية قد يدل على نشاط غير معتاد",
            
            "weekday": "التغير في نمط الاستخدام خلال أيام الأسبوع قد يشير إلى سلوك غير طبيعي أو مشبوه",
            
            "is_weekend": "الدخول أو النشاط خلال عطلة نهاية الأسبوع قد يكون مؤشرًا على سلوك عالي الخطورة",
            
            "is_off_hours": "النشاط خارج أوقات الدوام الرسمي يُعد من أهم مؤشرات التهديدات الداخلية",
            
            "is_night": "النشاط الليلي غالبًا ما يرتبط بمحاولات إخفاء أو سلوك غير معتاد",
            
            "user_event_index": "يعكس حجم النشاط التراكمي للمستخدم، والزيادة غير الطبيعية قد تشير إلى إساءة استخدام",
            
            "time_since_last_event": "الفواصل الزمنية القصيرة جدًا قد تدل على نشاط آلي أو اندفاع غير طبيعي",
            
            "days_since_first_activity": "الحسابات الجديدة غالبًا ما تكون أكثر عرضة للسلوك الخطر أو إساءة الاستخدام",
            
            "activity_rate": "معدل النشاط العام يساعد في بناء خط الأساس السلوكي للمستخدم",
            
            "is_new_user": "سلوك المستخدم الجديد يختلف عن المستخدمين المستقرين وقد يحمل مخاطر أعلى",
            
            "is_first_activity": "النشاط الأول للمستخدم غالبًا ما يكون غير مستقر أو غير متوقع",
            
            "daily_event_index": "زيادة عدد الأنشطة خلال اليوم الواحد قد تشير إلى تصعيد في السلوك",
            
            "hours_since_first_today": "يساعد في فهم توزع النشاط خلال اليوم وليس فقط عدده",
            
            "daily_activity_rate": "المعدل اليومي المرتفع للنشاط قد يدل على إساءة استخدام أو هجوم داخلي",
            
            "activity_is_missing": "غياب معلومات النشاط قد يشير إلى خلل أو محاولة إخفاء السلوك الحقيقي",
            
            "action_is_missing": "عدم تسجيل نوع الإجراء قد يدل على سلوك غير معتاد أو مشاكل في السجلات",
            
            "decoder_is_missing": "غياب المُفسِّر قد يشير إلى نوع حدث غير شائع أو مشبوه",
            
            "level_is_high": "مستويات الخطورة العالية غالبًا ما ترتبط بمحاولات تصعيد أو أنشطة ضارة",

    }

# =============================================================================
#  
#     dropped.update({
#         "Timestamp": "Used only for temporal feature derivation",
#         "User": "Identifier removed; used for grouping only",
#         "Agent_name": "Identifier removed; not behavioral",
#         "id": "Identifier removed; not behavioral",
#     })
# =============================================================================
    dropped.update({
    "Timestamp": "تم استخدامه فقط لاشتقاق الخصائص الزمنية (مثل الساعة ويوم الأسبوع)، ولا يُستخدم مباشرة كمدخل للنموذج",
    "User": "مُعرّف مستخدم تم استبعاده من الخصائص؛ استُخدم فقط للتجميع وتحليل السلوك عبر الزمن",
    "Agent_name": "مُعرّف نظام/برنامج تم استبعاده لأنه لا يعبّر عن سلوك المستخدم",
    "id": "مُعرّف سجل فريد تم استبعاده لعدم احتوائه على أي دلالة سلوكية",
    })


    for col in RAW_TEXT_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only engineered features + target
    X = df[features].copy()
    y = df["Anomaly"].astype(int).copy()

    documentation = FeatureDocumentation(
        selected_features=features,
        dropped_features=dropped,
        feature_rationales=feature_rationales,
    )
    return pd.concat([X, y], axis=1), documentation
