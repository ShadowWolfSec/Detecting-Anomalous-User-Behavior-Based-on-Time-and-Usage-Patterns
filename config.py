# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:01:08 2026

@author: rasheed
"""
from pathlib import Path
from typing import Dict

ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

RISK_BANDS = [
    (0.8, "High", "Immediate investigation recommended"),
    (0.5, "Medium", "Increase monitoring and verify context"),
    (0.0, "Low", "Normal behavior; continue routine monitoring"),
]

DERIVED_HELP = {
    "hour": "0-23: ساعة الحدث",
    "weekday": "0-6: الإثنين=0 ... السبت=5 ... الأحد=6",
    "is_weekend": "0/1: 1 إذا كان عطلة نهاية الأسبوع",
    "is_off_hours": "0/1: 1 إذا كان خارج 07:00-19:00",
    "is_night": "0/1: 1 إذا كان بين 22:00-05:00",
    "user_event_index": ">=0: عدد أحداث المستخدم حتى الآن",
    "time_since_last_event": "ساعات منذ آخر حدث لنفس المستخدم",
    "days_since_first_activity": "أيام منذ أول نشاط للمستخدم",
    "activity_rate": "معدل الأحداث في اليوم منذ أول نشاط",
    "is_new_user": "0/1: 1 إذا كان عمر الحساب أقل من 7 أيام",
    "is_first_activity": "0/1: 1 إذا كان هذا أول حدث للمستخدم",
    "daily_event_index": ">=0: عدد أحداث المستخدم اليوم",
    "hours_since_first_today": "ساعات منذ أول حدث للمستخدم اليوم",
    "daily_activity_rate": "معدل الأحداث لكل ساعة اليوم",
    "activity_is_missing": "0/1: حقل Activity مفقود",
    "action_is_missing": "0/1: حقل Action مفقود",
    "decoder_is_missing": "0/1: حقل Decoder name مفقود",
    "level_is_high": "0/1: المستوى High/Critical/Error",
}

PRESET_KEYS = {
    "Normal": "Normal office activity",
    "Medium": "Off-hours burst",
    "High": "Weekend admin change",
}

DERIVED_PRESETS = {
    "Normal": {
        "hour": 10,
        "weekday": 1,
        "is_weekend": 0,
        "is_off_hours": 0,
        "is_night": 0,
        "user_event_index": 12,
        "time_since_last_event": 2.0,
        "days_since_first_activity": 120.0,
        "activity_rate": 0.2,
        "is_new_user": 0,
        "is_first_activity": 0,
        "daily_event_index": 3,
        "hours_since_first_today": 3.0,
        "daily_activity_rate": 0.7,
        "activity_is_missing": 0,
        "action_is_missing": 0,
        "decoder_is_missing": 0,
        "level_is_high": 0,
    },
    "Medium": {
        "hour": 2,
        "weekday": 2,
        "is_weekend": 0,
        "is_off_hours": 1,
        "is_night": 1,
        "user_event_index": 30,
        "time_since_last_event": 0.3,
        "days_since_first_activity": 40.0,
        "activity_rate": 0.8,
        "is_new_user": 0,
        "is_first_activity": 0,
        "daily_event_index": 9,
        "hours_since_first_today": 1.0,
        "daily_activity_rate": 2.0,
        "activity_is_missing": 0,
        "action_is_missing": 0,
        "decoder_is_missing": 0,
        "level_is_high": 1,
    },
    "High": {
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
    },
}

DEFAULT_DERIVED_PRESETS = DERIVED_PRESETS.copy()

FEATURE_FIELDS = [
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
    "daily_event_index",
    "hours_since_first_today",
    "daily_activity_rate",
    "activity_is_missing",
    "action_is_missing",
    "decoder_is_missing",
    "level_is_high",
]

EXAMPLES: Dict[str, Dict[str, str]] = {
    "Normal office activity": {
        "user": "user_014",
        "activity": "file_access",
        "action": "read",
        "decoder_name": "syslog",
        "level": "info",
        "hour": "10:15",
    },
    "Off-hours burst": {
        "user": "user_091",
        "activity": "authentication",
        "action": "login_failed",
        "decoder_name": "authd",
        "level": "high",
        "hour": "02:40",
    },
    "Weekend admin change": {
        "user": "admin_002",
        "activity": "privilege_change",
        "action": "add_user",
        "decoder_name": "sudo",
        "level": "critical",
        "hour": "14:25",
    },
}