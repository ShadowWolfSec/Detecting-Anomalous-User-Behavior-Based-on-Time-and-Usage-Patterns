# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:01:08 2026

@author: rasheed
"""
from __future__ import annotations

import streamlit as st

from data_loader import load_css, get_available_models, build_data_presets
from scoring_service import score_single_event
from ui_components import (
    render_login,
    render_sidebar,
    render_input_form,
    render_prediction_result,
    render_analytics_expanders,
    render_batch_scanner,
    render_footer,
)


def main() -> None:
    st.set_page_config(page_title="UBA Test UI", layout="wide")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        load_css("assets/css/login_styles.css")
        render_login()
        return

    load_css("assets/css/styles.css")

    st.title("User Behavior Analytics – Anomaly Detection (Testing UI)")
    st.caption("Inference-only interface for behavioral anomaly scoring.")

    models = get_available_models()
    if not models:
        st.error("No trained models found in artifacts/models. Run model_training.py first.")
        st.stop()

    data_presets = build_data_presets()
    input_mode, model_path, example = render_sidebar(models, data_presets)

    submitted, raw_inputs, derived_inputs = render_input_form(input_mode, example)

    if submitted:
        score, band, X = score_single_event(
            input_mode, raw_inputs, derived_inputs, model_path
        )
        render_prediction_result(score, band, X)

    # الأقسام التحليلية السفلية وفحص الملفات
    st.markdown("---")
    render_analytics_expanders()
    render_batch_scanner(model_path)
    render_footer()


if __name__ == "__main__":
    main()
