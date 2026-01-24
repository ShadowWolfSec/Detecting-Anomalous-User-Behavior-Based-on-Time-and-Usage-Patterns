# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 13:40:45 2026

@author: rasheed
"""
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st

from config import DERIVED_HELP, PRESET_KEYS, EXAMPLES
from data_loader import (
    load_logo_data,
    load_importance_scores,
    load_reference_stats,
    load_cluster_report,
    load_cluster_plot,
    load_selected_features,
)
from scoring_service import (
    normalize_importance,
    prepare_metric_chart_df,
    score_batch_dataframe,
)


def render_login() -> None:
    logo_data = load_logo_data()
    
    st.markdown('<div class="shadow-login-page">', unsafe_allow_html=True)
    
    login_html = f"""
        <div class="login-wrapper-box">
            <div class="login-header-wrapper">
                <img src="data:image/png;base64,{logo_data}" class="login-hero-img" alt="Shadow Wolf Security" />
                <div class="login-sub">Secure Access • UBA Console</div>
            </div>
        </div>
        """
    st.markdown(login_html, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if username == "admin" and password == "admin":
                st.session_state["authenticated"] = True
                st.success("Login successful. جاري الدخول...")
                st.rerun()
            else:
                st.error("Invalid credentials.")
                
    st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar(
    models: List[Path], data_presets: Dict[str, Dict[str, float]]
) -> Tuple[str, Path, Dict[str, str]]:
    with st.sidebar:
        st.header("Configuration")
        input_mode = st.selectbox(
            "Input mode",
            ["Raw fields", "Derived features", "Both"],
        )
        st.markdown("**Quick presets**")
        preset_cols = st.columns(3)
        if preset_cols[0].button("Normal"):
            st.session_state["example_key"] = PRESET_KEYS["Normal"]
            st.session_state["derived_preset"] = data_presets["Normal"]
        if preset_cols[1].button("Medium"):
            st.session_state["example_key"] = PRESET_KEYS["Medium"]
            st.session_state["derived_preset"] = data_presets["Medium"]
        if preset_cols[2].button("High"):
            st.session_state["example_key"] = PRESET_KEYS["High"]
            st.session_state["derived_preset"] = data_presets["High"]

        model_path = st.selectbox(
            "Select model",
            models,
            format_func=lambda p: p.stem.replace("_", " ").title(),
        )
        example_keys_list = list(EXAMPLES.keys())
        current_example = st.session_state.get("example_key", example_keys_list[0])
        example_key = st.radio(
            "Example scenarios",
            example_keys_list,
            index=example_keys_list.index(current_example),
        )
        st.markdown("---")
        st.markdown("**Tips**")
        st.markdown("- Use realistic timestamps (no future dates)")
        st.markdown("- Keep fields aligned with dataset semantics")

    example = EXAMPLES[st.session_state.get("example_key", example_key)]
    return input_mode, model_path, example


def render_input_form(
    input_mode: str, example: Dict[str, str]
) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    st.subheader("Input event")
    raw_inputs: Dict[str, Any] = {}
    derived_inputs: Dict[str, Any] = {}

    with st.form("event_form"):
        if input_mode in {"Raw fields", "Both"}:
            col1, col2, col3 = st.columns(3)
            with col1:
                raw_inputs["user"] = st.text_input("User", value=example["user"])
                raw_inputs["activity"] = st.text_input("Activity", value=example["activity"])
            with col2:
                raw_inputs["action"] = st.text_input("Action", value=example["action"])
                raw_inputs["decoder_name"] = st.text_input("Decoder name", value=example["decoder_name"])
            with col3:
                levels = ["info", "low", "medium", "high", "critical", "error"]
                raw_inputs["level"] = st.selectbox(
                    "Level", levels, index=levels.index(example["level"])
                )

            raw_inputs["event_date"] = st.date_input("Event date", value=date.today())
            default_time = datetime.strptime(example["hour"], "%H:%M").time()
            raw_inputs["event_time"] = st.time_input("Event time", value=default_time)

        if input_mode in {"Derived features", "Both"}:
            st.markdown("**Derived feature inputs**")
            importance_df = load_importance_scores()
            if not importance_df.empty:
                top_features = (
                    importance_df.sort_values("importance_mean", ascending=False)
                    .head(8)["feature"]
                    .tolist()
                )
                st.info("Most influential features: " + ", ".join(top_features))
            st.caption("Use realistic ranges. Values outside these ranges may be clipped during training.")
            
            c1, c2, c3 = st.columns(3)
            preset_values = st.session_state.get("derived_preset", {})
            with c1:
                derived_inputs["hour"] = st.number_input(
                    "hour", min_value=0, max_value=23, value=int(preset_values.get("hour", 10)), help=DERIVED_HELP["hour"]
                )
                derived_inputs["weekday"] = st.number_input(
                    "weekday", min_value=0, max_value=6, value=int(preset_values.get("weekday", 1)), help=DERIVED_HELP["weekday"]
                )
                derived_inputs["is_weekend"] = st.number_input(
                    "is_weekend", min_value=0, max_value=1, value=int(preset_values.get("is_weekend", 0)), help=DERIVED_HELP["is_weekend"]
                )
                derived_inputs["is_off_hours"] = st.number_input(
                    "is_off_hours", min_value=0, max_value=1, value=int(preset_values.get("is_off_hours", 0)), help=DERIVED_HELP["is_off_hours"]
                )
                derived_inputs["is_night"] = st.number_input(
                    "is_night", min_value=0, max_value=1, value=int(preset_values.get("is_night", 0)), help=DERIVED_HELP["is_night"]
                )
                derived_inputs["user_event_index"] = st.number_input(
                    "user_event_index", min_value=0, max_value=100000, value=int(preset_values.get("user_event_index", 5)), help=DERIVED_HELP["user_event_index"]
                )
            with c2:
                derived_inputs["time_since_last_event"] = st.number_input(
                    "time_since_last_event", min_value=0.0, max_value=720.0, value=float(preset_values.get("time_since_last_event", 1.2)), help=DERIVED_HELP["time_since_last_event"]
                )
                derived_inputs["days_since_first_activity"] = st.number_input(
                    "days_since_first_activity", min_value=0.0, max_value=3650.0, value=float(preset_values.get("days_since_first_activity", 30.0)), help=DERIVED_HELP["days_since_first_activity"]
                )
                derived_inputs["activity_rate"] = st.number_input(
                    "activity_rate", min_value=0.0, max_value=1000.0, value=float(preset_values.get("activity_rate", 0.3)), help=DERIVED_HELP["activity_rate"]
                )
                derived_inputs["is_new_user"] = st.number_input(
                    "is_new_user", min_value=0, max_value=1, value=int(preset_values.get("is_new_user", 0)), help=DERIVED_HELP["is_new_user"]
                )
                derived_inputs["is_first_activity"] = st.number_input(
                    "is_first_activity", min_value=0, max_value=1, value=int(preset_values.get("is_first_activity", 0)), help=DERIVED_HELP["is_first_activity"]
                )
            with c3:
                derived_inputs["daily_event_index"] = st.number_input(
                    "daily_event_index", min_value=0, max_value=10000, value=int(preset_values.get("daily_event_index", 2)), help=DERIVED_HELP["daily_event_index"]
                )
                derived_inputs["hours_since_first_today"] = st.number_input(
                    "hours_since_first_today", min_value=0.0, max_value=24.0, value=float(preset_values.get("hours_since_first_today", 1.5)), help=DERIVED_HELP["hours_since_first_today"]
                )
                derived_inputs["daily_activity_rate"] = st.number_input(
                    "daily_activity_rate", min_value=0.0, max_value=1000.0, value=float(preset_values.get("daily_activity_rate", 0.6)), help=DERIVED_HELP["daily_activity_rate"]
                )
                derived_inputs["activity_is_missing"] = st.number_input(
                    "activity_is_missing", min_value=0, max_value=1, value=int(preset_values.get("activity_is_missing", 0)), help=DERIVED_HELP["activity_is_missing"]
                )
                derived_inputs["action_is_missing"] = st.number_input(
                    "action_is_missing", min_value=0, max_value=1, value=int(preset_values.get("action_is_missing", 0)), help=DERIVED_HELP["action_is_missing"]
                )
                derived_inputs["decoder_is_missing"] = st.number_input(
                    "decoder_is_missing", min_value=0, max_value=1, value=int(preset_values.get("decoder_is_missing", 0)), help=DERIVED_HELP["decoder_is_missing"]
                )
                derived_inputs["level_is_high"] = st.number_input(
                    "level_is_high", min_value=0, max_value=1, value=int(preset_values.get("level_is_high", 0)), help=DERIVED_HELP["level_is_high"]
                )

        submitted = st.form_submit_button("Score event")

    return submitted, raw_inputs, derived_inputs


def render_prediction_result(score: float, band: Dict[str, str], X: pd.DataFrame) -> None:
    """عرض نتيجة التنبؤ والتوصيات"""
    st.markdown("---")
    st.subheader("Result")
    st.metric("Anomaly Probability", f"{score * 100:.1f}%")
    st.progress(min(max(score, 0.0), 1.0))
    st.write(f"**Risk Level:** {band['label']}")
    st.write(f"**Recommendation:** {band['advice']}")

    with st.expander("قرار 0/1 حسب العتبة (اختياري)", expanded=False):
        threshold = st.slider("Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        decision = 1 if score >= threshold else 0
        label = "شاذ" if decision == 1 else "طبيعي"
        st.write(f"**Decision (0/1):** {decision} ({label})")
        st.caption("هذا القرار اختياري ولا يغيّر تصنيف Low/Medium/High الرئيسي.")

    st.subheader("Recommendations")
    if band["label"] == "High":
        st.markdown(
            "- Validate account ownership and recent credential changes\n"
            "- Review off-hours activity and privileged actions\n"
            "- Trigger incident response workflow if corroborated"
        )
    elif band["label"] == "Medium":
        st.markdown(
            "- Cross-check with recent user tickets and approvals\n"
            "- Increase alerting thresholds for this user\n"
            "- Monitor for recurring off-hours access"
        )
    else:
        st.markdown(
            "- Keep baseline monitoring\n"
            "- Maintain audit trail for future comparison"
        )

    st.subheader("Engineered feature values (this event)")
    st.dataframe(X, use_container_width=True)


def render_analytics_expanders() -> None:
    """(Feature importance, Performance, Clustering)"""
    with st.expander("Feature importance & impact", expanded=False):
        st.subheader("Most important features (global)")
        importance_df = load_importance_scores()
        if not importance_df.empty:
            top_k = st.slider("Top K features", min_value=5, max_value=30, value=15)
            top_df = importance_df.sort_values("importance_mean", ascending=False).head(top_k)
            st.dataframe(top_df, use_container_width=True)
            st.bar_chart(top_df.set_index("feature")["importance_mean"])
        else:
            st.info("Permutation importance not found. Run model_training.py first.")

        st.subheader("Feature impact report (percent contribution)")
        impact_df = normalize_importance(importance_df)
        if not impact_df.empty:
            top_impact = impact_df.head(10)[["feature", "impact_percent"]]
            st.dataframe(top_impact, use_container_width=True)
            st.bar_chart(top_impact.set_index("feature")["impact_percent"])
        else:
            st.info("No impact report available yet.")

    with st.expander("Model performance & statistics", expanded=False):
        st.subheader("Reference statistics")
        stats = load_reference_stats()
        if stats:
            if "validation" in stats:
                st.write("Validation metrics")
                st.dataframe(stats["validation"], use_container_width=True)
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.caption("Validation F1")
                    st.bar_chart(prepare_metric_chart_df(stats["validation"], "validation_f1"))
                with col_v2:
                    if "roc_auc" in stats["validation"].columns:
                        st.caption("Validation ROC-AUC")
                        st.bar_chart(prepare_metric_chart_df(stats["validation"], "roc_auc"))
            if "test" in stats:
                st.write("Test metrics")
                st.dataframe(stats["test"], use_container_width=True)
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.caption("Test F1")
                    st.bar_chart(prepare_metric_chart_df(stats["test"], "f1_score"))
                with col_t2:
                    st.caption("Test ROC-AUC")
                    st.bar_chart(prepare_metric_chart_df(stats["test"], "roc_auc"))
            
            st.subheader("Per-model performance summary")
            summary_df = stats.get("test") if "test" in stats else stats.get("validation")
            if summary_df is not None and not summary_df.empty:
                best_row = summary_df.sort_values(
                    summary_df.columns[summary_df.columns.str.contains("f1")][0],
                    ascending=False,
                ).head(1)
                st.write("Best model based on F1")
                st.dataframe(best_row, use_container_width=True)
        else:
            st.info("No reference metrics found yet. Run model_training.py and evaluation.py.")

    with st.expander("Clustering report & plots", expanded=False):
        cluster_report = load_cluster_report()
        if not cluster_report.empty:
            st.subheader("Cluster Report")
            st.dataframe(cluster_report, use_container_width=True)
        else:
            st.info("Cluster report not found. Run model_training.py first.")

        pca_df = load_cluster_plot("cluster_pca.csv")
        if not pca_df.empty and {"pca_1", "pca_2", "Cluster_ID"}.issubset(pca_df.columns):
            st.subheader("PCA Projection")
            st.scatter_chart(pca_df, x="pca_1", y="pca_2", color="Cluster_ID")
        else:
            st.info("PCA plot not available.")

        tsne_df = load_cluster_plot("cluster_tsne.csv")
        if not tsne_df.empty and {"tsne_1", "tsne_2", "Cluster_ID"}.issubset(tsne_df.columns):
            st.subheader("t-SNE Projection")
            st.scatter_chart(tsne_df, x="tsne_1", y="tsne_2", color="Cluster_ID")
        else:
            st.info("t-SNE plot not available.")


def render_batch_scanner(model_path: Path) -> None:
    """عرض قسم فحص ملفات الـ CSV الدفعية"""
    with st.expander("Batch file scan (CSV)", expanded=False):
        st.write("Upload a CSV file to score all rows and download النتائج.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            try:
                results = score_batch_dataframe(batch_df, model_path)
                st.subheader("Batch summary")
                st.dataframe(
                    results["risk_level"].value_counts().rename_axis("risk_level").to_frame("count")
                )
                st.dataframe(results.head(50), use_container_width=True)

                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download النتائج (CSV)",
                    data=csv_bytes,
                    file_name="ml5_batch_results.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Feature engineering or batch scoring failed: {exc}")


def render_footer() -> None:
    """ والتذييل السفلي"""
    st.subheader("Selected features")
    selected_features = load_selected_features()
    if selected_features:
        st.write(", ".join(selected_features))
    else:
        st.info("Selected features file not found.")

    st.markdown("---")
    st.caption("Developed by RASHEED AL-RAJEHI · © Shadow Wolf Security")