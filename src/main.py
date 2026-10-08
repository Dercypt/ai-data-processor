import json
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union

from analyzer import (
    analyze_dataset,
    detect_column_types,
    generate_summary,
    impute_missing_values,
)
from llm_service import get_ai_insights
from database import init_db, save_entry, get_all_entries, delete_entry

# 1. Initialize DB on app startup
init_db()

st.set_page_config(page_title="AI Data Processor", page_icon="📊", layout="wide")
st.title("AI Data Processor")

# --- SESSION STATE INITIALIZATION ---
if "generated_insight" not in st.session_state:
    st.session_state["generated_insight"] = None


def render_sidebar_insights(insights: str) -> None:
    """Render structured insights or fallback text inside the sidebar expander."""
    try:
        data = json.loads(insights)
        if isinstance(data, dict):
            summary = data.get("summary")
            risks = data.get("risks")
            recommendations = data.get("recommendations")

            if summary:
                st.markdown(f"**Summary:** {summary}")
            if isinstance(risks, list) and risks:
                st.markdown("**Risks:**")
                for r in risks:
                    st.markdown(f"- {r}")
            if isinstance(recommendations, list) and recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            return
    except (json.JSONDecodeError, TypeError):
        pass
    st.info(insights)


# --- SIDEBAR: HISTORY & MANAGEMENT ---
st.sidebar.title("Analysis History")

# Fetch updated history
history = get_all_entries()

if not history:
    st.sidebar.text("No past analyses.")
else:
    for entry_id, timestamp, title, insights in history:
        label = f"{title} ({timestamp[11:16]})"
        
        with st.sidebar.expander(label):
            st.caption(f"Date: {timestamp[:10]}")
            render_sidebar_insights(str(insights))
            
            # The Delete Button
            if st.button("🗑️ Delete", key=f"del_{entry_id}"):
                delete_entry(entry_id)
                st.rerun()


# --- INTERACTIVE VISUALIZATION HELPERS ---

def render_distribution_chart(df: pd.DataFrame, column: str) -> None:
    """Render an interactive distribution chart (histogram + box plot) using Plotly."""
    fig = px.histogram(
        df,
        x=column,
        marginal="box",
        title=f"Distribution & Outlier Box Plot: {column}",
        color_discrete_sequence=["#1f77b4"],
        template="plotly_white",
    )
    fig.update_layout(bargap=0.1, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(corr_data: Dict[str, Dict[str, float]]) -> None:
    """Render an interactive correlation heatmap using Plotly."""
    if not corr_data:
        st.info("At least 2 numeric columns are required to generate a correlation heatmap.")
        return
    corr_df = pd.DataFrame(corr_data)
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Pearson Correlation Matrix",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_skewness_kurtosis_chart(
    skew_data: Dict[str, float],
    kurt_data: Dict[str, float],
    skew_desc: Dict[str, str],
    kurt_desc: Dict[str, str],
) -> None:
    """Render skewness and kurtosis diagnostics chart and data table."""
    if not skew_data:
        st.info("No numeric columns found for skewness/kurtosis calculation.")
        return

    cols = list(skew_data.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cols,
        y=[skew_data[c] for c in cols],
        name="Skewness",
        marker_color="#ff7f0e",
    ))
    fig.add_trace(go.Bar(
        x=cols,
        y=[kurt_data[c] for c in cols],
        name="Excess Kurtosis",
        marker_color="#2ca02c",
    ))
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Normal Baseline (0)")
    fig.update_layout(
        barmode="group",
        title="Skewness & Kurtosis per Numeric Feature",
        xaxis_title="Feature",
        yaxis_title="Value",
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    summary_rows = []
    for c in cols:
        summary_rows.append({
            "Feature": c,
            "Skewness": skew_data[c],
            "Skewness Interpretation": skew_desc.get(c, "N/A"),
            "Kurtosis": kurt_data[c],
            "Kurtosis Interpretation": kurt_desc.get(c, "N/A"),
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)


def render_outlier_inspector(
    df: pd.DataFrame,
    outliers_data: Dict[str, object],
) -> None:
    """Render interactive outlier detection dashboard with method selection (IQR, Z-Score, Isolation Forest)."""
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if not numeric_cols:
        st.info("No numeric columns available for outlier detection.")
        return

    method = st.radio(
        "Select Outlier Detection Method:",
        options=["IQR (Interquartile Range)", "Z-Score (|z| > 3)", "Isolation Forest (ML Anomaly Detection)"],
        horizontal=True,
    )

    if method.startswith("IQR"):
        iqr_data = outliers_data.get("iqr", {})  # type: ignore[union-attr]
        if not isinstance(iqr_data, dict) or not iqr_data:
            st.info("No IQR outlier data available.")
            return

        table_rows = []
        for c in numeric_cols:
            if c in iqr_data:
                info = iqr_data[c]
                table_rows.append({
                    "Feature": c,
                    "Outlier Count": info.get("count", 0),
                    "Percentage (%)": f"{info.get('percentage', 0.0)}%",
                    "Lower Bound": info.get("lower_bound", 0.0),
                    "Upper Bound": info.get("upper_bound", 0.0),
                })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        selected_col = st.selectbox("Inspect Outliers for Feature (IQR):", numeric_cols, key="iqr_col_select")
        col_info = iqr_data.get(selected_col, {})
        outlier_indices = set(col_info.get("outlier_indices", []))

        plot_df = df.copy()
        plot_df["Status"] = ["Outlier" if i in outlier_indices else "Normal" for i in plot_df.index]
        fig = px.strip(
            plot_df,
            y=selected_col,
            color="Status",
            color_discrete_map={"Normal": "#1f77b4", "Outlier": "#d62728"},
            title=f"IQR Outliers for '{selected_col}' (Bounds: [{col_info.get('lower_bound')}, {col_info.get('upper_bound')}])",
            template="plotly_white",
        )
        fig.add_hline(y=col_info.get("lower_bound", 0), line_dash="dash", line_color="red", annotation_text="Lower IQR Bound")
        fig.add_hline(y=col_info.get("upper_bound", 0), line_dash="dash", line_color="red", annotation_text="Upper IQR Bound")
        st.plotly_chart(fig, use_container_width=True)

    elif method.startswith("Z-Score"):
        zscore_data = outliers_data.get("zscore", {})  # type: ignore[union-attr]
        if not isinstance(zscore_data, dict) or not zscore_data:
            st.info("No Z-Score outlier data available.")
            return

        table_rows = []
        for c in numeric_cols:
            if c in zscore_data:
                info = zscore_data[c]
                table_rows.append({
                    "Feature": c,
                    "Outlier Count": info.get("count", 0),
                    "Percentage (%)": f"{info.get('percentage', 0.0)}%",
                    "Mean": info.get("mean", 0.0),
                    "Std Dev": info.get("std", 0.0),
                    "Threshold": f"|z| > {info.get('threshold', 3.0)}",
                })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        selected_col = st.selectbox("Inspect Outliers for Feature (Z-Score):", numeric_cols, key="zscore_col_select")
        col_info = zscore_data.get(selected_col, {})
        outlier_indices = set(col_info.get("outlier_indices", []))

        plot_df = df.copy()
        plot_df["Status"] = ["Outlier" if i in outlier_indices else "Normal" for i in plot_df.index]
        fig = px.scatter(
            plot_df,
            x=plot_df.index,
            y=selected_col,
            color="Status",
            color_discrete_map={"Normal": "#1f77b4", "Outlier": "#d62728"},
            title=f"Z-Score Outliers for '{selected_col}' (|z| > 3.0)",
            labels={"x": "Index", "y": selected_col},
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # Isolation Forest
        iso_data = outliers_data.get("isolation_forest", {})  # type: ignore[union-attr]
        count = iso_data.get("count", 0) if isinstance(iso_data, dict) else 0
        pct = iso_data.get("percentage", 0.0) if isinstance(iso_data, dict) else 0.0
        outlier_indices = set(iso_data.get("outlier_indices", [])) if isinstance(iso_data, dict) else set()

        st.metric(label="Total Multidimensional Anomalies Detected", value=count, delta=f"{pct}% of dataset")

        plot_df = df.copy()
        plot_df["Status"] = ["Anomaly" if i in outlier_indices else "Normal" for i in plot_df.index]

        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_axis = st.selectbox("X-Axis Feature:", numeric_cols, index=0, key="iso_x")
            with c2:
                y_axis = st.selectbox("Y-Axis Feature:", numeric_cols, index=1, key="iso_y")

            fig = px.scatter(
                plot_df,
                x=x_axis,
                y=y_axis,
                color="Status",
                color_discrete_map={"Normal": "#1f77b4", "Anomaly": "#d62728"},
                title=f"Isolation Forest Anomaly Map: {x_axis} vs {y_axis}",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                plot_df,
                x=plot_df.index,
                y=numeric_cols[0],
                color="Status",
                color_discrete_map={"Normal": "#1f77b4", "Anomaly": "#d62728"},
                title=f"Isolation Forest Anomalies for '{numeric_cols[0]}'",
                labels={"x": "Index", "y": numeric_cols[0]},
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_preprocessing_pipeline(
    raw_df: pd.DataFrame,
    detected_types: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Render interactive Smart Preprocessing Pipeline controls:
    - Auto-detected column types (continuous, categorical, datetime, text)
    - User-configurable imputation (Mean, Median, Mode, Drop)
    """
    with st.expander("🛠️ Smart / Automated Preprocessing Pipeline", expanded=True):
        st.write("#### 🔍 Auto-Detected Column Types")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Continuous (🔢)", sum(1 for t in detected_types.values() if t == "continuous"))
        c2.metric("Categorical (🏷️)", sum(1 for t in detected_types.values() if t == "categorical"))
        c3.metric("Datetime (📅)", sum(1 for t in detected_types.values() if t == "datetime"))
        c4.metric("Text (📝)", sum(1 for t in detected_types.values() if t == "text"))

        total_rows = len(raw_df)
        type_rows = []
        for col in raw_df.columns:
            missing_cnt = int(raw_df[col].isnull().sum())
            missing_pct = round((missing_cnt / total_rows) * 100.0, 2) if total_rows > 0 else 0.0
            type_rows.append({
                "Feature": col,
                "Detected Type": detected_types.get(str(col), "unknown"),
                "Missing Values": missing_cnt,
                "Missing %": f"{missing_pct}%",
            })
        st.dataframe(pd.DataFrame(type_rows), use_container_width=True)

        missing_cols = [c for c in raw_df.columns if raw_df[c].isnull().any()]
        if not missing_cols:
            st.success("✅ Dataset has no missing values. No imputation required.")
            summary = generate_summary(raw_df)
            summary["column_types"] = detected_types
            return raw_df, summary

        st.markdown("#### ⚙️ User-Configurable Imputation")
        st.caption("Customize missing value imputation strategies across detected column types (Mean, Median, Mode, Drop).")

        mode = st.radio(
            "Strategy Mode:",
            options=["Global Strategy", "Per-Column Strategy"],
            horizontal=True,
            key="imputation_mode_radio",
        )

        processed_df = raw_df
        if mode == "Global Strategy":
            global_opt = st.selectbox(
                "Select Global Strategy (Mean, Median, Mode, Drop):",
                options=["Mean", "Median", "Mode", "Drop"],
                key="global_imputation_select",
            )
            try:
                processed_df = impute_missing_values(raw_df, strategy=global_opt.lower())
                st.info(f"Applied **{global_opt}** imputation globally across missing features.")
            except Exception as e:
                st.error(f"Imputation failed: {e}")
                processed_df = raw_df
        else:
            col_strategy_map: Dict[str, str] = {}
            st.write("Configure imputation per missing column:")
            cols_ui = st.columns(min(len(missing_cols), 3))
            for idx, col_name in enumerate(missing_cols):
                col_type = detected_types.get(col_name, "continuous")
                with cols_ui[idx % len(cols_ui)]:
                    if col_type == "continuous":
                        options = ["Mean", "Median", "Mode", "Drop"]
                    else:
                        options = ["Mode", "Drop"]
                    selected_strategy = st.selectbox(
                        f"`{col_name}` ({col_type}):",
                        options=options,
                        key=f"col_impute_{col_name}",
                    )
                    col_strategy_map[col_name] = selected_strategy.lower()

            try:
                processed_df = impute_missing_values(raw_df, strategy=col_strategy_map)
                st.info("Applied per-column imputation configuration.")
            except Exception as e:
                st.error(f"Imputation failed: {e}")
                processed_df = raw_df

        # Provide summary metrics after imputation
        initial_len = len(raw_df)
        final_len = len(processed_df)
        if final_len < initial_len:
            st.warning(f"⚠️ Row count reduced from {initial_len} to {final_len} due to row drop.")
        else:
            st.success(f"Row count preserved ({final_len} rows). All missing values resolved.")

        summary = generate_summary(processed_df)
        summary["column_types"] = detected_types
        return processed_df, summary


def render_structured_insights(insights: Union[Dict[str, object], str]) -> None:
    """Render structured AI insights (summary, risks, recommendations) with Streamlit components."""
    data: Dict[str, object] = {}
    if isinstance(insights, str):
        try:
            parsed = json.loads(insights)
            if isinstance(parsed, dict):
                data = parsed
            else:
                st.write(insights)
                return
        except (json.JSONDecodeError, TypeError):
            st.write(insights)
            return
    elif isinstance(insights, dict):
        data = insights
    else:
        st.write(insights)
        return

    summary = data.get("summary")
    risks = data.get("risks")
    recommendations = data.get("recommendations")

    if summary:
        st.markdown("#### 📋 Executive Summary")
        st.info(str(summary))

    if isinstance(risks, list) and risks:
        st.markdown("#### ⚠️ Key Risks & Anomalies")
        for risk in risks:
            st.warning(f"• {risk}")

    if isinstance(recommendations, list) and recommendations:
        st.markdown("#### 💡 Strategic Recommendations")
        for rec in recommendations:
            st.success(f"• {rec}")


# --- MAIN APP: UPLOAD & ANALYZE ---

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, object]] = None

    try:
        raw_df = pd.read_csv(file)
        if raw_df.empty:
            st.error("The uploaded CSV file is empty.")
        else:
            detected_types = detect_column_types(raw_df)
            df, summary = render_preprocessing_pipeline(raw_df, detected_types)
    except Exception as e:
        st.error(f"Failed to process CSV: {e}")

    if df is not None and summary is not None:
        st.write("### Data Preview", df.head())

        # UI Layout: Interactive Data Science & Statistics on left, AI Controls on right
        col1, col2 = st.columns([2.5, 1])

        with col1:
            st.write("### 📊 Statistical Data Science & Visual EDA")
            tab_summary, tab_corr, tab_skew, tab_outliers = st.tabs([
                "📋 Summary & Distribution",
                "🔥 Correlation Heatmap",
                "📈 Skewness & Kurtosis",
                "🎯 Outlier Detection",
            ])

            with tab_summary:
                st.write("#### Descriptive Statistics")
                stats_df = pd.DataFrame(summary.get("stats", {}))
                st.dataframe(stats_df, use_container_width=True)

                numeric_cols = list(df.select_dtypes(include=["number"]).columns)
                if numeric_cols:
                    selected_col = st.selectbox("Select Feature to Visualize Distribution:", numeric_cols)
                    render_distribution_chart(df, selected_col)

            with tab_corr:
                st.write("#### Correlation Analysis")
                render_correlation_heatmap(summary.get("correlation", {}))  # type: ignore[arg-type]

            with tab_skew:
                st.write("#### Skewness & Kurtosis Testing")
                render_skewness_kurtosis_chart(
                    summary.get("skewness", {}),  # type: ignore[arg-type]
                    summary.get("kurtosis", {}),  # type: ignore[arg-type]
                    summary.get("skewness_interpretation", {}),  # type: ignore[arg-type]
                    summary.get("kurtosis_interpretation", {}),  # type: ignore[arg-type]
                )

            with tab_outliers:
                st.write("#### Statistical Outlier Detection")
                render_outlier_inspector(df, summary.get("outliers", {}))  # type: ignore[arg-type]

        with col2:
            st.write("### AI Analysis Controls")

            custom_title = st.text_input("Name this analysis", value=getattr(file, "name", "dataset.csv"))

            if st.button("Generate Insights"):
                with st.spinner("Consulting the AI Analyst..."):
                    # Only aggregated statistics sent to LLM service (strictly adhering to Law 1)
                    ai_payload = {
                        "columns": summary.get("columns", []),
                        "rows": summary.get("rows", 0),
                        "stats": summary.get("stats", {}),
                        "skewness": summary.get("skewness", {}),
                        "kurtosis": summary.get("kurtosis", {}),
                        "correlation": summary.get("correlation", {}),
                    }
                    try:
                        insights = get_ai_insights(ai_payload)
                        st.session_state["generated_insight"] = insights

                        save_entry(custom_title, insights)
                        st.success("Saved to History!")

                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate structured insights: {e}")

    # --- DISPLAY RESULTS (Outside the button) ---
    if st.session_state["generated_insight"]:
        st.divider()
        st.write("### 🤖 Generated Insights")
        render_structured_insights(st.session_state["generated_insight"])