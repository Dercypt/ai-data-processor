import json
import streamlit as st
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple, Union

from analyzer import (
    DEFAULT_CHUNKSIZE,
    analyze_dataset,
    benchmark_imputation_drift,
    build_feature_dependency_graph,
    detect_column_types,
    evaluate_heuristic_rules,
    evaluate_memory_footprint,
    generate_summary,
    impute_missing_values,
    read_csv_chunked,
)
from llm_service import get_ai_insights
from database import (
    init_db,
    save_entry,
    get_all_entries,
    delete_entry,
    create_user,
    authenticate_user,
    UserAlreadyExistsError,
    AuthenticationError,
)

# 1. Initialize DB on app startup
init_db()

st.set_page_config(page_title="AI Data Processor", page_icon="📊", layout="wide")
st.title("AI Data Processor")

# --- SESSION STATE INITIALIZATION ---
if "generated_insight" not in st.session_state:
    st.session_state["generated_insight"] = None
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None


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


# --- SIDEBAR: AUTHENTICATION & WORKSPACE ---
st.sidebar.title("Analyst Workspace")

if st.session_state["current_user"] is None:
    st.sidebar.subheader("🔐 Analyst Login")
    auth_action = st.sidebar.radio("Account Action", ["Login", "Register"], horizontal=True, key="auth_action_radio")

    with st.sidebar.form("auth_form"):
        auth_username = st.text_input("Username", key="auth_user_field").strip()
        auth_password = st.text_input("Password", type="password", key="auth_pass_field").strip()
        auth_submit = st.form_submit_button(auth_action)

        if auth_submit:
            if not auth_username or not auth_password:
                st.sidebar.error("Username and password are required.")
            elif auth_action == "Register":
                try:
                    create_user(auth_username, auth_password)
                    st.sidebar.success(f"Account created for '{auth_username}'! You can now log in.")
                except UserAlreadyExistsError:
                    st.sidebar.error(f"Username '{auth_username}' is already taken.")
                except Exception as e:
                    st.sidebar.error(f"Registration failed: {e}")
            elif auth_action == "Login":
                try:
                    user_record = authenticate_user(auth_username, auth_password)
                    st.session_state["current_user"] = user_record
                    st.sidebar.success(f"Welcome back, {user_record['username']}!")
                    st.rerun()
                except AuthenticationError:
                    st.sidebar.error("Invalid username or password.")
                except Exception as e:
                    st.sidebar.error(f"Login failed: {e}")
else:
    current_username = st.session_state["current_user"]["username"]
    st.sidebar.markdown(f"👤 Logged in as: **{current_username}**")
    if st.sidebar.button("Logout", key="btn_logout"):
        st.session_state["current_user"] = None
        st.rerun()

st.sidebar.divider()

# --- SIDEBAR: HISTORY & MANAGEMENT ---
st.sidebar.title("Analysis History")

active_user = st.session_state["current_user"]
active_user_id = active_user["id"] if active_user else None

if active_user:
    st.sidebar.caption(f"Private history for **{active_user['username']}**")
else:
    st.sidebar.caption("Shared / Guest workspace")

# Fetch updated history for the active workspace
history = get_all_entries(user_id=active_user_id)

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
                try:
                    delete_entry(entry_id, user_id=active_user_id)
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to delete entry: {e}")


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


def render_feature_dependency_graph(
    df: pd.DataFrame,
    initial_graph_data: Dict[str, object],
) -> None:
    """Render interactive feature correlation dependency network graph with topological metrics and MST."""
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    if len(numeric_cols) < 2:
        st.info("At least 2 numeric features are required to generate a dependency network graph.")
        return

    st.markdown("#### 🕸️ Feature Correlation Dependency Network Graph")
    st.caption(
        "Topological graph modeling feature dependencies. Nodes represent continuous features; "
        "edges represent Pearson correlation $|r| \\ge \\tau$. Graph algorithms compute Degree Centrality, "
        "Connected Components (via Disjoint Set Union), and Maximum Spanning Tree (MST backbone via Kruskal's algorithm)."
    )

    c_cfg1, c_cfg2 = st.columns([2, 1])
    with c_cfg1:
        corr_thresh = st.slider(
            "Correlation Edge Threshold (τ):",
            min_value=0.10,
            max_value=0.95,
            value=float(initial_graph_data.get("threshold", 0.30)),
            step=0.05,
            key="graph_corr_thresh_slider",
        )
    with c_cfg2:
        mst_only = st.checkbox(
            "Show MST Backbone Only",
            value=False,
            help="Filters graph edges to Kruskal's Maximum Spanning Tree (MST) acyclic backbone.",
            key="graph_mst_only_toggle",
        )

    # Compute or reuse graph data based on slider
    if abs(corr_thresh - float(initial_graph_data.get("threshold", 0.30))) > 1e-4:
        graph_data = build_feature_dependency_graph(df, threshold=corr_thresh)
    else:
        graph_data = initial_graph_data

    metrics = graph_data.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Features (Nodes)", metrics.get("total_nodes", len(numeric_cols)))
    m2.metric("Dependencies (Edges)", metrics.get("total_edges", 0))
    m3.metric("Network Density", f"{metrics.get('density', 0.0):.3f}")
    m4.metric("Connected Components", metrics.get("num_connected_components", 1))
    isolated = metrics.get("isolated_nodes", [])
    m5.metric("Isolated Features", len(isolated) if isinstance(isolated, list) else 0)

    nodes_list = graph_data.get("nodes", [])
    edges_list = graph_data.get("edges", [])
    if not isinstance(nodes_list, list) or not isinstance(edges_list, list):
        return

    node_pos: Dict[str, Tuple[float, float]] = {}
    for n in nodes_list:
        if isinstance(n, dict):
            node_pos[str(n["id"])] = (float(n.get("x", 0.0)), float(n.get("y", 0.0)))

    fig = go.Figure()

    # Draw edges
    for e in edges_list:
        if not isinstance(e, dict):
            continue
        if mst_only and not bool(e.get("is_mst_backbone", False)):
            continue

        src = str(e["source"])
        tgt = str(e["target"])
        if src not in node_pos or tgt not in node_pos:
            continue

        x0, y0 = node_pos[src]
        x1, y1 = node_pos[tgt]
        corr_val = float(e.get("correlation", 0.0))
        weight_val = float(e.get("weight", abs(corr_val)))
        is_mst = bool(e.get("is_mst_backbone", False))

        if is_mst:
            line_color = "#1f77b4" if corr_val >= 0 else "#d62728"
            line_width = max(2.5 * weight_val, 1.5)
        else:
            line_color = "rgba(31, 119, 180, 0.4)" if corr_val >= 0 else "rgba(214, 39, 40, 0.4)"
            line_width = max(1.5 * weight_val, 0.8)

        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=line_width, color=line_color),
            hoverinfo="text",
            hovertext=f"{src} ↔ {tgt}<br>r = {corr_val:.3f} (weight={weight_val:.3f})<br>MST Backbone: {is_mst}",
            showlegend=False,
        ))

    # Draw nodes
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_size: List[float] = []
    node_color: List[int] = []

    for n in nodes_list:
        if not isinstance(n, dict):
            continue
        nid = str(n["id"])
        nx, ny = node_pos.get(nid, (0.0, 0.0))
        node_x.append(nx)
        node_y.append(ny)
        deg = int(n.get("degree", 0))
        cent = float(n.get("degree_centrality", 0.0))
        str_val = float(n.get("strength", 0.0))
        comm = int(n.get("community_id", 0))

        node_text.append(
            f"<b>{nid}</b><br>Degree: {deg}<br>Centrality: {cent:.3f}<br>Strength: {str_val:.3f}<br>Component: {comm}"
        )
        node_size.append(max(20 + 35 * cent, 16))
        node_color.append(comm)

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[str(n["id"]) for n in nodes_list if isinstance(n, dict)],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            colorscale="Viridis",
            line=dict(width=2, color="#333333"),
            showscale=False,
        ),
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Feature Dependency Network (τ = {corr_thresh:.2f}{' | MST Backbone' if mst_only else ''})",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Topological Centrality Leaderboard Table
    st.write("##### 🏆 Graph Centrality & Topological Hubs")
    cent_rows = []
    for n in sorted(nodes_list, key=lambda item: float(item.get("degree_centrality", 0.0)), reverse=True):  # type: ignore[arg-type]
        if isinstance(n, dict):
            cent_rows.append({
                "Feature": n["id"],
                "Degree": n["degree"],
                "Degree Centrality": f"{float(n.get('degree_centrality', 0.0)):.3f}",
                "Weighted Strength": f"{float(n.get('strength', 0.0)):.3f}",
                "Community (DSU Cluster)": n.get("community_id", 0),
            })
    st.dataframe(pd.DataFrame(cent_rows), use_container_width=True)


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
        options=[
            "IQR (Interquartile Range)",
            "Z-Score (|z| > 3)",
            "Custom Heuristic Rule Engine (Explainable AI)",
            "Isolation Forest (ML Anomaly Detection)",
        ],
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

    elif method.startswith("Custom Heuristic"):
        heur_data = outliers_data.get("heuristic_rules", {})  # type: ignore[union-attr]
        if not isinstance(heur_data, dict) or not heur_data:
            st.info("No heuristic rule anomaly data available.")
            return

        h_count = int(heur_data.get("total_anomalies", 0))
        h_pct = float(heur_data.get("anomaly_percentage", 0.0))
        h_rules_cnt = int(heur_data.get("total_rules", 0))
        outlier_indices = set(heur_data.get("anomalous_indices", []))

        hc1, hc2 = st.columns(2)
        hc1.metric("Heuristic Anomalies Flagged", h_count, delta=f"{h_pct}% of dataset")
        hc2.metric("Declarative Rules Synthesized", h_rules_cnt)

        st.info(
            "💡 **Computer Science Contribution - Explainable Rule Engine vs Blackbox Isolation Forest:**\n\n"
            "While Scikit-Learn's IsolationForest operates as an uninterpretable ensemble of random isolation trees, "
            "this custom engine synthesizes declarative heuristic rules (multivariate coupling discrepancies, "
            "statistical outer fences, and domain invariants) and computes a weighted composite anomaly score "
            "with full causality attribution for every observation."
        )

        rules_list = heur_data.get("rules_evaluated", [])
        if isinstance(rules_list, list) and rules_list:
            st.write("##### 📜 Synthesized Heuristic Rules & Violation Diagnostics")
            st.dataframe(pd.DataFrame(rules_list)[[
                "rule_id", "name", "rule_type", "severity", "violation_count", "violation_pct", "description"
            ]].rename(columns={
                "rule_id": "Rule ID",
                "name": "Rule Name",
                "rule_type": "Rule Type",
                "severity": "Severity Weight",
                "violation_count": "Violations",
                "violation_pct": "Violation %",
                "description": "Rule Logic",
            }), use_container_width=True)

        plot_df = df.copy()
        plot_df["Status"] = ["Heuristic Anomaly" if i in outlier_indices else "Normal" for i in plot_df.index]

        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_axis = st.selectbox("X-Axis Feature:", numeric_cols, index=0, key="heur_x")
            with c2:
                y_axis = st.selectbox("Y-Axis Feature:", numeric_cols, index=1, key="heur_y")

            fig = px.scatter(
                plot_df,
                x=x_axis,
                y=y_axis,
                color="Status",
                color_discrete_map={"Normal": "#1f77b4", "Heuristic Anomaly": "#d62728"},
                title=f"Heuristic Anomaly Map: {x_axis} vs {y_axis}",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                plot_df,
                x=plot_df.index,
                y=numeric_cols[0],
                color="Status",
                color_discrete_map={"Normal": "#1f77b4", "Heuristic Anomaly": "#d62728"},
                title=f"Heuristic Anomalies for '{numeric_cols[0]}'",
                labels={"x": "Index", "y": numeric_cols[0]},
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        top_anom = heur_data.get("top_anomalies", [])
        if isinstance(top_anom, list) and top_anom:
            st.write("##### 🔬 Explainable AI: Record-Level Attribution & Rule Violations")
            attribution_rows = []
            for item in top_anom:
                if isinstance(item, dict):
                    violations_str = "; ".join(
                        f"[{v.get('rule_id')}]: {v.get('description')}"
                        for v in item.get("violations", [])
                        if isinstance(v, dict)
                    )
                    attribution_rows.append({
                        "Record Index": item.get("index"),
                        "Composite Anomaly Score": item.get("composite_score"),
                        "Violated Rules Count": item.get("violated_rules_count"),
                        "Attributed Rule Violations": violations_str,
                    })
            st.dataframe(pd.DataFrame(attribution_rows), use_container_width=True)

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


def render_memory_evaluation_tab(file_or_data: Optional[object]) -> None:
    """
    Render interactive Memory Footprint Evaluation dashboard comparing chunk sizes.
    Demonstrates space complexity O(N) vs O(C) for CS faculty panels.
    """
    st.markdown("#### 🔬 Empirical Memory Footprint Evaluation across Chunk Sizes")
    st.caption(
        "Demonstrating asymptotic space complexity differences between monolithic $O(N)$ DataFrame "
        "loading versus bounded $O(C)$ streaming cleaning."
    )

    eval_col1, eval_col2 = st.columns([1, 2])
    with eval_col1:
        st.write("##### ⚙️ Benchmark Parameters")
        selected_sizes = st.multiselect(
            "Chunk Sizes to Compare (C):",
            options=[500, 1000, 2500, 5000, 10000, 25000],
            default=[500, 1000, 5000, 10000],
            key="benchmark_chunk_sizes",
        )
        run_btn = st.button("🚀 Run Empirical Benchmark", key="btn_run_mem_bench", use_container_width=True)

    with eval_col2:
        st.info(
            "💡 **Computer Science Asymptotic Complexity Note:**\n\n"
            "- **Monolithic `pd.read_csv`:** Space $\\Theta(N \\cdot M)$ auxiliary memory where $N$ is row count.\n"
            "- **Chunked Streaming (`chunksize=C`):** Working set strictly bounded to $\\Theta(C \\cdot M)$, "
            "reducing peak heap pressure and garbage collector pauses."
        )

    if run_btn:
        with st.spinner("Profiling heap allocations using tracemalloc..."):
            sizes_to_test = sorted(selected_sizes) if selected_sizes else [500, 1000, 5000, 10000]
            bench_results = evaluate_memory_footprint(file_or_data, chunk_sizes=sizes_to_test)
            st.session_state["benchmark_results"] = bench_results

    if "benchmark_results" in st.session_state and st.session_state["benchmark_results"]:
        bench_data = st.session_state["benchmark_results"]
        ds_info = bench_data.get("dataset_info", {})
        evals = bench_data.get("evaluations", [])

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Dataset Rows", f"{ds_info.get('total_rows', 0):,}")
        m2.metric("Baseline Peak RAM (O(N))", f"{bench_data.get('baseline_peak_kb', 0):,.1f} KB")
        m3.metric("Optimal Chunk Size (C*)", f"{bench_data.get('best_streaming_chunk_size', 0):,} rows")
        max_red = bench_data.get("max_memory_reduction_pct", 0.0)
        m4.metric("Max Memory Reduction", f"{max_red}%", delta=f"{max_red}% saved" if max_red > 0 else None)

        eval_df = pd.DataFrame(evals)

        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            fig_mem = px.bar(
                eval_df,
                x="chunk_size_label",
                y="peak_memory_kb",
                color="mode",
                title="Peak Memory Footprint (KB) by Chunk Size",
                labels={"chunk_size_label": "Ingestion Mode / Chunk Size", "peak_memory_kb": "Peak Memory (KB)"},
                template="plotly_white",
            )
            base_kb = bench_data.get("baseline_peak_kb", 0.0)
            fig_mem.add_hline(
                y=base_kb,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Monolithic O(N) Baseline ({base_kb} KB)",
            )
            fig_mem.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
            st.plotly_chart(fig_mem, use_container_width=True)

        with c_chart2:
            fig_time = px.line(
                eval_df[eval_df["chunk_size"].notna()],
                x="chunk_size",
                y="elapsed_seconds",
                markers=True,
                title="Execution Latency Tradeoff Curve",
                labels={"chunk_size": "Chunk Size (C)", "elapsed_seconds": "Wall-Clock Time (seconds)"},
                template="plotly_white",
            )
            fig_time.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_time, use_container_width=True)

        st.write("##### 📊 Quantitative Benchmark Results")
        display_cols = [
            "mode",
            "chunk_size_label",
            "peak_memory_kb",
            "peak_memory_mb",
            "elapsed_seconds",
            "throughput_rows_sec",
            "memory_reduction_pct",
            "num_chunks",
        ]
        col_names = {
            "mode": "Pipeline Strategy",
            "chunk_size_label": "Chunk Size",
            "peak_memory_kb": "Peak Memory (KB)",
            "peak_memory_mb": "Peak Memory (MB)",
            "elapsed_seconds": "Latency (s)",
            "throughput_rows_sec": "Throughput (rows/s)",
            "memory_reduction_pct": "RAM Reduction (%)",
            "num_chunks": "Chunk Iterations",
        }
        renamed_df = eval_df[display_cols].rename(columns=col_names)
        st.dataframe(renamed_df, use_container_width=True)


def render_imputation_drift_benchmark(
    df: pd.DataFrame,
    initial_drift_data: Dict[str, object],
) -> None:
    """Render interactive Imputation Data Drift Benchmark evaluating Wasserstein, KS, and PSI metrics."""
    st.markdown("#### 🔬 Imputation Algorithm Data Drift Benchmark")
    st.caption(
        "Quantitative empirical benchmark evaluating how different missing-data imputation algorithms "
        "alter the underlying probability distribution against mathematical data drift metrics."
    )

    with st.expander("💡 Mathematical Formulations of Distribution Drift Metrics", expanded=False):
        st.markdown(
            "- **Wasserstein Distance ($W_1$ / Earth Mover's Distance):** "
            "$W_1(P, Q) = \\int |F_P(t) - F_Q(t)| dt$. Measures the minimum mass-transportation cost "
            "to morph the imputed distribution into the true observed distribution.\n"
            "- **Kolmogorov-Smirnov Statistic ($D_{KS}$):** "
            "$D = \\sup_x |F_{\\text{imputed}}(x) - F_{\\text{observed}}(x)| \\in [0, 1]$. "
            "Measures maximum vertical deviation between empirical cumulative distribution functions.\n"
            "- **Population Stability Index (PSI):** "
            "$\\text{PSI} = \\sum_{b=1}^B (P_b - Q_b) \\ln(P_b / Q_b)$. Quantifies bin-wise probability divergence "
            "(PSI < 0.1: Stable, 0.1-0.25: Moderate Drift, > 0.25: Severe Drift).\n"
            "- **Moment Preservation ($\Delta\mu, \Delta\sigma$):** "
            "Absolute shift in first (mean) and second (standard deviation) central moments."
        )

    drift_data = initial_drift_data
    leaderboard = drift_data.get("leaderboard", [])
    if not isinstance(leaderboard, list) or not leaderboard:
        st.info("No numeric features with sufficient sample size (>=10) for imputation drift benchmarking.")
        return

    optimal_strategy = str(drift_data.get("optimal_strategy", "mean")).capitalize()
    st.success(f"🏆 **Optimal Imputation Strategy for this Dataset:** `{optimal_strategy}` (Minimizes empirical composite drift)")

    # Display Leaderboard
    st.write("##### 📊 Imputation Strategy Drift Scorecard")
    lb_df = pd.DataFrame(leaderboard)
    column_renames = {
        "rank": "Rank",
        "strategy": "Imputation Algorithm",
        "composite_drift_score": "Composite Drift Score",
        "avg_ks_statistic": "Avg KS Statistic (D)",
        "avg_psi": "Avg PSI",
        "avg_wasserstein": "Avg Wasserstein (W1)",
        "avg_mean_shift": "Avg Mean Shift (Δμ)",
        "avg_std_shift": "Avg Std Shift (Δσ)",
    }
    display_df = lb_df[[c for c in column_renames if c in lb_df.columns]].rename(columns=column_renames)
    st.dataframe(display_df, use_container_width=True)

    fig_lb = px.bar(
        lb_df,
        x="strategy",
        y="composite_drift_score",
        color="strategy",
        title="Composite Distribution Drift Score by Imputation Method (Lower is Better)",
        labels={"strategy": "Imputation Method", "composite_drift_score": "Composite Drift Score"},
        template="plotly_white",
    )
    fig_lb.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_lb, use_container_width=True)

    valid_features = drift_data.get("evaluated_features", [])
    if isinstance(valid_features, list) and valid_features:
        st.write("##### 🔍 Feature-Level Distribution Inspection")
        sel_feat = st.selectbox("Select Feature to Inspect:", valid_features, key="drift_feat_select")
        per_feat = drift_data.get("per_feature_results", {})
        if isinstance(per_feat, dict) and sel_feat in per_feat:
            feat_metrics = per_feat[sel_feat]
            feat_rows = []
            for strat, metrics_map in feat_metrics.items():
                if isinstance(metrics_map, dict):
                    feat_rows.append({
                        "Strategy": strat,
                        "Composite Drift Score": metrics_map.get("composite_score", 0.0),
                        "KS Statistic": metrics_map.get("ks_statistic", 0.0),
                        "PSI": metrics_map.get("psi", 0.0),
                        "Wasserstein (W1)": metrics_map.get("wasserstein", 0.0),
                        "Mean Shift": metrics_map.get("mean_shift", 0.0),
                        "Std Shift": metrics_map.get("std_shift", 0.0),
                    })
            st.dataframe(pd.DataFrame(feat_rows).sort_values("Composite Drift Score"), use_container_width=True)


# --- MAIN APP: UPLOAD & ANALYZE ---

st.write("### 📂 Tabular Dataset Ingestion")
with st.expander("⚙️ Dataset Ingestion & Batching Configuration", expanded=False):
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        use_chunked = st.checkbox(
            "Enable Chunked Ingestion (Batch Processing)",
            value=True,
            help="Parses CSV data in batch chunks during ingestion before in-memory DataFrame assembly.",
            key="cfg_use_chunked",
        )
    with cfg_col2:
        selected_chunksize = st.select_slider(
            "Ingestion Chunk Size (rows per batch):",
            options=[1000, 5000, 10000, 25000, 50000],
            value=DEFAULT_CHUNKSIZE,
            disabled=not use_chunked,
            key="cfg_chunksize",
        )

file = st.file_uploader("Upload CSV", type="csv")

if file:
    df: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, object]] = None

    try:
        if use_chunked:
            raw_df = read_csv_chunked(file, chunksize=selected_chunksize)
        else:
            raw_df = pd.read_csv(file)

        if hasattr(file, "seek"):
            file.seek(0)

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
            tab_summary, tab_corr, tab_network, tab_skew, tab_outliers, tab_drift, tab_memory = st.tabs([
                "📋 Summary & Distribution",
                "🔥 Correlation Heatmap",
                "🕸️ Dependency Network Graph",
                "📈 Skewness & Kurtosis",
                "🎯 Outlier Detection",
                "🔬 Imputation Drift Benchmark",
                "⚡ Memory & Chunk Evaluation",
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

            with tab_network:
                render_feature_dependency_graph(df, summary.get("dependency_graph", {}))  # type: ignore[arg-type]

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

            with tab_drift:
                render_imputation_drift_benchmark(df, summary.get("imputation_drift", {}))  # type: ignore[arg-type]

            with tab_memory:
                render_memory_evaluation_tab(file)

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

                        active_uid = (
                            st.session_state["current_user"]["id"]
                            if st.session_state.get("current_user")
                            else None
                        )
                        save_entry(custom_title, insights, user_id=active_uid)
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
else:
    with st.expander("🔬 CS Faculty Evaluation: Live Memory Benchmark Sandbox", expanded=False):
        st.caption(
            "No CSV uploaded yet. Run an empirical memory footprint evaluation against a synthetic 20,000-row dataset "
            "to benchmark monolithic $O(N)$ DataFrame loading vs bounded $O(C)$ streaming cleaning across chunk sizes:"
        )
        render_memory_evaluation_tab(None)