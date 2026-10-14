import io
import os
import streamlit as st
import pandas as pd
from typing import Dict, Optional, Tuple

from analyzer import generate_summary
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
from visualizations import (
    render_correlation_heatmap,
    render_distribution_chart,
    render_feature_dependency_graph,
    render_imputation_drift_benchmark,
    render_memory_evaluation_tab,
    render_outlier_inspector,
    render_preprocessing_pipeline,
    render_sidebar_insights,
    render_skewness_kurtosis_chart,
    render_structured_insights,
)

__all__ = [
    "get_current_user_id",
    "load_dataset",
    "render_sidebar_insights",
    "render_distribution_chart",
    "render_correlation_heatmap",
    "render_feature_dependency_graph",
    "render_skewness_kurtosis_chart",
    "render_outlier_inspector",
    "render_preprocessing_pipeline",
    "render_structured_insights",
    "render_memory_evaluation_tab",
    "render_imputation_drift_benchmark",
]

# Propagate Streamlit secrets configuration to environment if present
if hasattr(st, "secrets") and "SQLITE_DB_PATH" in st.secrets and "SQLITE_DB_PATH" not in os.environ:
    os.environ["SQLITE_DB_PATH"] = str(st.secrets["SQLITE_DB_PATH"])

# 1. Initialize DB on app startup
init_db()

st.set_page_config(page_title="AI Data Processor", page_icon="📊", layout="wide")
st.title("AI Data Processor")

# --- SESSION STATE INITIALIZATION ---
if "generated_insight" not in st.session_state:
    st.session_state["generated_insight"] = None
if "current_user" not in st.session_state:
    st.session_state["current_user"] = None
if "dataset_summary" not in st.session_state:
    st.session_state["dataset_summary"] = None
if "current_file_id" not in st.session_state:
    st.session_state["current_file_id"] = None


def get_current_user_id() -> Optional[int]:
    """Retrieve the active analyst's user ID from session state, if authenticated."""
    current_u = st.session_state.get("current_user")
    if isinstance(current_u, dict) and "id" in current_u:
        try:
            return int(current_u["id"])
        except (ValueError, TypeError):
            return None
    return None


@st.cache_data
def load_dataset(
    file_bytes: bytes,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]], Optional[str]]:
    """Ingest CSV tabular dataset and compute baseline summary statistics with caching."""
    if not file_bytes or len(file_bytes.strip()) == 0:
        return None, None, "The uploaded CSV file is empty."
    try:
        buffer = io.BytesIO(file_bytes)
        df = pd.read_csv(buffer)
        if df.empty:
            return None, None, "The uploaded CSV file is empty."
        summary = generate_summary(df)
        return df, summary, None
    except Exception as e:
        return None, None, str(e)


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
active_user_id = get_current_user_id()

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


# --- MAIN APP: UPLOAD & ANALYZE ---

st.write("### 📂 Tabular Dataset Ingestion")
with st.expander("⚙️ Dataset Preprocessing Configuration", expanded=False):
    selected_strategy = st.selectbox(
        "Imputation Strategy:",
        options=["mean", "median", "mode", "drop"],
        index=0,
        help="Missing value imputation strategy applied during dataset analysis.",
        key="cfg_imputation_strategy",
    )

file = st.file_uploader("Upload CSV", type="csv")

if file:
    file_id = getattr(file, "file_id", getattr(file, "name", "default_file"))
    if st.session_state.get("current_file_id") != file_id:
        st.session_state["current_file_id"] = file_id
        st.session_state["generated_insight"] = None
        st.session_state["imputation_drift_results"] = None
        st.session_state["benchmark_results"] = None

    df: Optional[pd.DataFrame] = None
    summary: Optional[Dict[str, object]] = None

    try:
        file_bytes: bytes = file.getvalue() if hasattr(file, "getvalue") else file.read()
        df, summary, err = load_dataset(file_bytes)

        if hasattr(file, "seek"):
            file.seek(0)

        if err:
            st.session_state["dataset_summary"] = None
            st.error(f"Failed to process CSV: {err}")
        else:
            st.session_state["dataset_summary"] = summary
    except Exception as e:
        st.session_state["dataset_summary"] = None
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

                        active_uid = get_current_user_id()
                        save_entry(custom_title, insights, user_id=active_uid)
                        st.success("Saved to History!")

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