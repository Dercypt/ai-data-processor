import io
import json
import os
import sys
import unittest
from unittest.mock import MagicMock

# Mock streamlit before importing modules that depend on it
mock_st = MagicMock()
mock_st.cache_data.side_effect = lambda *a, **k: a[0] if len(a) == 1 and callable(a[0]) and not k else (lambda f: f)
sys.modules["streamlit"] = mock_st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import components  # noqa: E402
import visualizations  # noqa: E402
import numpy as np
import pandas as pd


class TestVisualizations(unittest.TestCase):
    """Unit tests for the extracted visualizations and UI components module."""

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "a": np.random.normal(0, 1, 50),
            "b": np.random.normal(5, 2, 50),
            "c": ["cat", "dog"] * 25,
        })
        mock_st.reset_mock()
        mock_st.button.return_value = False
        mock_st.session_state = {}

        def mock_columns(spec):
            n = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(n)]

        mock_st.columns.side_effect = mock_columns

    def test_module_exports_and_components_reexport(self):
        """visualizations and components should export identical symbols in __all__."""
        expected_exports = [
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
        self.assertEqual(sorted(visualizations.__all__), sorted(expected_exports))
        self.assertEqual(sorted(components.__all__), sorted(expected_exports))
        for symbol in expected_exports:
            self.assertTrue(hasattr(visualizations, symbol))
            self.assertTrue(hasattr(components, symbol))
            self.assertIs(getattr(visualizations, symbol), getattr(components, symbol))

    def test_render_sidebar_insights_json(self):
        """render_sidebar_insights should parse JSON and render markdown."""
        insights_data = {
            "summary": "Everything looks healthy.",
            "risks": ["Slight skewness in column A"],
            "recommendations": ["Normalize feature A"],
        }
        visualizations.render_sidebar_insights(json.dumps(insights_data))
        self.assertTrue(mock_st.markdown.called)

    def test_render_sidebar_insights_fallback(self):
        """render_sidebar_insights should fall back to st.info on non-JSON text."""
        visualizations.render_sidebar_insights("Plain text insights")
        mock_st.info.assert_called_with("Plain text insights")

    def test_render_distribution_chart(self):
        """render_distribution_chart should build Plotly figure and call st.plotly_chart."""
        visualizations.render_distribution_chart(self.df, "a")
        mock_st.plotly_chart.assert_called_once()

    def test_render_correlation_heatmap_valid_and_empty(self):
        """render_correlation_heatmap should render heatmap or info when empty."""
        corr_data = {"a": {"a": 1.0, "b": 0.5}, "b": {"a": 0.5, "b": 1.0}}
        visualizations.render_correlation_heatmap(corr_data)
        self.assertTrue(mock_st.plotly_chart.called)

        mock_st.reset_mock()
        visualizations.render_correlation_heatmap({})
        mock_st.info.assert_called_once()
        mock_st.plotly_chart.assert_not_called()

    def test_render_feature_dependency_graph(self):
        """render_feature_dependency_graph should render network graph and leaderboard."""
        initial_graph_data = {
            "threshold": 0.30,
            "metrics": {
                "total_nodes": 2,
                "total_edges": 1,
                "density": 1.0,
                "num_connected_components": 1,
                "isolated_nodes": [],
            },
            "nodes": [
                {"id": "a", "x": 0.0, "y": 0.0, "degree": 1, "degree_centrality": 1.0, "strength": 0.5, "community_id": 0},
                {"id": "b", "x": 1.0, "y": 1.0, "degree": 1, "degree_centrality": 1.0, "strength": 0.5, "community_id": 0},
            ],
            "edges": [
                {"source": "a", "target": "b", "correlation": 0.5, "weight": 0.5, "is_mst_backbone": True},
            ],
        }
        mock_st.slider.return_value = 0.30
        mock_st.checkbox.return_value = False

        visualizations.render_feature_dependency_graph(self.df, initial_graph_data)
        self.assertTrue(mock_st.plotly_chart.called)
        self.assertTrue(mock_st.dataframe.called)

    def test_render_feature_dependency_graph_insufficient_numeric(self):
        """render_feature_dependency_graph should display info if < 2 numeric features."""
        single_df = pd.DataFrame({"cat": ["x", "y"]})
        visualizations.render_feature_dependency_graph(single_df, {})
        mock_st.info.assert_called_once()

    def test_render_skewness_kurtosis_chart(self):
        """render_skewness_kurtosis_chart should render bar chart and dataframe table."""
        skew_data = {"a": 0.1, "b": -0.2}
        kurt_data = {"a": 2.9, "b": 3.1}
        skew_desc = {"a": "Symmetric", "b": "Symmetric"}
        kurt_desc = {"a": "Mesokurtic", "b": "Mesokurtic"}

        visualizations.render_skewness_kurtosis_chart(skew_data, kurt_data, skew_desc, kurt_desc)
        mock_st.plotly_chart.assert_called_once()
        mock_st.dataframe.assert_called_once()

        mock_st.reset_mock()
        visualizations.render_skewness_kurtosis_chart({}, {}, {}, {})
        mock_st.info.assert_called_once()

    def test_render_outlier_inspector(self):
        """render_outlier_inspector should render for IQR, Z-Score, Heuristic, and Isolation Forest."""
        outliers_data = {
            "iqr": {"a": {"count": 1, "percentage": 2.0, "lower_bound": -2.0, "upper_bound": 2.0, "outlier_indices": [0]}},
            "zscore": {"a": {"count": 1, "percentage": 2.0, "mean": 0.0, "std": 1.0, "threshold": 3.0, "outlier_indices": [0]}},
            "heuristic_rules": {
                "total_anomalies": 1,
                "anomaly_percentage": 2.0,
                "total_rules": 1,
                "anomalous_indices": [0],
                "rules_evaluated": [
                    {"rule_id": "R1", "name": "Outer", "rule_type": "fence", "severity": 1.0, "violation_count": 1, "violation_pct": 2.0, "description": "fence"}
                ],
                "top_anomalies": [{"index": 0, "composite_score": 1.0, "violated_rules_count": 1, "violations": []}],
            },
            "isolation_forest": {"count": 1, "percentage": 2.0, "outlier_indices": [0]},
        }

        mock_st.selectbox.return_value = "a"

        # Test IQR
        mock_st.radio.return_value = "IQR (Interquartile Range)"
        visualizations.render_outlier_inspector(self.df, outliers_data)
        self.assertTrue(mock_st.plotly_chart.called)

        # Test Z-Score
        mock_st.reset_mock()
        mock_st.radio.return_value = "Z-Score (|z| > 3)"
        visualizations.render_outlier_inspector(self.df, outliers_data)
        self.assertTrue(mock_st.plotly_chart.called)

        # Test Heuristic Rules
        mock_st.reset_mock()
        mock_st.radio.return_value = "Custom Heuristic Rule Engine (Explainable AI)"
        visualizations.render_outlier_inspector(self.df, outliers_data)
        self.assertTrue(mock_st.plotly_chart.called)

        # Test Isolation Forest
        mock_st.reset_mock()
        mock_st.radio.return_value = "Isolation Forest (ML Anomaly Detection)"
        visualizations.render_outlier_inspector(self.df, outliers_data)
        self.assertTrue(mock_st.plotly_chart.called)

    def test_render_preprocessing_pipeline_no_missing(self):
        """render_preprocessing_pipeline returns clean df and summary without imputation if complete."""
        clean_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        detected = {"x": "continuous", "y": "continuous"}

        df_out, summary_out = visualizations.render_preprocessing_pipeline(clean_df, detected)
        self.assertEqual(len(df_out), 3)
        self.assertIn("column_types", summary_out)
        self.assertIn("dataset_summary", mock_st.session_state)
        self.assertEqual(mock_st.session_state["dataset_summary"], summary_out)
        mock_st.success.assert_called()

    def test_render_preprocessing_pipeline_with_imputation_persists_summary(self):
        """render_preprocessing_pipeline persists summary to session_state after imputation."""
        dirty_df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
        detected = {"x": "continuous", "y": "continuous"}

        mock_st.radio.return_value = "Global Strategy"
        mock_st.selectbox.return_value = "Mean"

        df_out, summary_out = visualizations.render_preprocessing_pipeline(dirty_df, detected)
        self.assertEqual(len(df_out), 3)
        self.assertEqual(int(df_out["x"].isnull().sum()), 0)
        self.assertIn("dataset_summary", mock_st.session_state)
        self.assertEqual(mock_st.session_state["dataset_summary"], summary_out)

    def test_render_preprocessing_pipeline_gates_mean_median_for_non_numeric_continuous(self):
        """Object column detected as continuous must not offer Mean/Median options."""
        raw_df = pd.DataFrame({
            "num_str_col": ["10", "20", "30", "40", np.nan],
            "pure_num_col": [1.0, 2.0, 3.0, 4.0, np.nan],
        })
        detected = {
            "num_str_col": "continuous",
            "pure_num_col": "continuous",
        }

        mock_st.radio.return_value = "Per-Column Strategy"
        options_recorded = {}

        def fake_selectbox(label, options, key=None):
            options_recorded[key] = options
            return options[0]

        mock_st.selectbox.side_effect = fake_selectbox
        try:
            visualizations.render_preprocessing_pipeline(raw_df, detected)
            self.assertEqual(
                options_recorded.get("col_impute_num_str_col"),
                ["Mode", "Forward Fill", "Random Draw", "Drop"],
            )
            self.assertEqual(
                options_recorded.get("col_impute_pure_num_col"),
                ["Mean", "Median", "Mode", "Zero", "Forward Fill", "Random Draw", "Drop"],
            )
        finally:
            mock_st.selectbox.side_effect = None

    def test_render_preprocessing_pipeline_suppresses_success_banner_on_imputation_error(self):
        """When impute_missing_values raises an error, success confirmation must never be rendered."""
        from unittest.mock import patch

        raw_df = pd.DataFrame({"col_a": [1.0, np.nan, 3.0]})
        detected = {"col_a": "continuous"}

        # 1. Per-Column Strategy failure
        mock_st.radio.return_value = "Per-Column Strategy"
        mock_st.selectbox.return_value = "Mean"

        with patch("visualizations.impute_missing_values", side_effect=ValueError("Cannot apply mean")):
            df_out, _ = visualizations.render_preprocessing_pipeline(raw_df, detected)
            self.assertEqual(len(df_out), 3)

        mock_st.error.assert_called_with("Imputation failed: Cannot apply mean")
        for call in mock_st.success.call_args_list:
            arg = call[0][0] if call[0] else ""
            self.assertNotIn("All missing values resolved", str(arg))
            self.assertNotIn("Row count preserved", str(arg))

        # Reset mocks
        mock_st.reset_mock()
        mock_st.session_state = {}

        # 2. Global Strategy failure
        mock_st.radio.return_value = "Global Strategy"
        mock_st.selectbox.return_value = "Drop"

        with patch("visualizations.impute_missing_values", side_effect=ValueError("Dropping missing values resulted in an empty dataset.")):
            df_out_global, _ = visualizations.render_preprocessing_pipeline(raw_df, detected)
            self.assertEqual(len(df_out_global), 3)

        mock_st.error.assert_called_with("Imputation failed: Dropping missing values resulted in an empty dataset.")
        for call in mock_st.success.call_args_list:
            arg = call[0][0] if call[0] else ""
            self.assertNotIn("All missing values resolved", str(arg))
            self.assertNotIn("Row count preserved", str(arg))

    def test_render_preprocessing_pipeline_numeric_string_continuous_end_to_end(self):
        """End-to-end edge case: object column with >=90% numeric strings and stray N/A string."""
        from analyzer import detect_column_types

        raw_df = pd.DataFrame({
            "col_mixed": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "N/A", np.nan],
        })
        detected = detect_column_types(raw_df)
        self.assertEqual(detected["col_mixed"], "continuous")
        self.assertFalse(pd.api.types.is_numeric_dtype(raw_df["col_mixed"]))

        mock_st.radio.return_value = "Per-Column Strategy"
        options_recorded = {}

        def fake_selectbox(label, options, key=None):
            options_recorded[key] = options
            return "Mode"

        mock_st.selectbox.side_effect = fake_selectbox
        try:
            df_out, summary = visualizations.render_preprocessing_pipeline(raw_df, detected)
            self.assertEqual(
                options_recorded.get("col_impute_col_mixed"),
                ["Mode", "Forward Fill", "Random Draw", "Drop"],
            )
            mock_st.error.assert_not_called()
            mock_st.success.assert_called()
            self.assertEqual(int(df_out["col_mixed"].isnull().sum()), 0)
        finally:
            mock_st.selectbox.side_effect = None

    def test_render_preprocessing_pipeline_extended_strategies_global(self):
        """render_preprocessing_pipeline executes Zero, Forward Fill, and Random Draw globally."""
        for strat in ["Zero", "Forward Fill", "Random Draw"]:
            mock_st.reset_mock()
            mock_st.session_state = {}
            mock_st.radio.return_value = "Global Strategy"
            mock_st.selectbox.return_value = strat

            dirty_df = pd.DataFrame({"x": [10.0, np.nan, 30.0], "y": ["a", np.nan, "c"]})
            detected = {"x": "continuous", "y": "categorical"}

            df_out, summary_out = visualizations.render_preprocessing_pipeline(dirty_df, detected)
            self.assertEqual(len(df_out), 3)
            self.assertEqual(int(df_out.isnull().sum().sum()), 0)
            mock_st.error.assert_not_called()
            mock_st.success.assert_called()

    def test_render_preprocessing_pipeline_extended_strategies_per_column(self):
        """render_preprocessing_pipeline configures Zero, Forward Fill, and Random Draw per column."""
        mock_st.reset_mock()
        mock_st.session_state = {}
        mock_st.radio.return_value = "Per-Column Strategy"

        def fake_selectbox(label, options, key=None):
            if key == "col_impute_num_col":
                return "Zero"
            elif key == "col_impute_cat_col":
                return "Forward Fill"
            return options[0]

        mock_st.selectbox.side_effect = fake_selectbox
        dirty_df = pd.DataFrame({"num_col": [1.0, np.nan, 3.0], "cat_col": ["a", np.nan, "b"]})
        detected = {"num_col": "continuous", "cat_col": "categorical"}

        try:
            df_out, summary_out = visualizations.render_preprocessing_pipeline(dirty_df, detected)
            self.assertEqual(len(df_out), 3)
            self.assertEqual(int(df_out.isnull().sum().sum()), 0)
            self.assertEqual(df_out.loc[1, "num_col"], 0.0)
            self.assertEqual(df_out.loc[1, "cat_col"], "a")
            mock_st.error.assert_not_called()
            mock_st.success.assert_called()
        finally:
            mock_st.selectbox.side_effect = None

    def test_render_structured_insights(self):
        """render_structured_insights should render summary, risks, and recommendations."""
        insights_dict = {
            "summary": "Good performance.",
            "risks": ["Risk 1"],
            "recommendations": ["Rec 1"],
        }
        visualizations.render_structured_insights(insights_dict)
        mock_st.info.assert_called_with("Good performance.")
        mock_st.warning.assert_called_with("• Risk 1")
        mock_st.success.assert_called_with("• Rec 1")

        # Also test with raw string fallback
        mock_st.reset_mock()
        visualizations.render_structured_insights("plain text note")
        mock_st.write.assert_called_with("plain text note")

    def test_render_imputation_drift_benchmark(self):
        """render_imputation_drift_benchmark should render leaderboard and charts."""
        drift_data = {
            "optimal_strategy": "mean",
            "leaderboard": [
                {
                    "rank": 1,
                    "strategy": "mean",
                    "composite_drift_score": 0.05,
                    "avg_ks_statistic": 0.01,
                    "avg_psi": 0.02,
                    "avg_wasserstein": 0.03,
                    "avg_mean_shift": 0.00,
                    "avg_std_shift": 0.01,
                }
            ],
            "evaluated_features": ["a"],
            "per_feature_results": {
                "a": {
                    "mean": {
                        "composite_score": 0.05,
                        "ks_statistic": 0.01,
                        "psi": 0.02,
                        "wasserstein": 0.03,
                        "mean_shift": 0.0,
                        "std_shift": 0.01,
                    }
                }
            },
        }
        mock_st.selectbox.return_value = "a"
        visualizations.render_imputation_drift_benchmark(self.df, drift_data)
        self.assertTrue(mock_st.plotly_chart.called)
        self.assertTrue(mock_st.dataframe.called)

        # Empty data prompts user to run benchmark
        mock_st.reset_mock()
        visualizations.render_imputation_drift_benchmark(self.df, {})
        mock_st.info.assert_called_once_with("Click '🚀 Run Benchmark' to evaluate imputation drift.")

        # Evaluated drift data with empty leaderboard displays sample size message
        mock_st.reset_mock()
        visualizations.render_imputation_drift_benchmark(self.df, {"leaderboard": []})
        mock_st.info.assert_called_once_with(
            "No numeric features with sufficient sample size (>=10) for imputation drift benchmarking."
        )

    def test_render_imputation_drift_benchmark_on_demand_button(self):
        """render_imputation_drift_benchmark should compute drift on demand when run button is clicked."""
        mock_st.reset_mock()
        mock_st.button.return_value = True
        visualizations.render_imputation_drift_benchmark(self.df, {})
        self.assertIn("imputation_drift_results", mock_st.session_state)
        self.assertTrue(mock_st.plotly_chart.called)
        self.assertTrue(mock_st.dataframe.called)

    def test_cached_generate_summary(self):
        """cached_generate_summary should compute and return dataset statistical summary."""
        summary = visualizations.cached_generate_summary(self.df)
        self.assertIn("columns", summary)
        self.assertIn("stats", summary)
        self.assertEqual(summary["rows"], len(self.df))

    def test_load_dataset_detects_column_types_without_full_summary(self):
        """load_dataset should detect column types without executing redundant full summary pipeline."""
        from unittest.mock import patch
        import main

        csv_bytes = b"feature1,feature2\n1.0,10\n2.0,20\n3.0,30"
        with patch("analyzer.generate_summary") as mock_summary:
            df, summary, err = main.load_dataset(csv_bytes)
            mock_summary.assert_not_called()

        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 3)
        self.assertIsNotNone(summary)
        self.assertIn("column_types", summary)
        self.assertEqual(summary["column_types"]["feature1"], "continuous")
        self.assertNotIn("dependency_graph", summary)
        self.assertNotIn("outliers", summary)

    def test_load_dataset_empty_input_handling(self):
        """load_dataset handles empty bytes and returns graceful error without crashing."""
        import main
        df, summary, err = main.load_dataset(b"")
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertEqual(err, "The uploaded CSV file is empty.")

    def test_load_dataset_delegates_to_analyzer(self):
        """main.load_dataset should delegate ingestion to analyzer module."""
        from unittest.mock import patch
        import main

        csv_bytes = b"a,b\n1,2\n"
        with patch("main.analyzer_load_dataset") as mock_ingest:
            mock_ingest.return_value = (pd.DataFrame({"a": [1], "b": [2]}), {"column_types": {"a": "continuous", "b": "continuous"}}, None)
            df, summary, err = main.load_dataset(csv_bytes)
            mock_ingest.assert_called_once_with(csv_bytes)

    def test_render_imputation_drift_benchmark_handles_raw_unimputed_df(self):
        """render_imputation_drift_benchmark runs benchmark on raw un-imputed data when clicked."""
        from unittest.mock import patch
        raw_df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]})
        mock_st.reset_mock()
        mock_st.button.return_value = True
        with patch("visualizations.benchmark_imputation_drift") as mock_bench:
            mock_bench.return_value = {
                "optimal_strategy": "mean",
                "leaderboard": [],
                "evaluated_features": [],
                "per_feature_results": {},
            }
            visualizations.render_imputation_drift_benchmark(raw_df, {})
            mock_bench.assert_called_once_with(raw_df)

    def test_main_generate_stream_cleaned_csv_delegates_to_stream_clean_to_csv(self):
        """main.generate_stream_cleaned_csv delegates to stream_clean_to_csv and produces clean CSV."""
        from unittest.mock import patch
        import main

        csv_bytes = b"feature1,feature2\n1.0,10\n,20\n3.0,\n"
        with patch("main.stream_clean_to_csv", wraps=main.stream_clean_to_csv) as mock_stream_clean:
            cleaned_str = main.generate_stream_cleaned_csv(csv_bytes, chunksize=2)
            mock_stream_clean.assert_called_once()

        self.assertIsInstance(cleaned_str, str)
        reloaded = pd.read_csv(io.StringIO(cleaned_str))
        self.assertEqual(len(reloaded), 3)
        self.assertEqual(reloaded.isnull().sum().sum(), 0)

    def test_main_exports_generate_stream_cleaned_csv(self):
        """main module should export generate_stream_cleaned_csv in __all__."""
        import main
        self.assertIn("generate_stream_cleaned_csv", main.__all__)
        self.assertTrue(callable(getattr(main, "generate_stream_cleaned_csv")))

    def test_main_exports_clear_analysis_session_state(self):
        """main module should export clear_analysis_session_state in __all__."""
        import main
        self.assertIn("clear_analysis_session_state", main.__all__)
        self.assertTrue(callable(getattr(main, "clear_analysis_session_state")))

    def test_clear_analysis_session_state_resets_artifacts_and_preserves_user(self):
        """clear_analysis_session_state must reset insights, summaries, and benchmarks to None while preserving user."""
        import streamlit as st
        import main

        st.session_state["current_user"] = {"id": 99, "username": "analyst_alpha"}
        st.session_state["generated_insight"] = {"executive_summary": "Top secret financial insights"}
        st.session_state["dataset_summary"] = {"columns": ["revenue", "profit"], "rows": 5000}
        st.session_state["imputation_drift_results"] = {"optimal_strategy": "mean", "drift": 0.01}
        st.session_state["benchmark_results"] = {"chunk_sizes": [100, 200], "peak_memory": [12.5, 8.2]}

        main.clear_analysis_session_state()

        self.assertIsNone(st.session_state["generated_insight"])
        self.assertIsNone(st.session_state["dataset_summary"])
        self.assertIsNone(st.session_state["imputation_drift_results"])
        self.assertIsNone(st.session_state["benchmark_results"])
        self.assertEqual(st.session_state["current_user"], {"id": 99, "username": "analyst_alpha"})

    def test_clear_analysis_session_state_edge_cases(self):
        """clear_analysis_session_state should gracefully handle missing keys or already None state."""
        import streamlit as st
        import main

        for key in ["generated_insight", "dataset_summary", "imputation_drift_results", "benchmark_results"]:
            if key in st.session_state:
                del st.session_state[key]

        main.clear_analysis_session_state()

        self.assertIsNone(st.session_state["generated_insight"])
        self.assertIsNone(st.session_state["dataset_summary"])
        self.assertIsNone(st.session_state["imputation_drift_results"])
        self.assertIsNone(st.session_state["benchmark_results"])

    def test_render_preprocessing_pipeline_normalizes_infinite_without_nans(self):
        """render_preprocessing_pipeline must normalize +/-inf to NaN and not short-circuit when no NaNs exist."""
        raw_df = pd.DataFrame({
            "feat_a": [1.0, 2.0, 3.0, 4.0],
            "feat_b": [10.0, np.inf, -np.inf, 40.0],
        })
        detected = {"feat_a": "continuous", "feat_b": "continuous"}

        mock_st.radio.return_value = "Global Strategy"
        mock_st.selectbox.return_value = "Mean"

        df_out, summary_out = visualizations.render_preprocessing_pipeline(raw_df, detected)
        self.assertEqual(len(df_out), 4)
        self.assertFalse(np.isinf(df_out["feat_b"]).any())
        self.assertEqual(int(df_out["feat_b"].isnull().sum()), 0)
        self.assertIn("dataset_summary", mock_st.session_state)
        self.assertIn("stats", summary_out)

    def test_live_ui_end_to_end_infinite_and_numeric_string_coercion(self):
        """End-to-end ingestion pipeline: CSVs with +/-inf or corrupted numeric strings must not crash Isolation Forest."""
        import main

        # 1. CSV with infinite values and no explicit NaNs
        inf_csv = "feature_x,feature_y\n" + "\n".join([f"{i}.0,{i * 2}.0" for i in range(1, 15)]) + "\n15.0,inf\n"
        raw_df, summary, err = main.load_dataset(inf_csv.encode("utf-8"))
        self.assertIsNone(err)
        self.assertIsNotNone(raw_df)
        self.assertFalse(np.isinf(raw_df["feature_y"]).any())
        self.assertTrue(raw_df["feature_y"].isnull().any())

        mock_st.radio.return_value = "Global Strategy"
        mock_st.selectbox.return_value = "Mean"

        df_out, clean_summary = visualizations.render_preprocessing_pipeline(raw_df, summary["column_types"])
        self.assertFalse(np.isinf(df_out["feature_y"]).any())
        self.assertEqual(int(df_out["feature_y"].isnull().sum()), 0)
        self.assertIn("isolation_forest", clean_summary["outliers"])

        # 2. CSV with continuous numeric strings and corrupted token without explicit NaNs
        str_csv = "feature_z\n" + "\n".join([f"{i}.0" for i in range(1, 15)]) + "\ncorrupted_val\n"
        raw_str_df, str_summary, err = main.load_dataset(str_csv.encode("utf-8"))
        self.assertIsNone(err)
        self.assertIsNotNone(raw_str_df)
        self.assertTrue(pd.api.types.is_numeric_dtype(raw_str_df["feature_z"]))
        self.assertTrue(raw_str_df["feature_z"].isnull().any())

        df_str_out, clean_str_summary = visualizations.render_preprocessing_pipeline(raw_str_df, str_summary["column_types"])
        self.assertTrue(pd.api.types.is_numeric_dtype(df_str_out["feature_z"]))
        self.assertEqual(int(df_str_out["feature_z"].isnull().sum()), 0)
        self.assertIn("stats", clean_str_summary)

    def test_main_exports_export_cleaned_csv(self):
        """main module should export export_cleaned_csv and convert_df_to_csv in __all__."""
        import main

        self.assertIn("export_cleaned_csv", main.__all__)
        self.assertIn("convert_df_to_csv", main.__all__)
        self.assertTrue(callable(getattr(main, "export_cleaned_csv")))
        self.assertTrue(callable(getattr(main, "convert_df_to_csv")))

    def test_export_cleaned_csv_serializes_dataframe_correctly(self):
        """export_cleaned_csv should serialize DataFrame to CSV string preserving values."""
        import main

        df = pd.DataFrame({"col_a": [1.0, 2.0, 3.0], "col_b": ["x", "y", "z"]})
        csv_str = main.export_cleaned_csv(df)
        self.assertIsInstance(csv_str, str)
        reloaded = pd.read_csv(io.StringIO(csv_str))
        pd.testing.assert_frame_equal(reloaded, df)

    def test_export_cleaned_csv_preserves_pipeline_imputation_strategy(self):
        """export_cleaned_csv must faithfully reflect the interactive pipeline's imputed values."""
        import main

        raw_df = pd.DataFrame({"feat_a": [1.0, 2.0, np.nan, 4.0, 100.0]})
        detected = {"feat_a": "continuous"}

        mock_st.radio.return_value = "Global Strategy"
        mock_st.selectbox.return_value = "Median"

        clean_df, _ = visualizations.render_preprocessing_pipeline(raw_df, detected)
        csv_str = main.export_cleaned_csv(clean_df)

        reloaded = pd.read_csv(io.StringIO(csv_str))
        self.assertEqual(int(reloaded["feat_a"].isnull().sum()), 0)
        # Median of [1.0, 2.0, 4.0, 100.0] is 3.0
        self.assertEqual(reloaded["feat_a"].iloc[2], 3.0)

    def test_generate_stream_cleaned_csv_with_explicit_fill_values(self):
        """generate_stream_cleaned_csv must apply explicit global fill_values without drift."""
        import main

        csv_bytes = b"feature1,feature2\n1.0,10\n,20\n3.0,\n"
        fill_map = {"feature1": 99.0, "feature2": 999.0}
        cleaned_str = main.generate_stream_cleaned_csv(csv_bytes, chunksize=2, fill_values=fill_map)

        reloaded = pd.read_csv(io.StringIO(cleaned_str))
        self.assertEqual(reloaded["feature1"].iloc[1], 99.0)
        self.assertEqual(reloaded["feature2"].iloc[2], 999.0)

    def test_main_exports_initialize_database_and_main(self):
        """main module should export initialize_database and main entrypoint in __all__."""
        import main
        self.assertIn("initialize_database", main.__all__)
        self.assertIn("main", main.__all__)
        self.assertTrue(callable(getattr(main, "initialize_database")))
        self.assertTrue(callable(getattr(main, "main")))

    def test_import_main_has_no_database_side_effects(self):
        """Importing main must not execute init_db or UI rendering at import time."""
        from unittest.mock import patch
        import sys

        sys.modules.pop("main", None)
        with patch("database.init_db") as mock_init, patch("database.get_all_entries") as mock_entries:
            import main  # noqa: F401
            mock_init.assert_not_called()
            mock_entries.assert_not_called()


if __name__ == "__main__":
    unittest.main()

