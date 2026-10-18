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

        # Empty data
        mock_st.reset_mock()
        visualizations.render_imputation_drift_benchmark(self.df, {})
        mock_st.info.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()

