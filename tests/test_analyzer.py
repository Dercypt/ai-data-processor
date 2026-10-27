import io
import unittest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analyzer import (
    analyze_dataset,
    benchmark_imputation_drift,
    compute_correlations,
    compute_skewness_and_kurtosis,
    detect_column_types,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    generate_summary,
    impute_missing_values,
    interpret_skewness,
    interpret_kurtosis,
    load_dataset,
)


class TestAnalyzer(unittest.TestCase):
    def setUp(self):
        # Generate reproducible sample data with outliers
        np.random.seed(42)
        n = 100
        data = {
            "feature_a": np.random.normal(loc=10.0, scale=2.0, size=n),
            "feature_b": np.random.exponential(scale=3.0, size=n),
            "category": ["A" if i % 2 == 0 else "B" for i in range(n)],
        }
        self.df = pd.DataFrame(data)
        # Inject known outliers
        self.df.loc[0, "feature_a"] = 100.0  # extreme high outlier
        self.df.loc[1, "feature_a"] = -50.0  # extreme low outlier

        csv_buffer = io.StringIO()
        self.df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        self.csv_file = csv_buffer

    def test_analyze_dataset_success_and_contracts(self):
        df_clean, summary, err = analyze_dataset(self.csv_file)
        self.assertIsNone(err)
        self.assertIsNotNone(df_clean)
        self.assertIsNotNone(summary)

        # Law 2: Row count & column structure preservation
        self.assertEqual(len(df_clean), 100)
        self.assertEqual(list(df_clean.columns), ["feature_a", "feature_b", "category"])
        self.assertEqual(df_clean.isnull().sum().sum(), 0)

        # Law 3: Return signature structure
        self.assertIn("columns", summary)
        self.assertIn("rows", summary)
        self.assertIn("stats", summary)
        self.assertIn("correlation", summary)
        self.assertIn("skewness", summary)
        self.assertIn("kurtosis", summary)
        self.assertIn("outliers", summary)

    def test_missing_values_imputation_integrity(self):
        # Data with NaN
        csv_data = io.StringIO("col_num,col_cat\n1.0,apple\n,banana\n3.0,\n")
        df_clean, summary, err = analyze_dataset(csv_data)
        self.assertIsNone(err)
        self.assertEqual(len(df_clean), 3)
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        self.assertEqual(df_clean.loc[1, "col_num"], 2.0)
        self.assertEqual(df_clean.loc[2, "col_cat"], "apple")

    def test_analyze_dataset_custom_imputation_strategies(self):
        # Test median strategy
        csv_data_median = io.StringIO("col_num,col_cat\n1.0,apple\n,banana\n3.0,\n")
        df_median, summary_med, err_med = analyze_dataset(csv_data_median, strategy="median")
        self.assertIsNone(err_med)
        self.assertEqual(len(df_median), 3)
        self.assertEqual(df_median.isnull().sum().sum(), 0)
        self.assertEqual(df_median.loc[1, "col_num"], 2.0)

        # Test drop strategy
        csv_data_drop = io.StringIO("col_num,col_cat\n1.0,apple\n,banana\n3.0,\n")
        df_drop, summary_drop, err_drop = analyze_dataset(csv_data_drop, strategy="drop")
        self.assertIsNone(err_drop)
        self.assertEqual(len(df_drop), 1)
        self.assertEqual(df_drop.iloc[0]["col_num"], 1.0)
        self.assertEqual(df_drop.iloc[0]["col_cat"], "apple")

        # Test dictionary strategy mapping
        csv_data_dict = io.StringIO("col_num,col_cat\n10.0,apple\n,banana\n20.0,\n")
        df_dict, summary_dict, err_dict = analyze_dataset(
            csv_data_dict,
            strategy={"col_num": "mean", "col_cat": "mode"},
        )
        self.assertIsNone(err_dict)
        self.assertEqual(len(df_dict), 3)
        self.assertEqual(df_dict.loc[1, "col_num"], 15.0)

    def test_correlations(self):
        corr = compute_correlations(self.df)
        self.assertIn("feature_a", corr)
        self.assertIn("feature_b", corr)
        self.assertAlmostEqual(corr["feature_a"]["feature_a"], 1.0, places=2)

    def test_skewness_and_kurtosis(self):
        skew, kurt, s_desc, k_desc = compute_skewness_and_kurtosis(self.df)
        self.assertIn("feature_a", skew)
        self.assertIn("feature_b", skew)
        self.assertIn("feature_a", kurt)
        self.assertIn("feature_b", kurt)
        self.assertIn("feature_a", s_desc)
        self.assertIn("feature_b", k_desc)

    def test_interpretations(self):
        self.assertEqual(interpret_skewness(0.1), "Approximately Symmetric")
        self.assertEqual(interpret_skewness(0.8), "Moderately Skewed (Positive)")
        self.assertEqual(interpret_skewness(1.5), "Highly Skewed (Positive)")
        self.assertEqual(interpret_skewness(-1.5), "Highly Skewed (Negative)")

        self.assertEqual(interpret_kurtosis(0.1), "Mesokurtic (Normal-like tails)")
        self.assertEqual(interpret_kurtosis(1.2), "Leptokurtic (Heavy tails / Outlier-prone)")
        self.assertEqual(interpret_kurtosis(-1.0), "Platykurtic (Light tails / Flat)")

    def test_iqr_outlier_detection(self):
        iqr_outliers = detect_outliers_iqr(self.df)
        self.assertIn("feature_a", iqr_outliers)
        self.assertGreater(iqr_outliers["feature_a"]["count"], 0)
        self.assertIn(0, iqr_outliers["feature_a"]["outlier_indices"])
        self.assertIn(1, iqr_outliers["feature_a"]["outlier_indices"])

    def test_zscore_outlier_detection(self):
        z_outliers = detect_outliers_zscore(self.df, threshold=3.0)
        self.assertIn("feature_a", z_outliers)
        self.assertGreater(z_outliers["feature_a"]["count"], 0)
        self.assertIn(0, z_outliers["feature_a"]["outlier_indices"])

    def test_isolation_forest_outlier_detection(self):
        iso_outliers = detect_outliers_isolation_forest(self.df)
        self.assertIn("count", iso_outliers)
        self.assertIn("percentage", iso_outliers)
        self.assertIn("outlier_indices", iso_outliers)
        self.assertGreater(iso_outliers["count"], 0)

    def test_error_handling_empty_dataset(self):
        empty_csv = io.StringIO("")
        df, summary, err = analyze_dataset(empty_csv)
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    def test_corrupted_input_handling(self):
        invalid_input = None
        df, summary, err = analyze_dataset(invalid_input)
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    def test_detect_column_types_all_modalities(self):
        test_df = pd.DataFrame({
            "continuous_int": [10, 20, 30, 40, 50],
            "continuous_float": [1.5, 2.7, 3.8, 4.2, 5.9],
            "categorical_str": ["Low", "Medium", "High", "Low", "Medium"],
            "categorical_bool": [True, False, True, True, False],
            "categorical_binary_num": [0, 1, 1, 0, 1],
            "datetime_str": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
            "datetime_parsed": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"]),
            "text_sentences": [
                "This is a long review of an excellent product.",
                "Customer service was somewhat slow and unhelpful today.",
                "Fast shipping and great packaging overall.",
                "Would definitely recommend this to friends and family.",
                "Decent quality for the price point offered.",
            ],
        })
        types = detect_column_types(test_df)
        self.assertEqual(types["continuous_int"], "continuous")
        self.assertEqual(types["continuous_float"], "continuous")
        self.assertEqual(types["categorical_str"], "categorical")
        self.assertEqual(types["categorical_bool"], "categorical")
        self.assertEqual(types["categorical_binary_num"], "categorical")
        self.assertEqual(types["datetime_str"], "datetime")
        self.assertEqual(types["datetime_parsed"], "datetime")
        self.assertEqual(types["text_sentences"], "text")

    def test_detect_column_types_empty_and_na(self):
        empty_df = pd.DataFrame({
            "empty_num": pd.Series([], dtype=float),
            "empty_str": pd.Series([], dtype=object),
        })
        types = detect_column_types(empty_df)
        self.assertEqual(types["empty_num"], "continuous")
        self.assertEqual(types["empty_str"], "text")

        all_na_df = pd.DataFrame({
            "all_na_num": pd.Series([np.nan, np.nan], dtype=float),
            "all_na_str": pd.Series([None, np.nan], dtype=object),
        })
        na_types = detect_column_types(all_na_df)
        self.assertEqual(na_types["all_na_num"], "continuous")
        self.assertEqual(na_types["all_na_str"], "text")

    def test_impute_missing_values_global_mean(self):
        df_missing = pd.DataFrame({
            "num_a": [10.0, 20.0, np.nan, 30.0],
            "cat_b": ["apple", "banana", "apple", np.nan],
        })
        imputed = impute_missing_values(df_missing, strategy="mean")
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        self.assertAlmostEqual(imputed.loc[2, "num_a"], 20.0)
        self.assertEqual(imputed.loc[3, "cat_b"], "apple")

    def test_impute_missing_values_global_median(self):
        df_missing = pd.DataFrame({
            "num_a": [10.0, 20.0, 100.0, np.nan],
            "cat_b": ["red", "blue", "red", np.nan],
        })
        # median of [10, 20, 100] is 20.0
        imputed = impute_missing_values(df_missing, strategy="median")
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        self.assertAlmostEqual(imputed.loc[3, "num_a"], 20.0)
        self.assertEqual(imputed.loc[3, "cat_b"], "red")

    def test_impute_missing_values_global_mode(self):
        df_missing = pd.DataFrame({
            "num_a": [5.0, 5.0, 10.0, np.nan],
            "cat_b": ["dog", "cat", "dog", np.nan],
        })
        imputed = impute_missing_values(df_missing, strategy="mode")
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        self.assertEqual(imputed.loc[3, "num_a"], 5.0)
        self.assertEqual(imputed.loc[3, "cat_b"], "dog")

    def test_impute_missing_values_global_zero(self):
        df_missing = pd.DataFrame({
            "num_a": [1.0, 2.0, np.nan, 4.0],
            "cat_b": ["apple", "banana", "apple", np.nan],
        })
        imputed = impute_missing_values(df_missing, strategy="zero")
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        self.assertAlmostEqual(imputed.loc[2, "num_a"], 0.0)
        self.assertEqual(imputed.loc[3, "cat_b"], "apple")

    def test_impute_missing_values_global_forward_fill(self):
        df_missing = pd.DataFrame({
            "num_a": [np.nan, 20.0, np.nan, 40.0],
            "cat_b": ["apple", np.nan, "cherry", np.nan],
        })
        imputed = impute_missing_values(df_missing, strategy="forward_fill")
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        # Leading NaN in num_a is backfilled from index 1 (20.0)
        self.assertEqual(imputed.loc[0, "num_a"], 20.0)
        # Index 2 in num_a is forward-filled from index 1 (20.0)
        self.assertEqual(imputed.loc[2, "num_a"], 20.0)
        # Index 1 in cat_b is forward-filled from index 0 ("apple")
        self.assertEqual(imputed.loc[1, "cat_b"], "apple")
        # Index 3 in cat_b is forward-filled from index 2 ("cherry")
        self.assertEqual(imputed.loc[3, "cat_b"], "cherry")

    def test_impute_missing_values_global_random_draw(self):
        df_missing = pd.DataFrame({
            "num_a": [10.0, 20.0, 30.0, np.nan, np.nan],
            "cat_b": ["alpha", "beta", np.nan, "alpha", np.nan],
        })
        imputed_1 = impute_missing_values(df_missing, strategy="random_draw", seed=42)
        imputed_2 = impute_missing_values(df_missing, strategy="random_draw", seed=42)
        self.assertEqual(imputed_1.isnull().sum().sum(), 0)
        # Reproducibility with seed
        pd.testing.assert_frame_equal(imputed_1, imputed_2)
        # Imputed values must be sampled from observed values
        self.assertTrue(imputed_1.loc[3, "num_a"] in [10.0, 20.0, 30.0])
        self.assertTrue(imputed_1.loc[4, "num_a"] in [10.0, 20.0, 30.0])
        self.assertTrue(imputed_1.loc[2, "cat_b"] in ["alpha", "beta"])
        self.assertTrue(imputed_1.loc[4, "cat_b"] in ["alpha", "beta"])

    def test_impute_missing_values_default_seed_determinism(self):
        """impute_missing_values defaults to seed=42 ensuring deterministic random draw across calls."""
        df_missing = pd.DataFrame({
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan],
            "cat": ["x", "y", "z", np.nan, np.nan, np.nan, "x", "y"],
        })
        # Global strategy without explicit seed parameter
        run_1 = impute_missing_values(df_missing, strategy="random_draw")
        run_2 = impute_missing_values(df_missing, strategy="random_draw")
        pd.testing.assert_frame_equal(run_1, run_2)
        self.assertEqual(run_1.isnull().sum().sum(), 0)

        # Per-column strategy without explicit seed parameter
        dict_run_1 = impute_missing_values(df_missing, strategy={"val": "random_draw", "cat": "random_draw"})
        dict_run_2 = impute_missing_values(df_missing, strategy={"val": "random_draw", "cat": "random_draw"})
        pd.testing.assert_frame_equal(dict_run_1, dict_run_2)
        self.assertEqual(dict_run_1.isnull().sum().sum(), 0)

        # Default run must match explicit seed=42
        explicit_run = impute_missing_values(df_missing, strategy="random_draw", seed=42)
        pd.testing.assert_frame_equal(run_1, explicit_run)

    def test_impute_missing_values_global_drop(self):
        df_missing = pd.DataFrame({
            "num_a": [1.0, 2.0, np.nan, 4.0],
            "cat_b": ["a", "b", "c", np.nan],
        })
        imputed = impute_missing_values(df_missing, strategy="drop")
        self.assertEqual(len(imputed), 2)
        self.assertEqual(imputed.isnull().sum().sum(), 0)
        self.assertEqual(list(imputed["num_a"]), [1.0, 2.0])

    def test_impute_missing_values_per_column_dict(self):
        df_missing = pd.DataFrame({
            "mean_col": [10.0, np.nan, 30.0, 40.0],
            "median_col": [1.0, 2.0, 100.0, np.nan],
            "mode_col": ["alpha", "beta", "alpha", np.nan],
            "drop_col": [100, 200, np.nan, 400],
        })
        strategy = {
            "mean_col": "mean",
            "median_col": "median",
            "mode_col": "mode",
            "drop_col": "drop",
        }
        imputed = impute_missing_values(df_missing, strategy=strategy)
        # Row 2 had NaN in drop_col, so it should be dropped
        # Remaining rows: indices 0, 1, 3 originally -> now 3 rows
        self.assertEqual(len(imputed), 3)
        self.assertEqual(imputed.isnull().sum().sum(), 0)

    def test_impute_missing_values_unspecified_fallback(self):
        df_missing = pd.DataFrame({
            "configured_col": [10.0, np.nan, 30.0],
            "unconfigured_col": ["x", "y", np.nan],
        })
        strategy = {"configured_col": "mean"}
        imputed = impute_missing_values(df_missing, strategy=strategy, default_unspecified="mode")
        self.assertEqual(imputed.isnull().sum().sum(), 0)

    def test_impute_missing_values_domain_errors(self):
        df_test = pd.DataFrame({
            "num": [1.0, 2.0, np.nan],
            "cat": ["a", "b", np.nan],
        })
        # Invalid strategy string
        with self.assertRaises(ValueError):
            impute_missing_values(df_test, strategy="invalid_strategy")

        # Mean applied to non-numeric column
        with self.assertRaises(ValueError):
            impute_missing_values(df_test, strategy={"cat": "mean"})

        # Median applied to non-numeric column
        with self.assertRaises(ValueError):
            impute_missing_values(df_test, strategy={"cat": "median"})

        # Zero applied to non-numeric column
        with self.assertRaises(ValueError):
            impute_missing_values(df_test, strategy={"cat": "zero"})

        # Column not found in DataFrame
        with self.assertRaises(ValueError):
            impute_missing_values(df_test, strategy={"non_existent": "drop"})

        # Invalid strategy type
        with self.assertRaises(TypeError):
            impute_missing_values(df_test, strategy=123)  # type: ignore[arg-type]

        # Dropping that results in empty dataset
        all_na_df = pd.DataFrame({"a": [np.nan, np.nan]})
        with self.assertRaises(ValueError):
            impute_missing_values(all_na_df, strategy="drop")

    def test_generate_summary_contains_column_types(self):
        df_sample = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["A", "B", "C"],
        })
        summary = generate_summary(df_sample)
        self.assertIn("column_types", summary)
        self.assertEqual(summary["column_types"]["num"], "continuous")
        self.assertEqual(summary["column_types"]["cat"], "categorical")

    def test_analyze_dataset_summary_includes_column_types(self):
        df_clean, summary, err = analyze_dataset(self.csv_file)
        self.assertIsNone(err)
        self.assertIsNotNone(summary)
        self.assertIn("column_types", summary)
        self.assertIn("feature_a", summary["column_types"])
        self.assertEqual(summary["column_types"]["feature_a"], "continuous")
        self.assertEqual(summary["column_types"]["category"], "categorical")

    def test_generate_summary_caching_decorator(self):
        """generate_summary should be callable and return consistent summary dict across invocations."""
        df_sample = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
        })
        summary_1 = generate_summary(df_sample)
        summary_2 = generate_summary(df_sample)
        self.assertEqual(summary_1["rows"], summary_2["rows"])

    def test_benchmark_optimal_strategy_parity_with_impute_missing_values(self):
        """benchmark_imputation_drift optimal_strategy must be directly executable by impute_missing_values."""
        # Run benchmark on sample dataset
        bench_result = benchmark_imputation_drift(self.df, missing_rate=0.15, seed=42)
        optimal = bench_result["optimal_strategy"]
        self.assertIsInstance(optimal, str)
        self.assertIn(
            optimal,
            ["mean", "median", "mode", "zero", "forward_fill", "random_draw"],
        )

        # Create a DataFrame with missing values
        dirty_df = self.df.copy()
        dirty_df.loc[0:10, "feature_a"] = np.nan
        dirty_df.loc[5:15, "category"] = np.nan

        # Applying the optimal strategy must execute cleanly without raising ValueError
        imputed_df = impute_missing_values(dirty_df, strategy=optimal)
        self.assertEqual(imputed_df.isnull().sum().sum(), 0)
        self.assertEqual(len(imputed_df), len(dirty_df))

        # Furthermore, all 6 benchmarked strategies must be directly supported
        leaderboard = bench_result["leaderboard"]
        for entry in leaderboard:
            strat_name = entry["strategy"]
            imputed_strat = impute_missing_values(dirty_df, strategy=strat_name)
            self.assertEqual(imputed_strat.isnull().sum().sum(), 0)
            self.assertEqual(len(imputed_strat), len(dirty_df))

    def test_extended_imputation_strategies_edge_cases(self):
        """Edge cases: all-NaN columns, single-value columns, and space-formatted strategy names."""
        all_na_numeric = pd.DataFrame({"num": [np.nan, np.nan, np.nan]})
        all_na_str = pd.DataFrame({"text": pd.Series([np.nan, np.nan, np.nan], dtype=object)})

        # Forward fill on all-NaN numeric
        imp_ff_num = impute_missing_values(all_na_numeric, strategy="forward fill")
        self.assertEqual(imp_ff_num.isnull().sum().sum(), 0)
        self.assertEqual(imp_ff_num.loc[0, "num"], 0.0)

        # Forward fill on all-NaN string
        imp_ff_str = impute_missing_values(all_na_str, strategy="forward_fill")
        self.assertEqual(imp_ff_str.isnull().sum().sum(), 0)
        self.assertEqual(imp_ff_str.loc[0, "text"], "Unknown")

        # Random draw on all-NaN numeric
        imp_rd_num = impute_missing_values(all_na_numeric, strategy="random draw")
        self.assertEqual(imp_rd_num.isnull().sum().sum(), 0)
        self.assertEqual(imp_rd_num.loc[0, "num"], 0.0)

        # Random draw on all-NaN string
        imp_rd_str = impute_missing_values(all_na_str, strategy="random_draw")
        self.assertEqual(imp_rd_str.isnull().sum().sum(), 0)
        self.assertEqual(imp_rd_str.loc[0, "text"], "Unknown")

        # Single observed value with missing values
        single_val_df = pd.DataFrame({"val": [42.0, np.nan, np.nan]})
        imp_single_ff = impute_missing_values(single_val_df, strategy="forward_fill")
        self.assertEqual(imp_single_ff.isnull().sum().sum(), 0)
        self.assertTrue((imp_single_ff["val"] == 42.0).all())

        imp_single_rd = impute_missing_values(single_val_df, strategy="random_draw", seed=1)
        self.assertEqual(imp_single_rd.isnull().sum().sum(), 0)
        self.assertTrue((imp_single_rd["val"] == 42.0).all())

    def test_analyzer_has_zero_streamlit_dependency(self):
        """analyzer module must have zero import or dependency on streamlit (separation of concerns)."""
        import analyzer

        self.assertFalse(hasattr(analyzer, "st"))
        with open(analyzer.__file__, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotIn("streamlit", content)

    def test_load_dataset_success(self):
        """load_dataset should parse valid tabular dataset and detect column types."""
        csv_bytes = b"col1,col2\n1.0,10\n2.0,20\n"
        df, summary, err = load_dataset(csv_bytes)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)
        self.assertEqual(len(df), 2)
        self.assertIn("column_types", summary)
        self.assertEqual(summary["column_types"]["col1"], "continuous")

    def test_load_dataset_empty_file(self):
        """load_dataset should return domain error on empty input bytes or streams."""
        df, summary, err = load_dataset(b"")
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertEqual(err, "The uploaded CSV file is empty.")

    def test_load_dataset_delegates_to_analyze_dataset(self):
        """load_dataset should delegate to analyze_dataset without generating full summary."""
        from unittest.mock import patch
        csv_bytes = b"x,y\n1,2\n3,4\n"
        with patch("analyzer.generate_summary") as mock_gen_summary:
            df, summary, err = load_dataset(csv_bytes)
            mock_gen_summary.assert_not_called()
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIn("column_types", summary)
        self.assertNotIn("dependency_graph", summary)

    def test_load_dataset_normalizes_infinite_values_without_nans(self):
        """load_dataset must normalize +/-inf to NaN even when no explicit NaNs exist."""
        csv_bytes = b"feature_a,feature_b\n10.0,1.0\n20.0,inf\n30.0,-inf\n40.0,4.0\n"
        df, summary, err = load_dataset(csv_bytes)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)
        # feature_b had inf and -inf, must be normalized to NaN
        self.assertFalse(np.isinf(df["feature_b"]).any())
        self.assertEqual(int(df["feature_b"].isnull().sum()), 2)
        self.assertTrue(df["feature_b"].isnull().any())

    def test_load_dataset_coerces_continuous_object_columns_without_nans(self):
        """load_dataset must coerce continuous object columns so invalid tokens become NaNs."""
        rows = [f"{i}.0" for i in range(1, 10)] + ["corrupted_token"]
        csv_str = "num_col\n" + "\n".join(rows) + "\n"
        df, summary, err = load_dataset(csv_str.encode("utf-8"))
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(df["num_col"]))
        self.assertEqual(int(df["num_col"].isnull().sum()), 1)
        self.assertTrue(df["num_col"].isnull().any())


    def test_impute_missing_values_normalizes_infinite_values(self):
        """impute_missing_values must normalize +/-inf to NaN and impute them cleanly without RuntimeWarnings."""
        df_inf = pd.DataFrame({
            "val": [10.0, 20.0, np.inf, -np.inf, np.nan],
        })

        # 1. Global Mean: mean of [10.0, 20.0] is 15.0; infinite values become NaNs and filled with 15.0
        imputed_mean = impute_missing_values(df_inf, strategy="mean")
        self.assertFalse(np.isneginf(imputed_mean["val"]).any())
        self.assertFalse(np.isposinf(imputed_mean["val"]).any())
        self.assertEqual(int(imputed_mean["val"].isnull().sum()), 0)
        self.assertEqual(list(imputed_mean["val"]), [10.0, 20.0, 15.0, 15.0, 15.0])

        # 2. Global Drop: rows with inf, -inf, or nan must be dropped
        imputed_drop = impute_missing_values(df_inf, strategy="drop")
        self.assertEqual(len(imputed_drop), 2)
        self.assertEqual(list(imputed_drop["val"]), [10.0, 20.0])

        # 3. Dict Strategy
        imputed_dict = impute_missing_values(df_inf, strategy={"val": "median"})
        self.assertEqual(int(imputed_dict["val"].isnull().sum()), 0)
        self.assertEqual(list(imputed_dict["val"]), [10.0, 20.0, 15.0, 15.0, 15.0])

    def test_analyze_dataset_with_infinite_csv_values(self):
        """analyze_dataset must parse CSVs with inf / -inf and produce finite summary statistics."""
        csv_data = b"feature_a,feature_b\n10.0,1.0\n20.0,inf\n30.0,-inf\n40.0,4.0\n"
        df, summary, err = analyze_dataset(csv_data)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)
        # Verify no inf or nan values exist in the cleaned DataFrame
        self.assertFalse(np.isinf(df.to_numpy(dtype=float)).any())
        self.assertEqual(int(df.isnull().sum().sum()), 0)
        # Verify outlier bounds are finite numbers and not NaN / inf
        iqr_b = summary["outliers"]["iqr"]["feature_b"]
        self.assertFalse(np.isnan(iqr_b["lower_bound"]))
        self.assertFalse(np.isnan(iqr_b["upper_bound"]))
        self.assertFalse(np.isinf(iqr_b["lower_bound"]))
        self.assertFalse(np.isinf(iqr_b["upper_bound"]))


    def test_impute_missing_values_coerces_continuous_object_columns(self):
        """impute_missing_values must coerce detected continuous object columns so invalid tokens become NaNs and get imputed."""
        df_mixed = pd.DataFrame({
            "continuous_str": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "corrupted_val"],
            "categorical_str": ["cat", "dog", "bird", "fish", "dog", "cat", "bird", "cat", "dog", None],
        })

        # Global mean: continuous_str coerced to float64, 'corrupted_val' coerced to NaN and filled with 50.0
        imputed_mean = impute_missing_values(df_mixed, strategy="mean")
        self.assertTrue(pd.api.types.is_numeric_dtype(imputed_mean["continuous_str"]))
        self.assertEqual(int(imputed_mean["continuous_str"].isnull().sum()), 0)
        self.assertEqual(imputed_mean["continuous_str"].iloc[-1], 50.0)
        # categorical_str remains object
        self.assertFalse(pd.api.types.is_numeric_dtype(imputed_mean["categorical_str"]))

        # Per-column strategy: continuous_str accepts numeric strategies like 'median'
        imputed_median = impute_missing_values(df_mixed, strategy={"continuous_str": "median", "categorical_str": "mode"})
        self.assertTrue(pd.api.types.is_numeric_dtype(imputed_median["continuous_str"]))
        self.assertEqual(imputed_median["continuous_str"].iloc[-1], 50.0)

        # Categorical column still rejects numeric strategies with ValueError
        with self.assertRaises(ValueError):
            impute_missing_values(df_mixed, strategy={"categorical_str": "mean"})

    def test_analyze_dataset_coerces_continuous_and_includes_in_downstream_metrics(self):
        """analyze_dataset must coerce continuous object columns and retain them across correlation and outlier analyses."""
        csv_rows = [
            "num_str,pure_num\n",
            "10.0,1\n",
            "20.0,2\n",
            "30.0,3\n",
            "40.0,4\n",
            "50.0,5\n",
            "60.0,6\n",
            "70.0,7\n",
            "80.0,8\n",
            "90.0,9\n",
            "corrupted_token,10\n",
        ]
        csv_bytes = "".join(csv_rows).encode("utf-8")
        df, summary, err = analyze_dataset(csv_bytes)

        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)

        # Verify df column dtype is numeric and clean
        self.assertTrue(pd.api.types.is_numeric_dtype(df["num_str"]))
        self.assertEqual(int(df["num_str"].isnull().sum()), 0)

        # Verify summary stats include numeric stats (mean, std) for num_str
        self.assertIn("num_str", summary["stats"])
        self.assertIn("mean", summary["stats"]["num_str"])

        # Verify correlation includes num_str
        self.assertIn("num_str", summary["correlation"])
        self.assertIn("pure_num", summary["correlation"]["num_str"])

        # Verify outlier detection includes num_str
        self.assertIn("num_str", summary["outliers"]["iqr"])
        self.assertIn("num_str", summary["outliers"]["zscore"])

    def test_inf_and_continuous_coercion_edge_cases(self):
        """Edge cases: string 'inf'/'-inf' tokens in continuous columns, all-inf columns, and non-standard indices."""
        # 1. Object column with numeric strings and literal 'inf'/' -inf' tokens
        df_str_inf = pd.DataFrame({
            "col": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "inf", "-inf"]
        })
        res1 = impute_missing_values(df_str_inf, strategy="mean")
        self.assertTrue(pd.api.types.is_numeric_dtype(res1["col"]))
        self.assertEqual(res1["col"].tolist(), [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 50.0, 50.0])
        self.assertFalse(np.isinf(res1["col"]).any())

        # 2. All-inf column with mean strategy fills with fallback 0.0
        df_all_inf = pd.DataFrame({"a": [np.inf, -np.inf, np.inf]})
        res2 = impute_missing_values(df_all_inf, strategy="mean")
        self.assertEqual(res2["a"].tolist(), [0.0, 0.0, 0.0])

        # 3. All-inf column with drop strategy raises domain ValueError (Law 2 / Principle 1)
        with self.assertRaises(ValueError):
            impute_missing_values(df_all_inf, strategy="drop")

        # 4. Non-standard DataFrame index is preserved after coercion and imputation
        df_custom_idx = pd.DataFrame(
            {"feat": ["10", "20", "30", "40", "50", "60", "70", "80", "90", "err"]},
            index=[100 + i for i in range(10)],
        )
        res4 = impute_missing_values(df_custom_idx, strategy="median")
        self.assertEqual(res4.index.tolist(), [100 + i for i in range(10)])
        self.assertTrue(pd.api.types.is_numeric_dtype(res4["feat"]))
        self.assertEqual(int(res4["feat"].isnull().sum()), 0)


if __name__ == "__main__":
    unittest.main()

