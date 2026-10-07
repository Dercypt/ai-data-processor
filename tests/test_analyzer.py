import io
import unittest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analyzer import (
    analyze_dataset,
    compute_correlations,
    compute_skewness_and_kurtosis,
    detect_column_types,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    generate_summary,
    interpret_skewness,
    interpret_kurtosis,
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
        self.assertEqual(df_clean.loc[1, "col_num"], 0.0)
        self.assertEqual(df_clean.loc[2, "col_cat"], "Unknown")

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


if __name__ == "__main__":
    unittest.main()
