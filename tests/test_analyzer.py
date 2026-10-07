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
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
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


if __name__ == "__main__":
    unittest.main()
