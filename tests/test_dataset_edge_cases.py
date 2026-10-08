"""
Automated unit tests for dataset edge cases (empty CSVs, corrupted headers, non-numeric values).
Compatible with pytest and Python unittest runners.
Adheres to system invariants in LAWS.md and PRINCIPLES.md.
"""

import io
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analyzer import (
    analyze_dataset,
    compute_correlations,
    compute_skewness_and_kurtosis,
    detect_column_types,
    detect_outliers_iqr,
    detect_outliers_isolation_forest,
    detect_outliers_zscore,
    generate_summary,
    impute_missing_values,
)


class TestDatasetEdgeCases(unittest.TestCase):
    """
    Test suite for dataset ingestion edge cases.
    Can be run via `pytest` or `python3 -m unittest discover`.
    """

    # =========================================================================
    # Edge Case Group 1: Empty CSVs
    # =========================================================================

    def test_empty_csv_zero_bytes(self):
        """A 0-byte completely empty CSV must return error containment without throwing."""
        empty_file = io.StringIO("")
        df, summary, err = analyze_dataset(empty_file)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)
        self.assertIn("No columns to parse from file", str(err))

    def test_empty_csv_headers_only_zero_rows(self):
        """A CSV with column headers but zero data rows must return an empty dataset error."""
        header_only_file = io.StringIO("id,name,value,timestamp\n")
        df, summary, err = analyze_dataset(header_only_file)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)
        self.assertIn("The uploaded CSV file is empty.", str(err))

    def test_empty_csv_whitespace_and_blank_lines(self):
        """A CSV containing only spaces, tabs, and blank newlines must return error containment."""
        whitespace_file = io.StringIO("   \n\n\t  \n   \n")
        df, summary, err = analyze_dataset(whitespace_file)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    def test_empty_csv_none_input(self):
        """Passing None instead of a valid file stream must return error containment (Law 3)."""
        df, summary, err = analyze_dataset(None)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    def test_empty_csv_empty_bytes_stream(self):
        """Passing an empty BytesIO stream must return error containment."""
        empty_bytes = io.BytesIO(b"")
        df, summary, err = analyze_dataset(empty_bytes)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    # =========================================================================
    # Edge Case Group 2: Corrupted Headers & Formatting
    # =========================================================================

    def test_corrupted_headers_jagged_inconsistent_columns(self):
        """Inconsistent field counts per row (parser error) must be contained gracefully."""
        corrupted_csv = io.StringIO("col1,col2\n1,2\n1,2,3,4,5\n6,7\n")
        df, summary, err = analyze_dataset(corrupted_csv)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)
        self.assertIn("Error tokenizing data", str(err))

    def test_corrupted_headers_unclosed_quotes(self):
        """Unclosed quotes in CSV headers or records must be contained without crashing."""
        unclosed_quotes_csv = io.StringIO('"col_a,col_b\n1,2\n3,4\n')
        df, summary, err = analyze_dataset(unclosed_quotes_csv)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    def test_corrupted_headers_duplicate_column_names(self):
        """CSVs with duplicated column headers must be parsed and processed cleanly."""
        duplicate_headers_csv = io.StringIO("metric,metric,metric\n10,20,30\n40,50,60\n")
        df, summary, err = analyze_dataset(duplicate_headers_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)

        # Law 2: Row count & column structure preservation
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 3)
        self.assertEqual(df.isnull().sum().sum(), 0)

        # Summary statistics should be populated
        self.assertEqual(summary["rows"], 2)
        self.assertEqual(len(summary["columns"]), 3)

    def test_corrupted_headers_whitespace_and_special_symbols(self):
        """Headers with leading/trailing whitespaces, currency symbols, and emojis."""
        special_headers_csv = io.StringIO(
            "  id  , revenue ($) , #tags, 📊 score \n"
            "1,100.5,alpha,0.92\n"
            "2,250.0,beta,0.85\n"
        )
        df, summary, err = analyze_dataset(special_headers_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 4)

    def test_corrupted_headers_missing_column_names(self):
        """CSVs with empty header entries (e.g. ',,col') must parse without crashing."""
        empty_header_csv = io.StringIO(",,category\n1,10.0,A\n2,20.0,B\n")
        df, summary, err = analyze_dataset(empty_header_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)
        self.assertEqual(len(df), 2)
        self.assertEqual(len(df.columns), 3)

    def test_corrupted_binary_stream_input(self):
        """Invalid binary / non-decodable byte streams must return error containment."""
        bad_binary = io.BytesIO(b"\x00\xff\xfe\x00\x01\x02\xaa\xbbcorrupted")
        df, summary, err = analyze_dataset(bad_binary)

        assert df is None
        assert summary is None
        assert err is not None
        self.assertIsNone(df)
        self.assertIsNone(summary)
        self.assertIsNotNone(err)

    # =========================================================================
    # Edge Case Group 3: Non-Numeric Values & Mixed Types
    # =========================================================================

    def test_non_numeric_values_in_numeric_columns(self):
        """Columns intended for numbers containing corrupted text or mixed strings."""
        mixed_csv = io.StringIO(
            "id,measurement,category\n"
            "1,10.5,apple\n"
            "2,corrupted_val,banana\n"
            "3,N/A,cherry\n"
            "4,30.2,date\n"
        )
        df, summary, err = analyze_dataset(mixed_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)

        # Law 2: No rows dropped, missing values eliminated
        self.assertEqual(len(df), 4)
        self.assertEqual(df.isnull().sum().sum(), 0)

        # Law 3: summary contract
        self.assertIn("columns", summary)
        self.assertIn("rows", summary)
        self.assertIn("column_types", summary)
        self.assertIn("stats", summary)

    def test_all_non_numeric_dataset(self):
        """A tabular dataset with zero numeric columns must process all summaries cleanly."""
        text_csv = io.StringIO(
            "name,city,status,feedback\n"
            "Alice,New York,Active,Great service and quick responses.\n"
            "Bob,London,Pending,Looking forward to the upcoming updates.\n"
            "Charlie,Tokyo,Active,Smooth integration and good UI design.\n"
        )
        df, summary, err = analyze_dataset(text_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)

        # Correlation matrix should be empty for non-numeric dataset
        self.assertEqual(summary["correlation"], {})
        self.assertEqual(summary["skewness"], {})
        self.assertEqual(summary["kurtosis"], {})

        # Outlier counts should be 0
        self.assertEqual(summary["outliers"]["iqr"], {})
        self.assertEqual(summary["outliers"]["zscore"], {})
        self.assertEqual(summary["outliers"]["isolation_forest"]["count"], 0)

        # Column types should classify columns as categorical or text
        types = summary["column_types"]
        self.assertIn(types["name"], ("categorical", "text"))
        self.assertIn(types["city"], ("categorical", "text"))
        self.assertEqual(types["feedback"], "text")

    def test_non_numeric_special_float_strings(self):
        """Values like 'inf', '-inf', 'nan', 'null' must be handled without data loss."""
        special_vals_csv = io.StringIO(
            "metric,label\n"
            "inf,High\n"
            "-inf,Low\n"
            "nan,Unknown\n"
            "null,None\n"
            "100.0,Normal\n"
        )
        df, summary, err = analyze_dataset(special_vals_csv)

        assert err is None
        assert df is not None
        assert summary is not None
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertIsNotNone(summary)

        # Law 2: Complete row count preserved
        self.assertEqual(len(df), 5)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_non_numeric_imputation_domain_errors(self):
        """impute_missing_values must raise explicit domain ValueError on non-numeric mean/median."""
        df_text = pd.DataFrame({
            "str_col": ["alpha", "beta", np.nan],
        })

        # Mean on non-numeric column must raise ValueError (Principle 1)
        with self.assertRaises(ValueError) as ctx_mean:
            impute_missing_values(df_text, strategy={"str_col": "mean"})
        self.assertIn("non-numeric column", str(ctx_mean.exception))

        # Median on non-numeric column must raise ValueError (Principle 1)
        with self.assertRaises(ValueError) as ctx_med:
            impute_missing_values(df_text, strategy={"str_col": "median"})
        self.assertIn("non-numeric column", str(ctx_med.exception))

        # Mode on non-numeric column must succeed cleanly
        imputed_mode = impute_missing_values(df_text, strategy={"str_col": "mode"})
        self.assertEqual(imputed_mode.isnull().sum().sum(), 0)
        self.assertEqual(len(imputed_mode), 3)

    def test_non_numeric_column_type_detection_modalities(self):
        """detect_column_types properly categorizes boolean, dates, text, and numeric."""
        df = pd.DataFrame({
            "num": [1, 2, 3, 4],
            "bool_flag": [True, False, True, False],
            "iso_date": ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"],
            "long_text": [
                "This is the first long sentence for testing.",
                "This is the second long sentence for testing.",
                "This is the third long sentence for testing.",
                "This is the fourth long sentence for testing.",
            ],
        })
        types = detect_column_types(df)
        self.assertEqual(types["num"], "continuous")
        self.assertEqual(types["bool_flag"], "categorical")
        self.assertEqual(types["iso_date"], "datetime")
        self.assertEqual(types["long_text"], "text")


if __name__ == "__main__":
    unittest.main()
