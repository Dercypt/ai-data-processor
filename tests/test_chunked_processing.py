"""
Unit and regression tests for chunked dataset ingestion, streaming cleaning,
and empirical memory footprint evaluation.
Adheres to system invariants in LAWS.md and PRINCIPLES.md.
"""

import io
import os
import sys
import unittest
import unittest.mock
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from analyzer import (
    DEFAULT_CHUNKSIZE,
    analyze_dataset,
    clean_chunk,
    create_synthetic_benchmark_csv,
    detect_column_types as from_analyzer_detect,
    evaluate_memory_footprint,
    load_and_clean_chunked,
    read_csv_chunked,
    stream_clean_dataset,
    stream_clean_to_csv,
)


class TestChunkedProcessing(unittest.TestCase):
    """Test suite for chunked CSV ingestion, streaming cleaning, and memory evaluation."""

    def setUp(self) -> None:
        np.random.seed(42)
        self.num_rows = 120
        self.sample_df = pd.DataFrame({
            "num_feat": np.random.normal(20.0, 5.0, size=self.num_rows),
            "cat_feat": [f"Cat_{i % 3}" for i in range(self.num_rows)],
            "text_feat": [f"sample text entry {i}" for i in range(self.num_rows)],
        })
        # Inject null values to verify Law 2 cleaning
        self.sample_df.loc[0, "num_feat"] = np.nan
        self.sample_df.loc[1, "cat_feat"] = np.nan
        self.sample_df.loc[2, "text_feat"] = np.nan

        self.csv_buffer = io.StringIO()
        self.sample_df.to_csv(self.csv_buffer, index=False)
        self.csv_content = self.csv_buffer.getvalue()

    def _get_stream(self) -> io.StringIO:
        return io.StringIO(self.csv_content)

    def test_clean_chunk_null_elimination_and_preservation(self) -> None:
        """clean_chunk must eliminate missing values while preserving row count (Law 2)."""
        chunk = self.sample_df.iloc[:5].copy()
        expected_mean = float(chunk["num_feat"].mean())
        cleaned = clean_chunk(chunk)

        self.assertEqual(len(cleaned), 5)
        self.assertEqual(list(cleaned.columns), list(self.sample_df.columns))
        self.assertEqual(cleaned.isnull().sum().sum(), 0)
        self.assertAlmostEqual(cleaned.loc[0, "num_feat"], expected_mean, places=4)
        self.assertIn(cleaned.loc[1, "cat_feat"], ["Cat_0", "Cat_1", "Cat_2", "Unknown"])
        self.assertFalse(pd.isna(cleaned.loc[2, "text_feat"]))

    def test_clean_chunk_single_column_type_detection_pass(self) -> None:
        """clean_chunk must not trigger redundant column type detection in impute_missing_values."""
        chunk = pd.DataFrame({
            "val": [10.0, np.nan, 30.0],
            "cat": ["A", "B", None],
        })
        with unittest.mock.patch("analyzer.detect_column_types", wraps=from_analyzer_detect) as mock_detect:
            cleaned = clean_chunk(chunk)
            self.assertEqual(mock_detect.call_count, 1)
            self.assertEqual(int(cleaned.isnull().sum().sum()), 0)

    def test_read_csv_chunked_various_chunk_sizes(self) -> None:
        """read_csv_chunked must reassemble dataset identically across chunk sizes."""
        for cs in [7, 25, 50, 200]:
            stream = self._get_stream()
            df_chunked = read_csv_chunked(stream, chunksize=cs)
            self.assertEqual(len(df_chunked), self.num_rows)
            self.assertEqual(list(df_chunked.columns), list(self.sample_df.columns))

    def test_read_csv_chunked_invalid_arguments(self) -> None:
        """read_csv_chunked must raise explicit ValueError on invalid arguments."""
        with self.assertRaises(ValueError):
            read_csv_chunked(None)

        with self.assertRaises(ValueError):
            read_csv_chunked(self._get_stream(), chunksize=0)

        with self.assertRaises(ValueError):
            read_csv_chunked(self._get_stream(), chunksize=-10)

        with self.assertRaises(ValueError):
            header_only = io.StringIO("col_a,col_b\n")
            read_csv_chunked(header_only)

    def test_stream_clean_dataset_bounded_memory(self) -> None:
        """stream_clean_dataset yields cleaned chunks iteratively with zero nulls."""
        stream = self._get_stream()
        chunk_size = 30
        chunks = list(stream_clean_dataset(stream, chunksize=chunk_size))

        self.assertEqual(len(chunks), 4)  # 120 / 30 = 4 chunks
        total_rows = 0
        for ch in chunks:
            self.assertLessEqual(len(ch), chunk_size)
            self.assertEqual(ch.isnull().sum().sum(), 0)
            total_rows += len(ch)
        self.assertEqual(total_rows, self.num_rows)

    def test_stream_clean_to_csv_output(self) -> None:
        """stream_clean_to_csv must stream-clean to output buffer in O(C) memory."""
        in_stream = self._get_stream()
        out_stream = io.StringIO()
        total_rows = stream_clean_to_csv(in_stream, out_stream, chunksize=25)

        self.assertEqual(total_rows, self.num_rows)
        out_stream.seek(0)
        reloaded = pd.read_csv(out_stream)
        self.assertEqual(len(reloaded), self.num_rows)
        self.assertEqual(reloaded.isnull().sum().sum(), 0)

    def test_load_and_clean_chunked(self) -> None:
        """load_and_clean_chunked returns complete cleaned DataFrame and detected types."""
        stream = self._get_stream()
        df_clean, col_types = load_and_clean_chunked(stream, chunksize=40)

        self.assertEqual(len(df_clean), self.num_rows)
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        self.assertIn("num_feat", col_types)
        self.assertEqual(col_types["num_feat"], "continuous")

    def test_load_and_clean_chunked_none_chunksize(self) -> None:
        """load_and_clean_chunked with chunksize=None cleans dataset identically in structure."""
        stream_chunked = self._get_stream()
        df_chunked, types_chunked = load_and_clean_chunked(stream_chunked, chunksize=30)

        stream_unchunked = self._get_stream()
        df_unchunked, types_unchunked = load_and_clean_chunked(stream_unchunked, chunksize=None)

        self.assertEqual(len(df_chunked), len(df_unchunked))
        self.assertEqual(types_chunked, types_unchunked)
        self.assertEqual(df_chunked.isnull().sum().sum(), 0)
        self.assertEqual(df_unchunked.isnull().sum().sum(), 0)
        self.assertAlmostEqual(df_chunked.loc[0, "num_feat"], df_unchunked.loc[0, "num_feat"], delta=1.5)

    def test_load_and_clean_chunked_invalid_arguments(self) -> None:
        """load_and_clean_chunked raises ValueError on None or empty input."""
        with self.assertRaises(ValueError):
            load_and_clean_chunked(None)
        with self.assertRaises(ValueError):
            load_and_clean_chunked(io.StringIO("col_a,col_b\n"))

    def test_analyze_dataset_with_chunksize(self) -> None:
        """analyze_dataset adheres to Law 2 & Law 3 when chunksize is provided."""
        stream = self._get_stream()
        df_clean, summary, err = analyze_dataset(stream, chunksize=15)

        self.assertIsNone(err)
        self.assertIsNotNone(df_clean)
        self.assertIsNotNone(summary)
        self.assertEqual(len(df_clean), self.num_rows)
        self.assertEqual(summary["rows"], self.num_rows)
        self.assertEqual(summary["columns"], list(self.sample_df.columns))

    def test_analyze_dataset_avoids_chunk_concat_overhead(self) -> None:
        """analyze_dataset should construct in-memory DataFrame directly without calling read_csv_chunked."""
        from unittest.mock import patch
        stream = self._get_stream()
        with patch("analyzer.read_csv_chunked") as mock_chunked:
            df_clean, summary, err = analyze_dataset(stream, chunksize=25)
            mock_chunked.assert_not_called()
        self.assertIsNone(err)
        self.assertIsNotNone(df_clean)
        self.assertEqual(len(df_clean), self.num_rows)

    def test_analyze_dataset_default_chunksize_backward_compatibility(self) -> None:
        """analyze_dataset called with single file argument continues to work seamlessly."""
        stream = self._get_stream()
        df_clean, summary, err = analyze_dataset(stream)

        self.assertIsNone(err)
        self.assertIsNotNone(df_clean)
        self.assertIsNotNone(summary)
        self.assertEqual(len(df_clean), self.num_rows)

    def test_create_synthetic_benchmark_csv(self) -> None:
        """Synthetic benchmark CSV generator produces valid tabular dataset bytes."""
        csv_bytes = create_synthetic_benchmark_csv(n_rows=500)
        self.assertIsInstance(csv_bytes, bytes)
        df_synth = pd.read_csv(io.BytesIO(csv_bytes))
        self.assertEqual(len(df_synth), 500)
        self.assertIn("metric_a", df_synth.columns)
        self.assertIn("segment", df_synth.columns)

    def test_evaluate_memory_footprint_synthetic_dataset(self) -> None:
        """evaluate_memory_footprint runs empirical benchmark and returns structured metrics."""
        bench = evaluate_memory_footprint(
            file_or_data=None,
            chunk_sizes=[200, 500, 1000],
            sample_rows_if_synthetic=1200,
        )

        self.assertIn("dataset_info", bench)
        self.assertIn("baseline_peak_kb", bench)
        self.assertIn("best_streaming_chunk_size", bench)
        self.assertIn("max_memory_reduction_pct", bench)
        self.assertIn("evaluations", bench)

        evals = bench["evaluations"]
        self.assertGreaterEqual(len(evals), 4)  # 1 baseline + 3 chunk sizes
        baseline_eval = evals[0]
        self.assertEqual(baseline_eval["mode"], "Monolithic Unchunked (O(N))")
        self.assertGreater(baseline_eval["peak_memory_bytes"], 0)

        for ev in evals[1:]:
            self.assertIn("peak_memory_kb", ev)
            self.assertIn("elapsed_seconds", ev)
            self.assertIn("throughput_rows_sec", ev)
            self.assertIn("memory_reduction_pct", ev)

    def test_evaluate_memory_footprint_with_custom_stream(self) -> None:
        """evaluate_memory_footprint works with user-supplied in-memory stream."""
        stream = self._get_stream()
        bench = evaluate_memory_footprint(
            file_or_data=stream,
            chunk_sizes=[20, 50],
        )

        self.assertEqual(bench["dataset_info"]["total_rows"], self.num_rows)
        self.assertGreater(bench["baseline_peak_kb"], 0)
        self.assertEqual(len(bench["evaluations"]), 3)

    def test_evaluate_memory_footprint_embedded_newlines_row_count(self) -> None:
        """evaluate_memory_footprint must accurately count rows even with embedded newlines."""
        multiline_df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "text": [
                "normal text",
                "text with\nembedded\nnewlines",
                "another\nmultiline entry",
                "clean record",
                "final\nmulti\nline\nrow",
            ],
            "value": [10.5, 20.0, 31.2, 45.1, 55.0],
        })
        csv_bytes = multiline_df.to_csv(index=False).encode("utf-8")
        bench = evaluate_memory_footprint(csv_bytes, chunk_sizes=[2])

        self.assertEqual(bench["dataset_info"]["total_rows"], 5)
        self.assertEqual(bench["dataset_info"]["total_columns"], 3)

    def test_evaluate_memory_footprint_delegates_to_stream_clean_dataset(self) -> None:
        """evaluate_memory_footprint must delegate streaming loop to stream_clean_dataset."""
        from unittest.mock import patch
        stream = self._get_stream()
        with patch("analyzer.stream_clean_dataset", wraps=stream_clean_dataset) as mock_stream_clean:
            bench = evaluate_memory_footprint(stream, chunk_sizes=[20, 50])
            self.assertEqual(mock_stream_clean.call_count, 2)
        self.assertEqual(len(bench["evaluations"]), 3)

    def test_evaluate_memory_footprint_chunk_size_larger_than_dataset(self) -> None:
        """evaluate_memory_footprint handles chunk size exceeding total row count gracefully."""
        stream = self._get_stream()
        bench = evaluate_memory_footprint(stream, chunk_sizes=[500])
        self.assertEqual(bench["dataset_info"]["total_rows"], self.num_rows)
        streaming_eval = bench["evaluations"][1]
        self.assertEqual(streaming_eval["num_chunks"], 1)

    def test_stream_clean_to_csv_with_embedded_newlines(self) -> None:
        """stream_clean_to_csv preserves records containing embedded newlines and eliminates nulls."""
        multiline_df = pd.DataFrame({
            "id": [1, 2, 3],
            "description": ["item 1", "item 2\nwith\nmultiline", np.nan],
            "value": [10.0, np.nan, 30.0],
        })
        in_buf = io.StringIO()
        multiline_df.to_csv(in_buf, index=False)
        in_buf.seek(0)
        out_buf = io.StringIO()

        total_rows = stream_clean_to_csv(in_buf, out_buf, chunksize=2)
        self.assertEqual(total_rows, 3)

        out_buf.seek(0)
        cleaned_reloaded = pd.read_csv(out_buf)
        self.assertEqual(len(cleaned_reloaded), 3)
        self.assertEqual(cleaned_reloaded.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
