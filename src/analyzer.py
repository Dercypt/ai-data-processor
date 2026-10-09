import io
import math
import os
import time
import tracemalloc
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DEFAULT_CHUNKSIZE: int = 10_000


def compute_correlations(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute Pearson correlation matrix across numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {}
    corr_matrix = numeric_df.corr(method="pearson")
    corr_matrix = corr_matrix.fillna(0.0)
    result: Dict[str, Dict[str, float]] = {}
    for col in corr_matrix.columns:
        result[str(col)] = {str(k): round(float(v), 4) for k, v in corr_matrix[col].to_dict().items()}
    return result


def interpret_skewness(val: float) -> str:
    """Classify skewness into standard statistical categories."""
    if abs(val) < 0.5:
        return "Approximately Symmetric"
    elif 0.5 <= val <= 1.0:
        return "Moderately Skewed (Positive)"
    elif val > 1.0:
        return "Highly Skewed (Positive)"
    elif -1.0 <= val <= -0.5:
        return "Moderately Skewed (Negative)"
    else:
        return "Highly Skewed (Negative)"


def interpret_kurtosis(val: float) -> str:
    """Classify excess kurtosis into standard statistical categories."""
    if abs(val) < 0.5:
        return "Mesokurtic (Normal-like tails)"
    elif val >= 0.5:
        return "Leptokurtic (Heavy tails / Outlier-prone)"
    else:
        return "Platykurtic (Light tails / Flat)"


def compute_skewness_and_kurtosis(
    df: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str], Dict[str, str]]:
    """Compute skewness, kurtosis, and their statistical interpretations for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    skew_dict: Dict[str, float] = {}
    kurt_dict: Dict[str, float] = {}
    skew_interp: Dict[str, str] = {}
    kurt_interp: Dict[str, str] = {}

    for col in numeric_df.columns:
        col_str = str(col)
        series = numeric_df[col]
        s_val = float(series.skew()) if len(series) > 2 else 0.0
        k_val = float(series.kurtosis()) if len(series) > 3 else 0.0
        if np.isnan(s_val):
            s_val = 0.0
        if np.isnan(k_val):
            k_val = 0.0
        skew_dict[col_str] = round(s_val, 4)
        kurt_dict[col_str] = round(k_val, 4)
        skew_interp[col_str] = interpret_skewness(s_val)
        kurt_interp[col_str] = interpret_kurtosis(k_val)

    return skew_dict, kurt_dict, skew_interp, kurt_interp


def detect_outliers_iqr(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, Union[int, float, List[int]]]]:
    """Detect outliers using the Interquartile Range (IQR) method (1.5 * IQR)."""
    numeric_df = df.select_dtypes(include=[np.number])
    iqr_results: Dict[str, Dict[str, Union[int, float, List[int]]]] = {}
    total_rows = len(df)
    if total_rows == 0:
        return iqr_results

    for col in numeric_df.columns:
        col_str = str(col)
        series = numeric_df[col]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices: List[int] = series.index[outlier_mask].tolist()
        count = len(outlier_indices)
        pct = round((count / total_rows) * 100.0, 2)
        iqr_results[col_str] = {
            "count": count,
            "percentage": pct,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "outlier_indices": outlier_indices,
        }
    return iqr_results


def detect_outliers_zscore(
    df: pd.DataFrame, threshold: float = 3.0
) -> Dict[str, Dict[str, Union[int, float, List[int]]]]:
    """Detect outliers using the Z-Score method (|z| > threshold)."""
    numeric_df = df.select_dtypes(include=[np.number])
    zscore_results: Dict[str, Dict[str, Union[int, float, List[int]]]] = {}
    total_rows = len(df)
    if total_rows == 0:
        return zscore_results

    for col in numeric_df.columns:
        col_str = str(col)
        series = numeric_df[col]
        std = float(series.std(ddof=0))
        mean = float(series.mean())
        if std == 0.0 or np.isnan(std):
            zscore_results[col_str] = {
                "count": 0,
                "percentage": 0.0,
                "threshold": threshold,
                "mean": round(mean, 4),
                "std": 0.0,
                "outlier_indices": [],
            }
            continue
        z_scores = ((series - mean) / std).abs()
        outlier_mask = z_scores > threshold
        outlier_indices: List[int] = series.index[outlier_mask].tolist()
        count = len(outlier_indices)
        pct = round((count / total_rows) * 100.0, 2)
        zscore_results[col_str] = {
            "count": count,
            "percentage": pct,
            "threshold": threshold,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "outlier_indices": outlier_indices,
        }
    return zscore_results


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
) -> Dict[str, Union[int, float, List[int]]]:
    """Detect multivariate anomalies using Scikit-Learn Isolation Forest."""
    numeric_df = df.select_dtypes(include=[np.number])
    total_rows = len(df)
    if total_rows < 2 or numeric_df.shape[1] == 0:
        return {
            "count": 0,
            "percentage": 0.0,
            "outlier_indices": [],
        }

    iso = IsolationForest(random_state=42, contamination="auto")
    preds = iso.fit_predict(numeric_df)
    outlier_mask = preds == -1
    outlier_indices: List[int] = df.index[outlier_mask].tolist()
    count = len(outlier_indices)
    pct = round((count / total_rows) * 100.0, 2)
    return {
        "count": count,
        "percentage": pct,
        "outlier_indices": outlier_indices,
    }


def detect_column_types(df: pd.DataFrame) -> Dict[str, str]:
    """
    Auto-detect semantic column types across tabular dataset:
    - 'continuous': numeric features (int, float)
    - 'categorical': low-cardinality nominal/ordinal features, booleans
    - 'datetime': timestamp and date features
    - 'text': free-form high-cardinality unstructured strings
    """
    column_types: Dict[str, str] = {}
    total_rows = len(df)

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        valid_series = series.dropna()

        if valid_series.empty:
            if pd.api.types.is_numeric_dtype(series):
                column_types[col_str] = "continuous"
            else:
                column_types[col_str] = "text"
            continue

        # 1. Check if explicitly datetime or boolean / categorical dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            column_types[col_str] = "datetime"
            continue

        if pd.api.types.is_bool_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            column_types[col_str] = "categorical"
            continue

        # 2. Check if numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            n_unique = int(valid_series.nunique())
            # Binary flag (0 and 1) is typically categorical
            if n_unique <= 2 and set(valid_series.unique()).issubset({0, 1, 0.0, 1.0}):
                column_types[col_str] = "categorical"
            else:
                column_types[col_str] = "continuous"
            continue

        # 3. For object / string series, check if it can be parsed as datetime
        sample = valid_series.head(100).astype(str)
        date_delimiters = {"-", "/", ":", "T"}
        has_date_symbols = bool(
            sample.apply(lambda s: any(delim in s for delim in date_delimiters)).mean() >= 0.8
        )
        if has_date_symbols:
            try:
                parsed_dates = pd.to_datetime(sample, errors="coerce")
                if float(parsed_dates.notna().mean()) >= 0.8:
                    column_types[col_str] = "datetime"
                    continue
            except (ValueError, TypeError):
                pass

        # 4. Check if object series contains numeric strings
        try:
            numeric_parsed = pd.to_numeric(sample, errors="coerce")
            if float(numeric_parsed.notna().mean()) >= 0.9:
                n_unique = int(valid_series.nunique())
                if n_unique <= 2:
                    column_types[col_str] = "categorical"
                else:
                    column_types[col_str] = "continuous"
                continue
        except (ValueError, TypeError):
            pass

        # 5. Distinguish between 'categorical' and 'text'
        n_unique = int(valid_series.nunique())
        cardinality_ratio = (n_unique / total_rows) if total_rows > 0 else 0.0
        sample_words = sample.apply(lambda s: len(s.split()))
        avg_words = float(sample_words.mean())
        max_words = int(sample_words.max())
        avg_char_len = float(sample.apply(len).mean())

        if avg_words > 3.0 or max_words > 10 or avg_char_len > 60.0 or (n_unique > 50 and cardinality_ratio > 0.5):
            column_types[col_str] = "text"
        else:
            column_types[col_str] = "categorical"

    return column_types


def impute_missing_values(
    df: pd.DataFrame,
    strategy: Union[str, Dict[str, str]] = "mean",
    default_unspecified: Optional[str] = None,
) -> pd.DataFrame:
    """
    Impute or clean missing values according to user-configurable strategies.
    Supported strategies:
    - 'mean': Fills missing values with the arithmetic mean (numeric columns only).
    - 'median': Fills missing values with the median (numeric columns only).
    - 'mode': Fills missing values with the most frequent value.
    - 'drop': Drops rows containing missing values in the targeted column(s).

    Parameters:
        df: Input DataFrame.
        strategy: Either a global strategy string ('mean', 'median', 'mode', 'drop')
                  or a column-to-strategy mapping dictionary.
        default_unspecified: Optional fallback strategy for columns not present in the strategy dictionary.

    Returns:
        pd.DataFrame with missing values imputed or dropped according to configuration.
    """
    if df.empty:
        return df.copy()

    valid_strategies = {"mean", "median", "mode", "drop"}

    if isinstance(strategy, str):
        strat_lower = strategy.strip().lower()
        if strat_lower not in valid_strategies:
            raise ValueError(
                f"Invalid imputation strategy '{strategy}'. "
                f"Supported strategies: 'mean', 'median', 'mode', 'drop'."
            )

        result_df = df.copy()
        if strat_lower == "drop":
            result_df = result_df.dropna().reset_index(drop=True)
            if result_df.empty and not df.empty:
                raise ValueError("Dropping missing values resulted in an empty dataset.")
            return result_df

        for col in result_df.columns:
            if not result_df[col].isnull().any():
                continue
            is_numeric = pd.api.types.is_numeric_dtype(result_df[col])
            if strat_lower == "mean":
                if not is_numeric:
                    mode_vals = result_df[col].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                    result_df[col] = result_df[col].fillna(fill_val)
                else:
                    mean_val = result_df[col].mean()
                    num_fill = float(mean_val) if not pd.isna(mean_val) else 0.0
                    result_df[col] = result_df[col].fillna(num_fill)
            elif strat_lower == "median":
                if not is_numeric:
                    mode_vals = result_df[col].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
                    result_df[col] = result_df[col].fillna(fill_val)
                else:
                    med_val = result_df[col].median()
                    num_fill = float(med_val) if not pd.isna(med_val) else 0.0
                    result_df[col] = result_df[col].fillna(num_fill)
            elif strat_lower == "mode":
                mode_vals = result_df[col].mode(dropna=True)
                if not mode_vals.empty:
                    result_df[col] = result_df[col].fillna(mode_vals.iloc[0])
                else:
                    fallback_val = 0.0 if is_numeric else "Unknown"
                    result_df[col] = result_df[col].fillna(fallback_val)

        return result_df

    elif isinstance(strategy, dict):
        result_df = df.copy()
        normalized_strat: Dict[str, str] = {}
        for col_key, strat_val in strategy.items():
            if col_key not in result_df.columns:
                raise ValueError(
                    f"Column '{col_key}' specified in imputation strategy not found in DataFrame columns."
                )
            strat_norm = str(strat_val).strip().lower()
            if strat_norm not in valid_strategies:
                raise ValueError(
                    f"Invalid imputation strategy '{strat_val}' for column '{col_key}'. "
                    f"Supported strategies: 'mean', 'median', 'mode', 'drop'."
                )
            if strat_norm in ("mean", "median") and not pd.api.types.is_numeric_dtype(result_df[col_key]):
                raise ValueError(
                    f"Cannot apply '{strat_norm}' imputation to non-numeric column '{col_key}'. "
                    f"Use 'mode' or 'drop' instead."
                )
            normalized_strat[col_key] = strat_norm

        if default_unspecified is not None:
            default_norm = default_unspecified.strip().lower()
            if default_norm not in valid_strategies:
                raise ValueError(
                    f"Invalid default_unspecified strategy '{default_unspecified}'. "
                    f"Supported strategies: 'mean', 'median', 'mode', 'drop'."
                )
            for col in result_df.columns:
                if col not in normalized_strat:
                    if default_norm in ("mean", "median") and not pd.api.types.is_numeric_dtype(result_df[col]):
                        normalized_strat[col] = "mode"
                    else:
                        normalized_strat[col] = default_norm

        # Step 1: Drop missing rows for drop-designated columns
        drop_cols = [c for c, s in normalized_strat.items() if s == "drop" and result_df[c].isnull().any()]
        if drop_cols:
            result_df = result_df.dropna(subset=drop_cols).reset_index(drop=True)
            if result_df.empty and not df.empty:
                raise ValueError("Dropping missing values resulted in an empty dataset.")

        # Step 2: Apply imputation to remaining configured columns
        for col, strat_norm in normalized_strat.items():
            if strat_norm == "drop" or not result_df[col].isnull().any():
                continue
            is_numeric = pd.api.types.is_numeric_dtype(result_df[col])
            if strat_norm == "mean":
                mean_val = result_df[col].mean()
                num_fill = float(mean_val) if not pd.isna(mean_val) else 0.0
                result_df[col] = result_df[col].fillna(num_fill)
            elif strat_norm == "median":
                med_val = result_df[col].median()
                num_fill = float(med_val) if not pd.isna(med_val) else 0.0
                result_df[col] = result_df[col].fillna(num_fill)
            elif strat_norm == "mode":
                mode_vals = result_df[col].mode(dropna=True)
                if not mode_vals.empty:
                    result_df[col] = result_df[col].fillna(mode_vals.iloc[0])
                else:
                    fallback_val = 0.0 if is_numeric else "Unknown"
                    result_df[col] = result_df[col].fillna(fallback_val)

        return result_df

    else:
        raise TypeError(
            f"Imputation strategy must be a str or Dict[str, str], got {type(strategy).__name__}."
        )


def generate_summary(df: pd.DataFrame) -> Dict[str, object]:
    """Generate statistical summary and column type classifications for a DataFrame."""
    col_types = detect_column_types(df)
    corr = compute_correlations(df)
    skew_vals, kurt_vals, skew_desc, kurt_desc = compute_skewness_and_kurtosis(df)
    iqr_outliers = detect_outliers_iqr(df)
    zscore_outliers = detect_outliers_zscore(df)
    iso_outliers = detect_outliers_isolation_forest(df)
    dependency_graph = build_feature_dependency_graph(df)
    heuristic_anomalies = evaluate_heuristic_rules(df)
    imputation_drift = benchmark_imputation_drift(df)

    summary: Dict[str, object] = {
        "columns": list(df.columns),
        "rows": int(len(df)),
        "stats": df.describe().to_dict(),
        "column_types": col_types,
        "correlation": corr,
        "skewness": skew_vals,
        "kurtosis": kurt_vals,
        "skewness_interpretation": skew_desc,
        "kurtosis_interpretation": kurt_desc,
        "outliers": {
            "iqr": iqr_outliers,
            "zscore": zscore_outliers,
            "isolation_forest": iso_outliers,
            "heuristic_rules": heuristic_anomalies,
        },
        "dependency_graph": dependency_graph,
        "heuristic_rules": heuristic_anomalies,
        "imputation_drift": imputation_drift,
    }
    return summary


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single DataFrame chunk by eliminating missing values without dropping rows (Law 2).
    Numeric features are imputed with 0, non-numeric with 'Unknown'.
    """
    cleaned = chunk.copy()
    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].fillna(0)
        else:
            cleaned[col] = cleaned[col].fillna("Unknown")
    return cleaned


def read_csv_chunked(
    file: object,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> pd.DataFrame:
    """
    Parse CSV tabular data in batch chunks and reassemble into a single DataFrame.
    Preserves full row count and column schemas.

    Note on memory complexity:
    Reassembling chunks into a single pd.DataFrame requires O(N) memory allocation
    for the complete dataset. For strictly bounded O(C) memory consumption where
    the full dataset is never held in memory at once, use stream_clean_dataset()
    or stream_clean_to_csv() instead.
    """
    if file is None:
        raise ValueError("Invalid file path or buffer object type: <class 'NoneType'>")
    if chunksize <= 0:
        raise ValueError(f"chunksize must be a positive integer, got {chunksize}.")

    reader = pd.read_csv(file, chunksize=chunksize)
    chunks: List[pd.DataFrame] = []
    for chunk in reader:
        chunks.append(chunk)

    if not chunks:
        raise ValueError("The uploaded CSV file is empty.")

    df = pd.concat(chunks, ignore_index=True)
    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    return df


def stream_clean_dataset(
    file: object,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream-clean a tabular dataset chunk-by-chunk in strictly bounded O(C) memory.
    Yields each cleaned DataFrame chunk iteratively.
    """
    if file is None:
        raise ValueError("Invalid file path or buffer object type: <class 'NoneType'>")
    if chunksize <= 0:
        raise ValueError(f"chunksize must be a positive integer, got {chunksize}.")

    reader = pd.read_csv(file, chunksize=chunksize)
    for chunk in reader:
        yield clean_chunk(chunk)


def stream_clean_to_csv(
    input_file: object,
    output_file: object,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> int:
    """
    Stream-clean an input CSV and write directly to an output CSV file/buffer in O(C) memory.
    Returns the total number of rows processed.
    """
    total_rows = 0
    first_chunk = True
    for cleaned_chunk in stream_clean_dataset(input_file, chunksize=chunksize):
        cleaned_chunk.to_csv(
            output_file,
            index=False,
            header=first_chunk,
            mode="w" if first_chunk else "a",
        )
        first_chunk = False
        total_rows += len(cleaned_chunk)
    return total_rows


def load_and_clean_chunked(
    file: object,
    chunksize: Optional[int] = DEFAULT_CHUNKSIZE,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load and clean a tabular dataset using iterative chunked processing.
    Eliminates missing values chunk-by-chunk without dropping rows or columns (Law 2).
    Reassembles cleaned chunks into a full DataFrame for downstream in-memory analysis (O(N) memory).

    Returns:
        Tuple of (cleaned_df, detected_column_types).
    """
    if file is None:
        raise ValueError("Invalid file path or buffer object type: <class 'NoneType'>")

    if chunksize is not None and chunksize > 0:
        cleaned_chunks: List[pd.DataFrame] = []
        for cleaned_chunk in stream_clean_dataset(file, chunksize=chunksize):
            cleaned_chunks.append(cleaned_chunk)

        if not cleaned_chunks:
            raise ValueError("The uploaded CSV file is empty.")

        df = pd.concat(cleaned_chunks, ignore_index=True)
    else:
        df = pd.read_csv(file)
        if df.empty:
            raise ValueError("The uploaded CSV file is empty.")
        df = clean_chunk(df)

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    col_types = detect_column_types(df)
    return df, col_types


def analyze_dataset(
    file: object,
    chunksize: Optional[int] = DEFAULT_CHUNKSIZE,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]], Optional[str]]:
    """
    Ingest, clean, and compute comprehensive statistical data science summaries.
    Supports chunked ingestion batching via `chunksize`.
    Adheres strictly to Law 2 (cleaning integrity) and Law 3 (return contract).

    Note on memory complexity:
    The analysis phase and return contract require the full DataFrame in memory (O(N))
    to compute global statistics and enable downstream visualization. For bounded
    O(C) memory operations, use stream_clean_to_csv().
    """
    try:
        df, col_types = load_and_clean_chunked(file, chunksize=chunksize)
        summary = generate_summary(df)
        summary["column_types"] = col_types
        return df, summary, None
    except Exception as e:
        return None, None, str(e)


def create_synthetic_benchmark_csv(n_rows: int = 20_000) -> bytes:
    """
    Generate a synthetic tabular CSV dataset for benchmarking memory footprints across chunk sizes.
    Features numeric, categorical, and text columns with injected missing values.
    """
    np.random.seed(42)
    data = {
        "id": np.arange(n_rows),
        "metric_a": np.random.normal(50.0, 15.0, size=n_rows),
        "metric_b": np.random.exponential(5.0, size=n_rows),
        "segment": [f"Cluster_{i % 8}" for i in range(n_rows)],
        "notes": [f"Transaction observation record {i % 50}" for i in range(n_rows)],
    }
    df = pd.DataFrame(data)
    df.loc[::7, "metric_a"] = np.nan
    df.loc[::11, "segment"] = np.nan
    csv_str = df.to_csv(index=False)
    return csv_str.encode("utf-8")


def _extract_csv_bytes(file_or_data: Optional[object], sample_rows_if_synthetic: int = 20_000) -> bytes:
    """Extract raw CSV byte buffer from various input sources or synthetic fallback."""
    if file_or_data is None:
        return create_synthetic_benchmark_csv(n_rows=sample_rows_if_synthetic)
    if isinstance(file_or_data, bytes):
        return file_or_data
    if isinstance(file_or_data, str):
        if os.path.exists(file_or_data):
            with open(file_or_data, "rb") as f:
                return f.read()
        return file_or_data.encode("utf-8")
    if isinstance(file_or_data, pd.DataFrame):
        return file_or_data.to_csv(index=False).encode("utf-8")
    if hasattr(file_or_data, "getvalue"):
        val = file_or_data.getvalue()
        if isinstance(val, str):
            return val.encode("utf-8")
        elif isinstance(val, bytes):
            return val
    if hasattr(file_or_data, "read"):
        if hasattr(file_or_data, "seek"):
            file_or_data.seek(0)
        content = file_or_data.read()
        if hasattr(file_or_data, "seek"):
            file_or_data.seek(0)
        if isinstance(content, str):
            return content.encode("utf-8")
        elif isinstance(content, bytes):
            return content
    raise TypeError(f"Unsupported file_or_data type: {type(file_or_data).__name__}")


def evaluate_memory_footprint(
    file_or_data: Optional[object] = None,
    chunk_sizes: Optional[List[int]] = None,
    sample_rows_if_synthetic: int = 20_000,
) -> Dict[str, object]:
    """
    Empirically evaluate and compare memory footprints and execution latency across chunk sizes.
    Demonstrates O(N) monolithic space complexity vs O(C) bounded chunked streaming cleaning.
    Utilizes Python standard library tracemalloc for deterministic heap tracking.
    """
    if chunk_sizes is None:
        chunk_sizes = [500, 1000, 5000, 10000]

    csv_bytes = _extract_csv_bytes(file_or_data, sample_rows_if_synthetic)
    inspect_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=5)
    total_rows = sum(1 for _ in io.BytesIO(csv_bytes)) - 1
    total_cols = len(inspect_df.columns)

    results: List[Dict[str, Union[str, int, float, None]]] = []

    # 1. Baseline: Monolithic Unchunked O(N) Ingestion
    tracemalloc.start()
    t0 = time.perf_counter()
    bio_base = io.BytesIO(csv_bytes)
    df_base = pd.read_csv(bio_base)
    for col in df_base.columns:
        if pd.api.types.is_numeric_dtype(df_base[col]):
            df_base[col] = df_base[col].fillna(0)
        else:
            df_base[col] = df_base[col].fillna("Unknown")
    t1 = time.perf_counter()
    _, peak_base_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    baseline_kb = round(peak_base_bytes / 1024.0, 2)
    baseline_mb = round(peak_base_bytes / (1024.0 * 1024.0), 3)
    baseline_time = round(t1 - t0, 4)
    baseline_throughput = round(total_rows / baseline_time, 1) if baseline_time > 0 else 0.0

    results.append({
        "mode": "Monolithic Unchunked (O(N))",
        "chunk_size": None,
        "chunk_size_label": "None (Full O(N))",
        "peak_memory_bytes": peak_base_bytes,
        "peak_memory_kb": baseline_kb,
        "peak_memory_mb": baseline_mb,
        "elapsed_seconds": baseline_time,
        "throughput_rows_sec": baseline_throughput,
        "memory_reduction_pct": 0.0,
        "num_chunks": 1,
    })

    best_streaming_chunk_size = chunk_sizes[0]
    max_memory_reduction_pct = 0.0

    # 2. Evaluate Pure Stream Cleaning O(C) across chunk sizes
    for cs in chunk_sizes:
        if cs <= 0:
            continue
        tracemalloc.start()
        t0_stream = time.perf_counter()
        bio_stream = io.BytesIO(csv_bytes)
        stream_reader = pd.read_csv(bio_stream, chunksize=cs)
        chunk_count = 0
        cleaned_row_count = 0
        for chunk in stream_reader:
            chunk_count += 1
            cleaned_row_count += len(clean_chunk(chunk))
        t1_stream = time.perf_counter()
        _, peak_stream_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        stream_kb = round(peak_stream_bytes / 1024.0, 2)
        stream_mb = round(peak_stream_bytes / (1024.0 * 1024.0), 3)
        stream_time = round(t1_stream - t0_stream, 4)
        stream_throughput = round(cleaned_row_count / stream_time, 1) if stream_time > 0 else 0.0
        reduction_pct = round(((peak_base_bytes - peak_stream_bytes) / peak_base_bytes) * 100.0, 2)

        if reduction_pct > max_memory_reduction_pct:
            max_memory_reduction_pct = reduction_pct
            best_streaming_chunk_size = cs

        results.append({
            "mode": f"Stream Cleaning (O(C), C={cs})",
            "chunk_size": cs,
            "chunk_size_label": f"{cs:,} rows",
            "peak_memory_bytes": peak_stream_bytes,
            "peak_memory_kb": stream_kb,
            "peak_memory_mb": stream_mb,
            "elapsed_seconds": stream_time,
            "throughput_rows_sec": stream_throughput,
            "memory_reduction_pct": reduction_pct,
            "num_chunks": chunk_count,
        })

    return {
        "dataset_info": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "raw_size_bytes": len(csv_bytes),
            "raw_size_mb": round(len(csv_bytes) / (1024.0 * 1024.0), 2),
        },
        "baseline_peak_kb": baseline_kb,
        "baseline_peak_mb": baseline_mb,
        "best_streaming_chunk_size": best_streaming_chunk_size,
        "max_memory_reduction_pct": max_memory_reduction_pct,
        "evaluations": results,
    }


class DisjointSetUnion:
    """
    Disjoint Set Union (DSU / Union-Find) data structure with path compression
    and union-by-rank heuristics for O(alpha(V)) disjoint component tracking.
    """

    def __init__(self, elements: List[str]) -> None:
        self.parent: Dict[str, str] = {e: e for e in elements}
        self.rank: Dict[str, int] = {e: 0 for e in elements}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        curr = item
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, item_a: str, item_b: str) -> bool:
        root_a = self.find(item_a)
        root_b = self.find(item_b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


def build_feature_dependency_graph(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> Dict[str, object]:
    """
    Construct an automated feature correlation dependency network graph.
    Computes graph theoretical topological metrics:
    - Degree and weighted strength centrality per feature
    - Connected component clustering via Disjoint Set Union (DSU)
    - Maximum Spanning Tree (MST) dependency backbone via Kruskal's algorithm
    - Deterministic 2D planar coordinates via spring embedding relaxation
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Correlation threshold must be between 0.0 and 1.0, got {threshold}")

    numeric_df = df.select_dtypes(include=[np.number])
    cols: List[str] = [str(c) for c in numeric_df.columns]
    num_nodes = len(cols)

    if num_nodes < 2:
        empty_nodes: List[Dict[str, Union[str, int, float]]] = [
            {
                "id": c,
                "degree": 0,
                "degree_centrality": 0.0,
                "strength": 0.0,
                "community_id": 0,
                "x": 0.0,
                "y": 0.0,
            }
            for c in cols
        ]
        return {
            "nodes": empty_nodes,
            "edges": [],
            "metrics": {
                "total_nodes": num_nodes,
                "total_edges": 0,
                "density": 0.0,
                "num_connected_components": num_nodes,
                "isolated_nodes": cols,
                "hub_nodes": [],
            },
            "threshold": threshold,
        }

    corr_matrix = numeric_df.corr(method="pearson").fillna(0.0)

    # 1. Collect candidate undirected edges meeting threshold
    raw_edges: List[Dict[str, Union[str, float]]] = []
    for i in range(num_nodes):
        u = cols[i]
        for j in range(i + 1, num_nodes):
            v = cols[j]
            r = float(corr_matrix.loc[u, v])
            weight = abs(r)
            if weight >= threshold:
                raw_edges.append({
                    "source": u,
                    "target": v,
                    "weight": round(weight, 4),
                    "correlation": round(r, 4),
                })

    # 2. Kruskal's Algorithm for Maximum Spanning Forest (MST backbone)
    sorted_edges = sorted(raw_edges, key=lambda e: float(e["weight"]), reverse=True)
    dsu_mst = DisjointSetUnion(cols)
    edges: List[Dict[str, Union[str, float, bool]]] = []
    for edge in sorted_edges:
        u = str(edge["source"])
        v = str(edge["target"])
        is_mst = dsu_mst.union(u, v)
        edges.append({
            "source": u,
            "target": v,
            "weight": edge["weight"],
            "correlation": edge["correlation"],
            "is_mst_backbone": is_mst,
        })

    # 3. Connected Components clustering via DSU
    dsu_components = DisjointSetUnion(cols)
    for edge in raw_edges:
        dsu_components.union(str(edge["source"]), str(edge["target"]))

    component_roots = sorted(list({dsu_components.find(c) for c in cols}))
    root_to_comm: Dict[str, int] = {r: idx for idx, r in enumerate(component_roots)}

    # 4. Degree Centrality and Weighted Node Strength
    degrees: Dict[str, int] = {c: 0 for c in cols}
    strengths: Dict[str, float] = {c: 0.0 for c in cols}
    for edge in raw_edges:
        u = str(edge["source"])
        v = str(edge["target"])
        w = float(edge["weight"])
        degrees[u] += 1
        degrees[v] += 1
        strengths[u] += w
        strengths[v] += w

    # 5. Deterministic Circular & Spring Embedding Coordinates
    pos_x: Dict[str, float] = {}
    pos_y: Dict[str, float] = {}
    for idx, c in enumerate(cols):
        theta = (2.0 * math.pi * idx / num_nodes) - (math.pi / 2.0)
        pos_x[c] = round(math.cos(theta), 4)
        pos_y[c] = round(math.sin(theta), 4)

    # 25 iterations of spring force relaxation
    k_repulse = 0.08
    k_attract = 0.15
    dt = 0.1
    for _ in range(25):
        disp_x: Dict[str, float] = {c: 0.0 for c in cols}
        disp_y: Dict[str, float] = {c: 0.0 for c in cols}

        for i in range(num_nodes):
            u = cols[i]
            for j in range(i + 1, num_nodes):
                v = cols[j]
                dx = pos_x[u] - pos_x[v]
                dy = pos_y[u] - pos_y[v]
                dist = math.sqrt(dx * dx + dy * dy) + 1e-4
                force = k_repulse / (dist * dist)
                disp_x[u] += (dx / dist) * force
                disp_y[u] += (dy / dist) * force
                disp_x[v] -= (dx / dist) * force
                disp_y[v] -= (dy / dist) * force

        for edge in raw_edges:
            u = str(edge["source"])
            v = str(edge["target"])
            w = float(edge["weight"])
            dx = pos_x[v] - pos_x[u]
            dy = pos_y[v] - pos_y[u]
            dist = math.sqrt(dx * dx + dy * dy)
            force = k_attract * dist * w
            disp_x[u] += dx * force
            disp_y[u] += dy * force
            disp_x[v] -= dx * force
            disp_y[v] -= dy * force

        for c in cols:
            pos_x[c] += disp_x[c] * dt
            pos_y[c] += disp_y[c] * dt

    max_radius = max(math.sqrt(pos_x[c] ** 2 + pos_y[c] ** 2) for c in cols)
    if max_radius < 1e-5:
        max_radius = 1.0

    nodes: List[Dict[str, Union[str, int, float]]] = []
    for c in cols:
        deg = degrees[c]
        deg_cent = round(deg / (num_nodes - 1), 4) if num_nodes > 1 else 0.0
        comm_id = root_to_comm[dsu_components.find(c)]
        norm_x = round(pos_x[c] / max_radius, 4)
        norm_y = round(pos_y[c] / max_radius, 4)
        nodes.append({
            "id": c,
            "degree": deg,
            "degree_centrality": deg_cent,
            "strength": round(strengths[c], 4),
            "community_id": comm_id,
            "x": norm_x,
            "y": norm_y,
        })

    total_edges = len(edges)
    possible_edges = (num_nodes * (num_nodes - 1)) / 2.0
    density = round(total_edges / possible_edges, 4) if possible_edges > 0 else 0.0
    isolated = [c for c in cols if degrees[c] == 0]
    avg_deg = sum(degrees.values()) / num_nodes if num_nodes > 0 else 0.0
    hubs = sorted(
        [c for c in cols if degrees[c] >= avg_deg and degrees[c] > 0],
        key=lambda c: degrees[c],
        reverse=True,
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "metrics": {
            "total_nodes": num_nodes,
            "total_edges": total_edges,
            "density": density,
            "num_connected_components": len(component_roots),
            "isolated_nodes": isolated,
            "hub_nodes": hubs,
        },
        "threshold": threshold,
    }


class HeuristicRule:
    """
    Declarative representation of an interpretable domain rule or heuristic constraint.
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: str,
        severity: float,
        rule_type: str,
        features: List[str],
        predicate: Callable[[pd.DataFrame], pd.Series],
    ) -> None:
        if not (0.0 <= severity <= 1.0):
            raise ValueError(f"Rule severity must be between 0.0 and 1.0, got {severity}")
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = round(float(severity), 2)
        self.rule_type = rule_type
        self.features = features
        self.predicate = predicate


def synthesize_heuristic_rules(df: pd.DataFrame) -> List[HeuristicRule]:
    """
    Synthesize domain-grounded heuristic rules automatically from dataset properties:
    1. Extreme Z-Score tail anomalies (|z| > 3.5)
    2. Extreme IQR outer fences (Q1 - 3*IQR, Q3 + 3*IQR)
    3. Cross-feature bivariate coupling violations (diverging from expected linear regression)
    4. Non-negative domain invariant violations
    """
    rules: List[HeuristicRule] = []
    numeric_df = df.select_dtypes(include=[np.number])
    cols = [str(c) for c in numeric_df.columns]

    for col in cols:
        series = numeric_df[col].dropna()
        if len(series) < 3:
            continue

        std = float(series.std(ddof=0))
        mean = float(series.mean())

        # 1. Extreme Z-Score Tail Rule
        if std > 1e-6:
            def make_zscore_pred(c: str, m: float, s: float) -> Callable[[pd.DataFrame], pd.Series]:
                return lambda d: (d[c] - m).abs() / s > 3.5

            rules.append(
                HeuristicRule(
                    rule_id=f"R_ZSCORE_{col}",
                    name=f"Extreme Z-Score ({col})",
                    description=f"Value deviates > 3.5 standard deviations from mean ({mean:.2f} ± {std:.2f})",
                    severity=0.80,
                    rule_type="extreme_tail",
                    features=[col],
                    predicate=make_zscore_pred(col, mean, std),
                )
            )

        # 2. Extreme IQR Outer Fence Rule
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr > 1e-6:
            low_fence = q1 - 3.0 * iqr
            high_fence = q3 + 3.0 * iqr

            def make_iqr_pred(c: str, lf: float, hf: float) -> Callable[[pd.DataFrame], pd.Series]:
                return lambda d: (d[c] < lf) | (d[c] > hf)

            rules.append(
                HeuristicRule(
                    rule_id=f"R_IQR_{col}",
                    name=f"Outer IQR Fence ({col})",
                    description=f"Value falls outside extreme 3.0x IQR fences [{low_fence:.2f}, {high_fence:.2f}]",
                    severity=0.85,
                    rule_type="outer_fence",
                    features=[col],
                    predicate=make_iqr_pred(col, low_fence, high_fence),
                )
            )

        # 3. Non-Negative Domain Invariant Rule
        if float((series >= 0).mean()) >= 0.98 and float(series.min()) < 0:
            def make_nonneg_pred(c: str) -> Callable[[pd.DataFrame], pd.Series]:
                return lambda d: d[c] < 0

            rules.append(
                HeuristicRule(
                    rule_id=f"R_NONNEG_{col}",
                    name=f"Non-Negative Invariant ({col})",
                    description=f"Value is negative in predominantly non-negative feature {col}",
                    severity=0.90,
                    rule_type="domain_invariant",
                    features=[col],
                    predicate=make_nonneg_pred(col),
                )
            )

    # 4. Cross-Feature Bivariate Coupling Discrepancy Rules
    if len(cols) >= 2:
        pearson_matrix = numeric_df.corr(method="pearson").fillna(0.0)
        spearman_matrix = numeric_df.corr(method="spearman").fillna(0.0)
        for i in range(len(cols)):
            col_a = cols[i]
            for j in range(i + 1, len(cols)):
                col_b = cols[j]
                r_p = float(pearson_matrix.loc[col_a, col_b])
                r_s = float(spearman_matrix.loc[col_a, col_b])
                r = r_s if abs(r_s) >= abs(r_p) else r_p
                if abs(r) >= 0.70:
                    s_a = numeric_df[col_a].dropna()
                    s_b = numeric_df[col_b].dropna()
                    mean_a = float(s_a.median())
                    mean_b = float(s_b.median())
                    iqr_a = float(s_a.quantile(0.75) - s_a.quantile(0.25))
                    iqr_b = float(s_b.quantile(0.75) - s_b.quantile(0.25))
                    std_a = (iqr_a / 1.349) if iqr_a > 1e-6 else float(s_a.std(ddof=0))
                    std_b = (iqr_b / 1.349) if iqr_b > 1e-6 else float(s_b.std(ddof=0))

                    if std_a > 1e-6 and std_b > 1e-6:
                        residual_std = math.sqrt(max(1.0 - (r * r), 0.05))

                        def make_coupling_pred(
                            ca: str, cb: str, ma: float, sa: float, mb: float, sb: float, corr: float, r_std: float
                        ) -> Callable[[pd.DataFrame], pd.Series]:
                            return lambda d: (
                                ((d[ca] - ma) / sa) - corr * ((d[cb] - mb) / sb)
                            ).abs() / r_std > 3.5

                        rules.append(
                            HeuristicRule(
                                rule_id=f"R_COUPLING_{col_a}_{col_b}",
                                name=f"Coupling Discrepancy ({col_a} vs {col_b})",
                                description=(
                                    f"Observed value breaks expected strong correlation (r={r:.2f}) "
                                    f"between {col_a} and {col_b} by > 3.5 residual standard errors"
                                ),
                                severity=0.95,
                                rule_type="cross_feature_coupling",
                                features=[col_a, col_b],
                                predicate=make_coupling_pred(
                                    col_a, col_b, mean_a, std_a, mean_b, std_b, r, residual_std
                                ),
                            )
                        )

    return rules


def evaluate_heuristic_rules(
    df: pd.DataFrame,
    rules: Optional[List[HeuristicRule]] = None,
) -> Dict[str, object]:
    """
    Evaluate declarative heuristic domain rules across dataset.
    Provides explainable anomaly attribution:
    - Composite anomaly score per record based on weighted rule violations
    - Rule-by-rule violation diagnostics and attribution
    - Replaces blackbox IsolationForest with explainable AI evaluation
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "total_anomalies": 0,
            "anomaly_percentage": 0.0,
            "anomalous_indices": [],
            "total_rules": 0,
            "rules_evaluated": [],
            "record_attributions": {},
            "top_anomalies": [],
        }

    if rules is None:
        rules = synthesize_heuristic_rules(df)

    if not rules:
        return {
            "total_anomalies": 0,
            "anomaly_percentage": 0.0,
            "anomalous_indices": [],
            "total_rules": 0,
            "rules_evaluated": [],
            "record_attributions": {},
            "top_anomalies": [],
        }

    total_weight = sum(r.severity for r in rules)
    row_violations: Dict[int, List[Dict[str, Union[str, float]]]] = {i: [] for i in df.index}
    rules_summary: List[Dict[str, Union[str, int, float]]] = []

    for rule in rules:
        try:
            mask = rule.predicate(df).fillna(False)
            violating_indices = df.index[mask].tolist()
            v_count = len(violating_indices)
            v_pct = round((v_count / total_rows) * 100.0, 2)

            for idx in violating_indices:
                row_violations[idx].append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "description": rule.description,
                    "severity": rule.severity,
                    "rule_type": rule.rule_type,
                })

            rules_summary.append({
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "severity": rule.severity,
                "rule_type": rule.rule_type,
                "features": ", ".join(rule.features),
                "violation_count": v_count,
                "violation_pct": v_pct,
            })
        except Exception:
            continue

    anomalous_indices: List[int] = []
    scores: Dict[int, float] = {}
    top_records: List[Dict[str, object]] = []

    for idx, v_list in row_violations.items():
        if not v_list:
            scores[idx] = 0.0
            continue
        row_weight = sum(float(v["severity"]) for v in v_list)
        score = round(row_weight / total_weight, 4) if total_weight > 0 else 0.0
        scores[idx] = score

        has_severe = any(float(v["severity"]) >= 0.85 for v in v_list)
        if score >= 0.20 or has_severe:
            anomalous_indices.append(idx)
            top_records.append({
                "index": idx,
                "composite_score": score,
                "violated_rules_count": len(v_list),
                "violations": v_list,
            })

    top_records.sort(key=lambda r: float(r["composite_score"]), reverse=True)
    count = len(anomalous_indices)
    pct = round((count / total_rows) * 100.0, 2)

    return {
        "total_anomalies": count,
        "anomaly_percentage": pct,
        "anomalous_indices": anomalous_indices,
        "total_rules": len(rules_summary),
        "rules_evaluated": rules_summary,
        "record_attributions": {r["index"]: r["violations"] for r in top_records[:50]},
        "top_anomalies": top_records[:20],
    }


def compute_wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute 1D first Wasserstein Distance (Earth Mover's Distance) between two empirical distributions.
    W_1(u, v) = integral |F_u(t) - F_v(t)| dt
    Evaluated over uniform quantile discretization grid.
    """
    u_clean = u[~np.isnan(u)]
    v_clean = v[~np.isnan(v)]
    if len(u_clean) == 0 or len(v_clean) == 0:
        raise ValueError("Arrays must contain at least one non-null value for Wasserstein computation.")

    q_grid = np.linspace(0.01, 0.99, 100)
    q_u = np.quantile(u_clean, q_grid)
    q_v = np.quantile(v_clean, q_grid)
    w_dist = float(np.mean(np.abs(q_u - q_v)))
    return round(w_dist, 4)


def compute_ks_statistic_1d(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov statistic D between two empirical distributions.
    D = sup_x |F_u(x) - F_v(x)| in [0.0, 1.0].
    """
    u_clean = u[~np.isnan(u)]
    v_clean = v[~np.isnan(v)]
    if len(u_clean) == 0 or len(v_clean) == 0:
        raise ValueError("Arrays must contain at least one non-null value for KS statistic computation.")

    all_points = np.sort(np.unique(np.concatenate([u_clean, v_clean])))
    u_sorted = np.sort(u_clean)
    v_sorted = np.sort(v_clean)

    cdf_u = np.searchsorted(u_sorted, all_points, side="right") / float(len(u_clean))
    cdf_v = np.searchsorted(v_sorted, all_points, side="right") / float(len(v_clean))
    d_stat = float(np.max(np.abs(cdf_u - cdf_v)))
    return round(d_stat, 4)


def compute_population_stability_index(
    observed: np.ndarray,
    imputed: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between observed baseline and post-imputation distributions.
    PSI = sum((P_b - Q_b) * ln(P_b / Q_b))
    Features epsilon Laplace smoothing to prevent numerical overflow.
    """
    obs_clean = observed[~np.isnan(observed)]
    imp_clean = imputed[~np.isnan(imputed)]
    if len(obs_clean) == 0 or len(imp_clean) == 0:
        raise ValueError("Arrays must contain at least one non-null value for PSI computation.")
    if bins < 2:
        raise ValueError(f"Number of bins must be >= 2, got {bins}")

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(obs_clean, quantiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    obs_counts, _ = np.histogram(obs_clean, bins=bin_edges)
    imp_counts, _ = np.histogram(imp_clean, bins=bin_edges)

    eps = 1e-4
    p = (obs_counts + eps) / (len(obs_clean) + eps * len(obs_counts))
    q = (imp_counts + eps) / (len(imp_clean) + eps * len(imp_counts))

    psi = float(np.sum((p - q) * np.log(p / q)))
    return round(max(psi, 0.0), 4)


def benchmark_imputation_drift(
    df: pd.DataFrame,
    missing_rate: float = 0.15,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Empirical benchmark comparing imputation algorithms against statistical data drift metrics:
    - Wasserstein Distance (Earth Mover's Distance)
    - Kolmogorov-Smirnov Statistic (D_ks)
    - Population Stability Index (PSI)
    - Mean and standard deviation shift
    Evaluated using controlled Missing Completely At Random (MCAR) injection.
    """
    if not (0.01 <= missing_rate <= 0.50):
        raise ValueError(f"Missing rate must be between 0.01 and 0.50, got {missing_rate}")

    numeric_df = df.select_dtypes(include=[np.number])
    valid_cols: List[str] = [
        str(c) for c in numeric_df.columns
        if numeric_df[c].dropna().nunique() > 2 and len(numeric_df[c].dropna()) >= 10
    ]

    if not valid_cols:
        return {
            "evaluated_features": [],
            "missing_rate": missing_rate,
            "optimal_strategy": "mean",
            "leaderboard": [],
            "per_feature_results": {},
        }

    rng = np.random.RandomState(seed)
    strategies = ["mean", "median", "mode", "zero", "forward_fill", "random_draw"]

    strategy_totals: Dict[str, Dict[str, float]] = {
        s: {"wasserstein": 0.0, "ks": 0.0, "psi": 0.0, "mean_shift": 0.0, "std_shift": 0.0, "score": 0.0}
        for s in strategies
    }
    per_feature: Dict[str, Dict[str, Dict[str, float]]] = {}

    for col in valid_cols:
        col_vals = numeric_df[col].dropna().values.astype(float)
        n = len(col_vals)
        mask = rng.rand(n) < missing_rate

        # Ensure at least 1 masked and at least 3 observed
        if mask.sum() == 0:
            mask[rng.choice(n)] = True
        if (~mask).sum() < 3:
            mask = np.zeros(n, dtype=bool)
            mask[: max(1, int(n * missing_rate))] = True

        y_true = col_vals
        y_obs = col_vals[~mask]
        std_true = float(np.std(y_true)) + 1e-6
        mean_obs = float(np.mean(y_obs))
        median_obs = float(np.median(y_obs))

        # Mode calculation
        vals_unique, counts = np.unique(y_obs, return_counts=True)
        mode_obs = float(vals_unique[np.argmax(counts)])

        per_feature[col] = {}

        for strat in strategies:
            y_imp = col_vals.copy()
            if strat == "mean":
                y_imp[mask] = mean_obs
            elif strat == "median":
                y_imp[mask] = median_obs
            elif strat == "mode":
                y_imp[mask] = mode_obs
            elif strat == "zero":
                y_imp[mask] = 0.0
            elif strat == "forward_fill":
                last_val = mean_obs
                for i in range(n):
                    if mask[i]:
                        y_imp[i] = last_val
                    else:
                        last_val = y_imp[i]
            elif strat == "random_draw":
                y_imp[mask] = rng.choice(y_obs, size=mask.sum(), replace=True)

            w_dist = compute_wasserstein_1d(y_imp, y_true)
            ks_stat = compute_ks_statistic_1d(y_imp, y_true)
            psi_val = compute_population_stability_index(y_true, y_imp)
            m_shift = round(abs(float(np.mean(y_imp) - np.mean(y_true))), 4)
            s_shift = round(abs(float(np.std(y_imp) - np.std(y_true))), 4)

            # Normalized Composite Drift Score (0.0 to 1.0, lower is better)
            norm_w = min(w_dist / std_true, 1.0)
            norm_psi = min(psi_val / 0.25, 1.0)
            comp_score = round(0.35 * ks_stat + 0.35 * norm_psi + 0.30 * norm_w, 4)

            per_feature[col][strat] = {
                "wasserstein": w_dist,
                "ks_statistic": ks_stat,
                "psi": psi_val,
                "mean_shift": m_shift,
                "std_shift": s_shift,
                "composite_score": comp_score,
            }

            strategy_totals[strat]["wasserstein"] += w_dist
            strategy_totals[strat]["ks"] += ks_stat
            strategy_totals[strat]["psi"] += psi_val
            strategy_totals[strat]["mean_shift"] += m_shift
            strategy_totals[strat]["std_shift"] += s_shift
            strategy_totals[strat]["score"] += comp_score

    # Compute averages across evaluated features
    num_f = len(valid_cols)
    leaderboard: List[Dict[str, Union[str, int, float]]] = []
    for strat in strategies:
        avg_w = round(strategy_totals[strat]["wasserstein"] / num_f, 4)
        avg_ks = round(strategy_totals[strat]["ks"] / num_f, 4)
        avg_psi = round(strategy_totals[strat]["psi"] / num_f, 4)
        avg_m = round(strategy_totals[strat]["mean_shift"] / num_f, 4)
        avg_s = round(strategy_totals[strat]["std_shift"] / num_f, 4)
        avg_score = round(strategy_totals[strat]["score"] / num_f, 4)

        leaderboard.append({
            "strategy": strat,
            "composite_drift_score": avg_score,
            "avg_ks_statistic": avg_ks,
            "avg_psi": avg_psi,
            "avg_wasserstein": avg_w,
            "avg_mean_shift": avg_m,
            "avg_std_shift": avg_s,
        })

    leaderboard.sort(key=lambda x: float(x["composite_drift_score"]))
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank

    optimal_strategy = str(leaderboard[0]["strategy"]) if leaderboard else "mean"

    return {
        "evaluated_features": valid_cols,
        "missing_rate": missing_rate,
        "optimal_strategy": optimal_strategy,
        "leaderboard": leaderboard,
        "per_feature_results": per_feature,
    }