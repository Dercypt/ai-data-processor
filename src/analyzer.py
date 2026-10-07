import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, List, Optional, Tuple, Union


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


def analyze_dataset(
    file: object,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]], Optional[str]]:
    """
    Ingest, clean, and compute comprehensive statistical data science summaries.
    Adheres strictly to Law 2 (cleaning integrity) and Law 3 (return contract).
    """
    try:
        df: pd.DataFrame = pd.read_csv(file)
        if df.empty:
            return None, None, "The uploaded CSV file is empty."

        # Eliminate missing values without dropping rows or columns (Law 2)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("Unknown")

        corr = compute_correlations(df)
        skew_vals, kurt_vals, skew_desc, kurt_desc = compute_skewness_and_kurtosis(df)
        iqr_outliers = detect_outliers_iqr(df)
        zscore_outliers = detect_outliers_zscore(df)
        iso_outliers = detect_outliers_isolation_forest(df)

        summary: Dict[str, object] = {
            "columns": list(df.columns),
            "rows": int(len(df)),
            "stats": df.describe().to_dict(),
            "correlation": corr,
            "skewness": skew_vals,
            "kurtosis": kurt_vals,
            "skewness_interpretation": skew_desc,
            "kurtosis_interpretation": kurt_desc,
            "outliers": {
                "iqr": iqr_outliers,
                "zscore": zscore_outliers,
                "isolation_forest": iso_outliers,
            },
        }
        return df, summary, None
    except Exception as e:
        return None, None, str(e)