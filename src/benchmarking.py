
"""
benchmarking.py
----------------
Compare our models against NAM and a Naive (daily persistence) baseline
on both CSI and GHI. Produces aligned CSVs, metrics, skill scores, and plots.

Key entry-point:
    run_benchmark_report(...)

This module makes **no** assumptions about your training code. You just
provide the timestamps for target labels, your y_true (CSI) and y_pred (CSI),
plus a reference dataframe containing the needed columns (truth, NAM, clear-sky).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========================
# Metrics & skill helpers
# ========================

def smart_persistence_csi(csi_series: pd.Series,
                          timestamps: pd.DatetimeIndex,
                          days_back: int = 1) -> np.ndarray:
    # persist yesterday’s CSI at the same intraday time
    return daily_persistence(csi_series, timestamps, days_back=days_back)

def smart_persistence_ghi(csi_series: pd.Series,
                          ghi_cs_series: pd.Series,
                          timestamps: pd.DatetimeIndex,
                          days_back: int = 1) -> np.ndarray:
    # Smart: persist CSI, then scale by TODAY’s clear-sky GHI
    sp_csi = daily_persistence(csi_series, timestamps, days_back=days_back)
    ghi_cs_today = align_series_to_timestamps(ghi_cs_series, timestamps)
    return ghi_from_csi(sp_csi, ghi_cs_today)

def _pick_three_day_window(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # choose the first complete day in the series and take 3 days from there
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    days = idx.normalize().unique()
    if len(days) == 0:
        return idx
    start = days[0]
    end = start + pd.Timedelta(days=3)
    return idx[(idx >= start) & (idx < end)]



def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error in %.
    Note: ignores zero/near-zero y_true entries (|y| <= 1e-8).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-8)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric MAPE in %.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (denom > 1e-8)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a standard set of regression metrics.
    """
    return {
        "RMSE": _rmse(y_true, y_pred),
        "MAE": _mae(y_true, y_pred)
       # "MAPE_%": _mape(y_true, y_pred),
        #"sMAPE_%": _smape(y_true, y_pred),
        #"R2": _r2(y_true, y_pred),
    }


def skill_score(model_err: float, ref_err: float) -> float:
    """
    Skill = 1 - (model_err / ref_err).
    Positive => improvement over reference; Negative => worse than reference.
    Uses errors with "lower is better" semantics (e.g., RMSE, MAE).
    """
    if ref_err is None or np.isnan(ref_err) or ref_err == 0:
        return np.nan
    if model_err is None or np.isnan(model_err):
        return np.nan
    return float(1.0 - (model_err / ref_err))


# ===================================
# Series alignment & baseline builders
# ===================================

def _ensure_datetime_index(s: pd.Series) -> pd.Series:
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, utc=True, errors="coerce")
    return s.sort_index()


def align_series_to_timestamps(series: pd.Series, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """
    Align a series to exact timestamps using index-based selection (no fill).
    Missing timestamps produce NaNs.
    """
    series = _ensure_datetime_index(series.copy())
    ts = pd.DatetimeIndex(timestamps)  # ensure DatetimeIndex
    aligned = series.reindex(ts)
    return aligned.values.astype(float)


def daily_persistence(series: pd.Series, timestamps: pd.DatetimeIndex, days_back: int = 1) -> np.ndarray:
    """
    Naive day-ahead baseline:
        value(t) = series(t - days_back * 1 day)
    This works with any regular/irregular sampling since we directly reindex
    at (timestamps - days_back days).
    """
    series = _ensure_datetime_index(series.copy())
    ts = pd.DatetimeIndex(timestamps) - pd.Timedelta(days=days_back)
    vals = series.reindex(ts).values.astype(float)
    return vals


# ==========================
# CSI/GHI conversions
# ==========================

def csi_from_ghi(ghi: np.ndarray, ghi_cs: np.ndarray) -> np.ndarray:
    ghi = np.asarray(ghi, dtype=float)
    ghi_cs = np.asarray(ghi_cs, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = np.where(ghi_cs > 1e-8, ghi / ghi_cs, np.nan)
    return np.clip(csi, 0.0, 2.0)


def ghi_from_csi(csi: np.ndarray, ghi_cs: np.ndarray) -> np.ndarray:
    csi = np.asarray(csi, dtype=float)
    ghi_cs = np.asarray(ghi_cs, dtype=float)
    ghi = csi * ghi_cs
    return np.clip(ghi, 0.0, None)


# ==========================
# Plotting helpers
# ==========================

def _plot_timeseries(df: pd.DataFrame, cols: List[str], title: str, out_path: Path):
    """
    Line plot overlay; one figure per call. No explicit colors (default matplotlib).
    """
    plt.figure(figsize=(14, 4))
    for c in cols:
        plt.plot(df.index, df[c], label=c)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path):
    """
    Scatter of predictions vs truth with y=x line.
    """
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    # Diagonal
    arr = np.array([y_true, y_pred], dtype=float)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if np.isfinite(vmin) and np.isfinite(vmax):
        plt.plot([vmin, vmax], [vmin, vmax])
        plt.xlim(vmin, vmax)
        plt.ylim(vmin, vmax)
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_bar(df: pd.DataFrame, title: str, out_path: Path):
    """
    Bar chart for a small metric table (one figure per call).
    """
    plt.figure(figsize=(8, 4))
    ax = df.plot(kind="bar")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# ========================================
# Main report: CSI & GHI, NAM & Naive
# ========================================

def run_benchmark_report(
    out_dir: str,
    label_times: pd.DatetimeIndex,
    y_true_csi: np.ndarray,
    y_pred_csi: np.ndarray,
    reference_df: pd.DataFrame,
    column_map: Dict[str, str],
    tag: str = "validation",
    make_plots: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Compare your model (CSI predictions) against NAM and a daily-persistence baseline
    on both CSI and GHI.

    Parameters
    ----------
    out_dir : str
        Directory to save CSVs and plots.
    label_times : DatetimeIndex
        Target timestamps (length must match flattened y_true_csi & y_pred_csi).
    y_true_csi : array-like
        Ground-truth CSI at label_times (flattened).
    y_pred_csi : array-like
        Model-predicted CSI at label_times (flattened).
    reference_df : DataFrame
        DataFrame indexed by time with required columns (see column_map).
    column_map : Dict[str, str]
        Keys (case-sensitive):
          - "truth_csi": true CSI column (e.g., "CSI_ghi") [required]
          - "truth_ghi": true GHI column (optional; if missing, we reconstruct as CSI * ghi_cs)
          - "ghi_cs": clear-sky GHI column (required)
          - "nam_ghi": NAM GHI column (required)
          - "nam_csi": NAM CSI column (optional; if missing, computed as nam_ghi/ghi_cs)
    tag : str
        Suffix for files (e.g., "validation" or "test").
    make_plots : bool
        If True, exports PNG plots.

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict with metric tables and skill summaries.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Validate lengths
    label_times = pd.DatetimeIndex(label_times)
    y_true_csi = np.asarray(y_true_csi, dtype=float).reshape(-1)
    y_pred_csi = np.asarray(y_pred_csi, dtype=float).reshape(-1)
    if len(label_times) != len(y_true_csi) or len(label_times) != len(y_pred_csi):
        raise ValueError(
            f"Length mismatch: label_times={len(label_times)}, "
            f"y_true_csi={len(y_true_csi)}, y_pred_csi={len(y_pred_csi)}"
        )

    # Prepare reference frame
    ref = reference_df.copy()
    if not isinstance(ref.index, pd.DatetimeIndex):
        ref.index = pd.to_datetime(ref.index, utc=True, errors="coerce")
    ref = ref.sort_index()

    # Column names
    truth_csi_col = column_map.get("truth_csi", "CSI_ghi")
    ghi_cs_col    = column_map.get("ghi_cs", "ghi_cs")
    nam_ghi_col   = column_map.get("nam_ghi", "nam_ghi")
    truth_ghi_col = column_map.get("truth_ghi", None)
    nam_csi_col   = column_map.get("nam_csi", None)

    # Fetch arrays at label times
    true_csi = align_series_to_timestamps(ref[truth_csi_col], label_times)

    if truth_ghi_col and (truth_ghi_col in ref.columns):
        true_ghi = align_series_to_timestamps(ref[truth_ghi_col], label_times)
    else:
        # reconstruct from CSI and clear-sky
        ghi_cs_vals = align_series_to_timestamps(ref[ghi_cs_col], label_times)
        true_ghi = ghi_from_csi(true_csi, ghi_cs_vals)

    # Our model CSI and corresponding GHI
    model_csi = np.clip(y_pred_csi, 0.0, 2.0)
    ghi_cs_vals = align_series_to_timestamps(ref[ghi_cs_col], label_times)
    model_ghi = ghi_from_csi(model_csi, ghi_cs_vals)

    # NAM
    nam_ghi_vals = align_series_to_timestamps(ref[nam_ghi_col], label_times)
    if nam_csi_col and (nam_csi_col in ref.columns):
        nam_csi_vals = align_series_to_timestamps(ref[nam_csi_col], label_times)
    else:
        nam_csi_vals = csi_from_ghi(nam_ghi_vals, ghi_cs_vals)

    # Naive daily persistence for CSI & GHI
    naive_csi_vals = daily_persistence(ref[truth_csi_col], label_times, days_back=1)
    if truth_ghi_col and truth_ghi_col in ref.columns:
        naive_ghi_vals = daily_persistence(ref[truth_ghi_col], label_times, days_back=1)
    else:
        # build GHI series = CSI * GHI_cs, then persist
        ghi_series = ref[truth_csi_col] * ref[ghi_cs_col]
        ghi_series.name = "true_GHI_series"
        naive_ghi_vals = daily_persistence(ghi_series, label_times, days_back=1)

    
    
    # Build a tidy comparison table
    # --- Smart Persistence baseline ---
    sp_csi_vals = smart_persistence_csi(ref[truth_csi_col], label_times, days_back=1)
    sp_ghi_vals = smart_persistence_ghi(ref[truth_csi_col], ref[ghi_cs_col], label_times, days_back=1)

    # Build tidy comparison table (rename columns accordingly)
    comp_df = pd.DataFrame(
        {
            "timestamp": label_times,
            "true_CSI": true_csi,
            "our_CSI": model_csi,
            "NAM_CSI": nam_csi_vals,
            "SP_CSI": sp_csi_vals,     # Smart Persistence
            "true_GHI": true_ghi,
            "our_GHI": model_ghi,
            "NAM_GHI": nam_ghi_vals,
            "SP_GHI": sp_ghi_vals,     # Smart Persistence
        }
    ).set_index("timestamp")

    # Metrics tables (Ours, NAM, Smart Persistence)
    metrics_csi = pd.DataFrame(
        {
            "Ours": compute_metrics(comp_df["true_CSI"].values, comp_df["our_CSI"].values),
            "NAM":  compute_metrics(comp_df["true_CSI"].values, comp_df["NAM_CSI"].values),
            "SP":   compute_metrics(comp_df["true_CSI"].values, comp_df["SP_CSI"].values),
        }
    )
    metrics_ghi = pd.DataFrame(
        {
            "Ours": compute_metrics(comp_df["true_GHI"].values, comp_df["our_GHI"].values),
            "NAM":  compute_metrics(comp_df["true_GHI"].values, comp_df["NAM_GHI"].values),
            "SP":   compute_metrics(comp_df["true_GHI"].values, comp_df["SP_GHI"].values),
        }
    )

    # --- Forecast skill vs Smart Persistence (positive = better than SP) ---
    skill_vs_sp_ours = {
        "CSI_RMSE_Skill_Ours_vs_SP": skill_score(metrics_csi.loc["RMSE","Ours"], metrics_csi.loc["RMSE","SP"]),
        "CSI_MAE_Skill_Ours_vs_SP":  skill_score(metrics_csi.loc["MAE","Ours"],  metrics_csi.loc["MAE","SP"]),
        "GHI_RMSE_Skill_Ours_vs_SP": skill_score(metrics_ghi.loc["RMSE","Ours"], metrics_ghi.loc["RMSE","SP"]),
        "GHI_MAE_Skill_Ours_vs_SP":  skill_score(metrics_ghi.loc["MAE","Ours"],  metrics_ghi.loc["MAE","SP"]),
    }
    skill_vs_sp_nam = {
        "CSI_RMSE_Skill_NAM_vs_SP":  skill_score(metrics_csi.loc["RMSE","NAM"],  metrics_csi.loc["RMSE","SP"]),
        "CSI_MAE_Skill_NAM_vs_SP":   skill_score(metrics_csi.loc["MAE","NAM"],   metrics_csi.loc["MAE","SP"]),
        "GHI_RMSE_Skill_NAM_vs_SP":  skill_score(metrics_ghi.loc["RMSE","NAM"],  metrics_ghi.loc["RMSE","SP"]),
        "GHI_MAE_Skill_NAM_vs_SP":   skill_score(metrics_ghi.loc["MAE","NAM"],   metrics_ghi.loc["MAE","SP"]),
    }

    if make_plots:
        three_idx = _pick_three_day_window(comp_df.index)
        comp3 = comp_df.loc[three_idx]

        # GHI: Ours vs Smart Persistence
        _plot_timeseries(
            comp3[["true_GHI", "our_GHI", "SP_GHI"]],
            ["true_GHI", "our_GHI", "SP_GHI"],
            f"GHI: Ours vs Smart Persistence (3 days, {tag})",
            out_path / f"ghi_vs_sp_3days_{tag}.png",
        )
        # GHI: Ours vs NAM
        _plot_timeseries(
            comp3[["true_GHI", "our_GHI", "NAM_GHI"]],
            ["true_GHI", "our_GHI", "NAM_GHI"],
            f"GHI: Ours vs NAM (3 days, {tag})",
            out_path / f"ghi_vs_nam_3days_{tag}.png",
        )
        # CSI: Ours vs Smart Persistence
        _plot_timeseries(
            comp3[["true_CSI", "our_CSI", "SP_CSI"]],
            ["true_CSI", "our_CSI", "SP_CSI"],
            f"CSI: Ours vs Smart Persistence (3 days, {tag})",
            out_path / f"csi_vs_sp_3days_{tag}.png",
        )
        # CSI: Ours vs NAM
        _plot_timeseries(
            comp3[["true_CSI", "our_CSI", "NAM_CSI"]],
            ["true_CSI", "our_CSI", "NAM_CSI"],
            f"CSI: Ours vs NAM (3 days, {tag})",
            out_path / f"csi_vs_nam_3days_{tag}.png",
        )

        # Scatters
        _plot_scatter(comp_df["true_GHI"].values, comp_df["our_GHI"].values, f"Our Model: GHI ({tag})", out_path / f"scatter_ghi_{tag}.png")
        _plot_scatter(comp_df["true_CSI"].values, comp_df["our_CSI"].values, f"Our Model: CSI ({tag})", out_path / f"scatter_csi_{tag}.png")

        # Bars (errors only: RMSE & MAE)
        _plot_bar(metrics_csi.loc[["RMSE","MAE"]], f"CSI Errors ({tag})", out_path / f"bar_csi_errors_{tag}.png")
        _plot_bar(metrics_ghi.loc[["RMSE","MAE"]], f"GHI Errors ({tag})", out_path / f"bar_ghi_errors_{tag}.png")

    # Build return structure
    # Persist outputs
    comp_df.to_csv(out_path / f"comparison_{tag}.csv", index=True)
    metrics_csi.to_csv(out_path / f"metrics_CSI_{tag}.csv")
    metrics_ghi.to_csv(out_path / f"metrics_GHI_{tag}.csv")

    # Write forecast skill vs Smart Persistence (Ours and NAM)
    with open(out_path / f"skill_vs_sp_summary_{tag}.txt", "w") as f:
        f.write("Forecast Skill vs Smart Persistence (positive => better than SP)\n")
        for k, v in {**skill_vs_sp_ours, **skill_vs_sp_nam}.items():
            f.write(f"{k}: {np.nan if v is None or np.isnan(v) else float(v):.4f}\n")

    # Return structure used by the pipeline to log skills in the summary
    return {
        "CSI": metrics_csi.to_dict(),
        "GHI": metrics_ghi.to_dict(),
        "Skill_vs_SP_Ours": skill_vs_sp_ours,
        "Skill_vs_SP_NAM":  skill_vs_sp_nam,
    }


__all__ = [
    "run_benchmark_report",
    "compute_metrics",
    "skill_score",
    "align_series_to_timestamps",
    "daily_persistence",
    "csi_from_ghi",
    "ghi_from_csi",
]
