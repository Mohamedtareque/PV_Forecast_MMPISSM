"""
Utility helpers to assess model predictions against NAM baselines.

Primary responsibilities:
    * Build a reference frame that aligns per-bin forecast targets with their
      corresponding timestamps and NAM day-ahead forecasts.
    * Compute clear-sky irradiance using pvlib so that all comparisons are made
      in both GHI and clear-sky index space.
    * Aggregate comparison tables and metrics that quantify how much the
      learned model improves upon the NAM guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional
from .preprocessing import to_fixedgrid_multiindex


import numpy as np
import pandas as pd

import pvlib

logger = logging.getLogger(__name__)


@dataclass
class SiteLocation:
    """Container for site metadata needed by pvlib."""

    latitude: float
    longitude: float
    altitude: float = 0.0
    timezone: str = "UTC"


def _load_processed_dataframe(
    csv_path: str,
    tz: str = "UTC",
    drop_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load the processed CSV and return a datetime-indexed frame."""
    # --- FIX 1: Update parse_dates ---
    # Only parse 'measurement_time' and let the drop_columns handle the rest
    df = pd.read_csv(
        csv_path,
        parse_dates=["measurement_time"], 
    )

    if drop_columns:
        df = df.drop(columns=list(drop_columns), errors="ignore")

    # # --- FIX 2: Force all known numeric columns to be numbers ---
    # # This list should include all features + targets
    # numeric_cols = ['ghi', 'dni', 'solar_zenith', 'time_gap_hours', 'time_gap_norm',
    #    'day_boundary_flag', 'hour_progression', 'absolute_hour', 'GHI_cs',
    #    'DNI_cs', 'CSI_ghi', 'CSI_dni', 'season_flag', 'hour_sin', 'hour_cos',
    #    'month_sin', 'month_cos', 'nam_ghi', 'nam_dni', 'nam_cc',
    # ,'B_CSI_ghi_8h', 'V_CSI_ghi_8h', 'L_CSI_ghi_8h',
    #    'B_CSI_ghi_9h', 'V_CSI_ghi_9h', 'L_CSI_ghi_9h', 'B_CSI_ghi_10h',
    #    'V_CSI_ghi_10h', 'L_CSI_ghi_10h', 'B_CSI_ghi_11h', 'V_CSI_ghi_11h',
    #    'L_CSI_ghi_11h', 'B_CSI_ghi_12h', 'V_CSI_ghi_12h', 'L_CSI_ghi_12h',
    #    'B_CSI_ghi_13h', 'V_CSI_ghi_13h', 'L_CSI_ghi_13h', 'B_CSI_ghi_14h',
    #    'V_CSI_ghi_14h', 'L_CSI_ghi_14h', 'B_CSI_ghi_15h', 'V_CSI_ghi_15h',
    #    'L_CSI_ghi_15h', 'B_CSI_ghi_16h', 'V_CSI_ghi_16h', 'L_CSI_ghi_16h',
    #    'B_CSI_ghi_17h', 'V_CSI_ghi_17h', 'L_CSI_ghi_17h', 'B_CSI_ghi_18h',
    #    'V_CSI_ghi_18h', 'L_CSI_ghi_18h', 'B_CSI_ghi_19h', 'V_CSI_ghi_19h',
    #    'L_CSI_ghi_19h', '80_dwsw', '80_cloud_cover', '56_dwsw',
    #    '56_cloud_cover', '20_dwsw', '20_cloud_cover', '88_dwsw',
    #    '88_cloud_cover', 'AVG(R)', 'STD(R)', 'ENT(R)', 'AVG(G)', 'STD(G)',
    #    'ENT(G)', 'AVG(B)', 'STD(B)', 'ENT(B)', 'AVG(RB)', 'STD(RB)', 'ENT(RB)',
    #    'AVG(NRB)', 'STD(NRB)', 'ENT(NRB)']
    
    # # This loop converts all string numbers to floats
    # # and all non-numeric strings (like "N/A") to NaN
    # for col in numeric_cols:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values("measurement_time")
    df = df.set_index("measurement_time")
    if tz:
        df.index = df.index.tz_convert(tz) if df.index.tz else df.index.tz_localize(tz)
    return df




def build_reference_from_existing(
    df: pd.DataFrame,
    time_col: str = "measurement_time",         # your main UTC timestamp
    nam_time_col: str = "nam_target_time",      # NAM valid time (if present)
    meas_ghi_col: str = "ghi",
    nam_ghi_col: str = "nam_ghi",
    cs_ghi_col: str = "GHI_cs",                 # clear-sky already in your data
    actual_csi_col: str = "CSI_ghi",            # if present
):
    """
    Construct a reference frame from already prepared columns in df.
    No re-gridding, no pvlib; uses existing clear-sky & NAM columns.
    Index = UTC tz-aware 'target_time' (NAM valid time if available, else measurement_time).
    """
    # 1) Normalize timestamps to tz-aware UTC
    def _to_utc_series(s):
        s = pd.to_datetime(s, utc=True, errors="coerce")
        return s

    has_nam_time = nam_time_col in df.columns
    target_time = _to_utc_series(df[nam_time_col] if has_nam_time else df[time_col])

    # 2) Build reference on this single time key
    ref = pd.DataFrame(index=pd.Index(target_time, name="target_time"))
    ref = ref[~ref.index.duplicated(keep="first")].sort_index()

    # 3) Attach columns that already exist
    def _attach(colname, outname=None):
        if colname in df.columns:
            s = pd.Series(df[colname].values, index=ref.index)
            ref[outname or colname] = s

    _attach(meas_ghi_col,  "actual_ghi")
    _attach(nam_ghi_col,   "nam_ghi")
    _attach(cs_ghi_col,    "clear_sky_ghi")
    _attach(actual_csi_col, "actual_csi")  # optional if you already computed CSI

    # 4) Compute nam_csi if possible
    if "nam_ghi" in ref.columns and "clear_sky_ghi" in ref.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            ref["nam_csi"] = ref["nam_ghi"] / ref["clear_sky_ghi"]
            ref["nam_csi"].replace([np.inf, -np.inf], np.nan, inplace=True)
    ref.fillna(0, inplace=True)

    return ref

def _metric_rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


def _metric_mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - truth)))


def _metric_mape(pred: np.ndarray, truth: np.ndarray) -> Optional[float]:
    mask = (np.abs(truth) > 1e-3) & np.isfinite(truth) & np.isfinite(pred)
    if not mask.any():
        return None
    return float(np.mean(np.abs((pred[mask] - truth[mask]) / truth[mask])) * 100.0)


def build_comparison_table(
    predictions: np.ndarray,
    targets: np.ndarray,
    sample_indices: np.ndarray,
    label_index: pd.Index,
    reference_df: pd.DataFrame,
    fold_id: int,
) -> pd.DataFrame:
    """
    Align model outputs with actual measurements and NAM forecasts.

    Parameters
    ----------
    predictions : np.ndarray
        Array shaped (N, horizon_days, K) from evaluate_model.
    targets : np.ndarray
        Same shape array of actual values.
    sample_indices : np.ndarray
        Indices of the samples relative to the full dataset (val_idx).
    label_index : pd.Index
        Datetime index mapping each sample to the target date.
    reference_df : pd.DataFrame
        MultiIndex frame returned by build_reference_kbin_frame.
    fold_id : int
        Current cross-validation fold identifier.
    """
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must be shaped (N, horizon, K).")

    if predictions.shape[1] != 1:
        raise NotImplementedError(
            "Comparison metrics currently support horizon_days == 1."
        )

    comparison_rows = []
    unique_dates = reference_df.index.get_level_values("date").unique()

    for cursor, sample_idx in enumerate(sample_indices):
        if sample_idx >= len(label_index):
            logger.warning(
                "Sample index %s exceeds label index bounds (%s). Skipping.",
                sample_idx,
                len(label_index),
            )
            continue

        label_ts = label_index[sample_idx]
        date_key = pd.Timestamp(label_ts.date())
        if date_key not in unique_dates:
            logger.debug(
                "Date %s not found in reference dataframe; skipping sample %s.",
                date_key,
                sample_idx,
            )
            continue

        day_slice = reference_df.loc[date_key]
        day_df = (
            day_slice.reset_index()
            .sort_values("bin_id")
            .reset_index(drop=True)
            .copy()
        )
        day_df.insert(0, "target_date", date_key)

        if len(day_df) != predictions.shape[2]:
            logger.warning(
                "Mismatch between bins (%s) and prediction shape (%s) for date %s.",
                len(day_df),
                predictions.shape[2],
                date_key,
            )
            continue

        day_df["fold_id"] = fold_id
        day_df["sample_index"] = sample_idx
        day_df["sample_order"] = cursor

        # Populate model outputs
        day_df["model_csi_pred"] = predictions[cursor, 0, :]
        day_df["model_ghi_pred"] = (
            day_df["model_csi_pred"] * day_df["clear_sky_ghi"]
        )

        # Actuals obtained from evaluate_model outputs (safer than relying on reference)
        day_df["actual_csi"] = targets[cursor, 0, :]
        day_df["actual_ghi"] = day_df["actual_csi"] * day_df["clear_sky_ghi"]

        comparison_rows.append(day_df[
            [
                "fold_id",
                "sample_index",
                "sample_order",
                "target_date",
                "bin_id",
                "target_time",
                "clear_sky_ghi",
                "actual_csi",
                "actual_ghi",
                "nam_ghi",
                "nam_csi",
                "model_csi_pred",
                "model_ghi_pred",
            ]
        ])

    if not comparison_rows:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "sample_index",
                "sample_order",
                "target_date",
                "bin_id",
                "target_time",
                "clear_sky_ghi",
                "actual_csi",
                "actual_ghi",
                "nam_ghi",
                "nam_csi",
                "model_csi_pred",
                "model_ghi_pred",
            ]
        )

    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    return comparison_df


def compute_nam_comparison_metrics(comparison_df: pd.DataFrame) -> Dict[str, float]:
    """Aggregate scalar metrics comparing the model to the NAM baseline."""
    if comparison_df.empty:
        logger.warning("Comparison dataframe is empty; returning NaN metrics.")
        return {}

    metrics: Dict[str, float] = {}

    daylight_mask = comparison_df["clear_sky_ghi"] > 5.0
    if not daylight_mask.any():
        logger.warning("No daylight points detected; skipping GHI metrics.")
        daylight_mask = comparison_df.index == comparison_df.index  # all True

    actual_ghi = comparison_df.loc[daylight_mask, "actual_ghi"].to_numpy()
    model_ghi = comparison_df.loc[daylight_mask, "model_ghi_pred"].to_numpy()
    nam_ghi = comparison_df.loc[daylight_mask, "nam_ghi"].to_numpy()

    metrics["model_rmse_ghi"] = _metric_rmse(model_ghi, actual_ghi)
    metrics["nam_rmse_ghi"] = _metric_rmse(nam_ghi, actual_ghi)
    metrics["model_mae_ghi"] = _metric_mae(model_ghi, actual_ghi)
    metrics["nam_mae_ghi"] = _metric_mae(nam_ghi, actual_ghi)

    mape_model = _metric_mape(model_ghi, actual_ghi)
    mape_nam = _metric_mape(nam_ghi, actual_ghi)
    if mape_model is not None:
        metrics["model_mape_ghi"] = mape_model
    if mape_nam is not None:
        metrics["nam_mape_ghi"] = mape_nam

    actual_csi = comparison_df["actual_csi"].to_numpy()
    model_csi = comparison_df["model_csi_pred"].to_numpy()
    nam_csi = comparison_df["nam_csi"].to_numpy()
    mask_csi = ~np.isnan(actual_csi) & ~np.isnan(model_csi)
    mask_csi_nam = mask_csi & ~np.isnan(nam_csi)

    if mask_csi.any():
        metrics["model_rmse_csi"] = _metric_rmse(model_csi[mask_csi], actual_csi[mask_csi])
        metrics["model_mae_csi"] = _metric_mae(model_csi[mask_csi], actual_csi[mask_csi])
        mape_csi_model = _metric_mape(model_csi[mask_csi], actual_csi[mask_csi])
        if mape_csi_model is not None:
            metrics["model_mape_csi"] = mape_csi_model

    if mask_csi_nam.any():
        metrics["nam_rmse_csi"] = _metric_rmse(nam_csi[mask_csi_nam], actual_csi[mask_csi_nam])
        metrics["nam_mae_csi"] = _metric_mae(nam_csi[mask_csi_nam], actual_csi[mask_csi_nam])
        mape_csi_nam = _metric_mape(nam_csi[mask_csi_nam], actual_csi[mask_csi_nam])
        if mape_csi_nam is not None:
            metrics["nam_mape_csi"] = mape_csi_nam

    def _skill(model_value: float, baseline_value: Optional[float]) -> Optional[float]:
        if baseline_value in (None, 0.0):
            return None
        return float((1.0 - model_value / baseline_value) * 100.0)

    for metric_name in ("rmse", "mae", "mape"):
        ghi_skill = _skill(
            metrics.get(f"model_{metric_name}_ghi"),
            metrics.get(f"nam_{metric_name}_ghi"),
        )
        if ghi_skill is not None:
            metrics[f"{metric_name}_skill_ghi_pct"] = ghi_skill

        csi_skill = _skill(
            metrics.get(f"model_{metric_name}_csi"),
            metrics.get(f"nam_{metric_name}_csi"),
        )
        if csi_skill is not None:
            metrics[f"{metric_name}_skill_csi_pct"] = csi_skill

    metrics["n_rows"] = int(len(comparison_df))
    metrics["n_daylight_rows"] = int(daylight_mask.sum())

    return metrics
