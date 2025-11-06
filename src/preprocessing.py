"""
Purpose: This module contains all the core, heavy-lifting functions that transform the raw time-series data into the tensor format 
required by the model.


Content:

- process_splits_to_kbins()
- build_model_arrays()
- _tensor_from_norm()
- validate_roundtrip_split() and its helpers (_reconstruct_hourly_from_kbins, validate_roundtrip_day)
- Other low-level helper functions like normalize_day_to_kbins.

These functions represent a distinct, logical step in the pipeline: feature engineering and data shaping. 
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass

# -------- Interpolation primitives --------

try:
    from scipy.interpolate import PchipInterpolator as _PCHIP
    _HAVE_PCHIP = True
except Exception:
    _HAVE_PCHIP = False
    _PCHIP = None


#! Must be moved out into another module responsilbe for dataSpliting.
@dataclass
class SplitData:
    train: Optional[pd.DataFrame] = None
    val: Optional[pd.DataFrame] = None
    test: Optional[pd.DataFrame] = None

@dataclass
class KBinConfig:
    K: int = 60
    tz: Optional[str] = None 
    use_pchip: bool = True
    irradiance_mode: str = "pchip_renorm"  # "pchip_renorm" | "conservative" | "hold"
    clamp_irradiance_nonneg: bool = True
    post_smooth_window: int = 0  # >0 only when irradiance_mode="conservative"


DEFAULT_STRATEGY = {
    "ghi": "irradiance",
    "dni": "irradiance",
    "CSI_ghi": "irradiance",
    "CSI_dni": "irradiance",

    "air_temp": "continuous",
    "relhum": "continuous",
    "windsp": "continuous",

    "solar_zenith": "continuous",
    "solar_elevation": "continuous",
    "GHI_cs": "continuous",
    "DNI_cs": "continuous",

    "winddirection_sin": "windvec",
    "winddirection_cos": "windvec",

    "season_flag": "categorical",
    "is_daylight": "categorical",
    "day_boundary_flag": "categorical",

    "hour_sin": "temporal_recompute",
    "hour_cos": "temporal_recompute",
    "month_sin": "temporal_recompute",
    "month_cos": "temporal_recompute",
    "absolute_hour": "temporal_recompute",
    "hour_progression": "temporal_recompute",
    "time_gap_hours": "temporal_recompute",
    "time_gap_norm": "temporal_recompute",
}


def ensure_datetime_index(df: pd.DataFrame, timestamp_col: Optional[str]=None, tz: Optional[str]=None) -> pd.DataFrame:
    out = df.copy()
    if timestamp_col is not None and timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], "coerce")
        out = out.set_index(timestamp_col)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)

    #! UTC is the standard.
    if tz is not None:
        if out.index.tz is None:
            out.index = out.index.tz_localize(tz)
        else:
            out.index = out.index.tz_convert(tz)
    return out.sort_index()



def _linear_interp_at(target_index: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """ 
        This function takes an existing time series with irregular timestamps and 
        calculates what the values would have been at a new, specified set of timestamps 
        using time-weighted linear interpolation. 
        
    """
    # creates a complete, ordered timeline that includes both the original data points and the new points where we need to estimate values.
    union = series.index.union(target_index).unique().sort_values()
    tmp = series.reindex(union)
    tmp = tmp.interpolate(method="time", limit_area="inside")
    return tmp.reindex(target_index)


def _nearest_at(target_index: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """
    Finds the value from the nearest timestamp by first filling NaN gaps
    and then selecting the target times.
    """
    return series.reindex(target_index, method='nearest')

def _pchip_interp_at(target_index: pd.DatetimeIndex, series: pd.Series) -> pd.Series:
    """ interpolate continuous physical data like temperature or solar irradiance compared to simple linear interpolation.
    Uisng PCHIP (Piecewise Cubic Hermite Interpolating Polynomial), which creates a smooth curve through the data points while being shape-preserving. 
    Also it avoids the "overshoot" or "wiggles" that a standard spline interpolation might create, resulting in a more physically realistic estimation.
    --> It is like drawing a smooth, natural-looking curve through the data points without ever creating artificial peaks or valleys.
    """
    if not _HAVE_PCHIP:
        return _linear_interp_at(target_index, series)
    x = (series.index.view("int64") / 1e9) / 3600.0  # hours since epoch
    y = series.values.astype(float)
    xi = (target_index.view("int64") / 1e9) / 3600.0
    finite_mask = np.isfinite(y)
    if finite_mask.sum() < 2:
        return _linear_interp_at(target_index, series)
    pchip = _PCHIP(x[finite_mask], y[finite_mask])
    yi = pchip(xi)
    return pd.Series(yi, index=target_index)


def _conservative_remap_to_bins(hourly_series: pd.Series, K: int) -> pd.Series:
    """
    handling physical quantities like solar irradiance, resamples an hourly series into K bins, not by just picking points, 
    but by ensuring the total "energy" or quantity of the original signal is conserved.
    """
    n_hours = len(hourly_series) # the total number of hours for the day, it is variable from one season to another.
    if n_hours == 0: # it will not happen, but it is safe guard.
        return pd.Series([], dtype=float)
    
    edges = np.linspace(0.0, float(n_hours), K+1) # divides the total duration of the day into K equal intervals.
    vals = hourly_series.values.astype(float)
    bin_vals = np.zeros(K, dtype=float)

    for b in range(K):
        b0, b1 = edges[b], edges[b+1]
        dur = b1 - b0
        i_start = int(math.floor(b0))
        i_end   = int(math.ceil(b1) - 1)
        acc = 0.0
        for i in range(i_start, i_end+1):
            if i < 0 or i >= n_hours:
                continue
            h0, h1 = float(i), float(i+1)
            # finds the exact start and end of the overlapping region between the new bin and the current hourly block.
            o0, o1 = max(b0, h0), min(b1, h1)
            # calculates the length of this overlap. If there's no overlap, this will be zero.
            overlap = max(0.0, o1 - o0)
            if overlap > 0.0:
                acc += vals[i] * (overlap / dur)
        bin_vals[b] = acc
    start = hourly_series.index[0]
    centers = start + pd.to_timedelta((np.arange(K) + 0.5) * (n_hours / K), unit="h")
    return pd.Series(bin_vals, index=pd.DatetimeIndex(centers))


def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """
    smooths out fluctuations in a numerical array by applying a simple moving average. 
    It replaces each value in the array with the average of itself and its neighbors within a specified window. 
    Main Goal is to reduce noise in time series data.
    """
    if window <= 1: # Base Case
        return arr
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window) / window # convolution kernel, all points in the window are weighted equally.
    return np.convolve(padded, kernel, mode="valid") # only compute the output for positions where the kernel and the array fully overlap.

# -------- Solar-phase utilities --------

def _kbin_times_from_day(df_day: pd.DataFrame, K: int):
    """
    map the time between sunrise and sunset to a normalized interval from 0 to 1. 
    In this system, sunrise is always at phase 0 and sunset is always at phase 1, regardless of the season.
    calculates the exact timestamps that correspond to the center of K equal-sized bins within this normalized solar phase. The normalized position is called tau (τ).

    Returns: 
    target_time: The new DatetimeIndex with K evenly spaced timestamps.
    tau: The corresponding solar phase for each timestamp.
    day_len_hours: The original length of the day.
    """
    df_day = df_day.sort_index()
    sunrise = df_day.index[0]
    sunset  = df_day.index[-1]
    day_len_hours = float(len(df_day))
    tau = (np.arange(K) + 0.5) / K
    target_time = sunrise + pd.to_timedelta(tau * day_len_hours, unit="h")
    return target_time, tau, day_len_hours

def _recompute_temporals(target_index: pd.DatetimeIndex, sunrise: pd.Timestamp, K: int, day_len_hours: float):

    """
    This function creates new, accurate time-based features for the K evenly spaced timestamps generated by _kbin_times_from_day.
    """
    out = {}
    hours_float = target_index.hour + target_index.minute/60.0
    out["absolute_hour"]   = pd.Series(hours_float, index=target_index)
    # The number of hours that have passed since sunrise for each timestamp.
    out["hour_progression"] = ((target_index - sunrise) / pd.Timedelta(hours=1)).astype(float)
    out["hour_sin"] = pd.Series(np.sin(2*np.pi*hours_float/24.0), index=target_index)
    out["hour_cos"] = pd.Series(np.cos(2*np.pi*hours_float/24.0), index=target_index)
    months = target_index.month
    out["month_sin"] = pd.Series(np.sin(2*np.pi*(months-1)/12.0), index=target_index)
    out["month_cos"] = pd.Series(np.cos(2*np.pi*(months-1)/12.0), index=target_index)
    out["time_gap_hours"] = pd.Series(day_len_hours / K, index=target_index)
    out["time_gap_norm"]  = pd.Series(1.0 / K, index=target_index)
    return out


# -------- Per-day normalization --------

def normalize_day_to_kbins(df_day: pd.DataFrame, K: int, cfg, strategy_map=None) -> pd.DataFrame:
    """
    takes a DataFrame containing a single day of hourly data and transforms it into a new DataFrame with a fixed number of K rows, 
    where each row represents an even slice of "solar time."
    """
    if strategy_map is None:
        strategy_map = DEFAULT_STRATEGY
    if df_day.empty: 
        return pd.DataFrame()

    df_day = df_day.sort_index()
    sunrise, sunset = df_day.index[0], df_day.index[-1]
    # calculate the K new timestamps (target_time) that are evenly spaced between the day's sunrise and sunset.
    target_time, tau, day_len_hours = _kbin_times_from_day(df_day, K)

    out = pd.DataFrame(index=target_time) # empty dataframe metadata
    # Each of the K rows now knows its bin_id (0 to K-1), its solar_phase (a value from 0 to 1), and the day's sunrise/sunset times.
    out["bin_id"] = np.arange(K, dtype=int)
    out["solar_phase"] = tau
    out["sunrise"] = sunrise
    out["sunset"]  = sunset
    out["day_len_hours"] = day_len_hours

    # Handling wind direction.
    have_wvec = ("winddirection_sin" in df_day.columns) and ("winddirection_cos" in df_day.columns)
    if have_wvec and strategy_map.get("winddirection_sin") == "windvec":
        sin_i = _pchip_interp_at(target_time, df_day["winddirection_sin"]) if (cfg.use_pchip and _HAVE_PCHIP) else _linear_interp_at(target_time, df_day["winddirection_sin"])
        cos_i = _pchip_interp_at(target_time, df_day["winddirection_cos"]) if (cfg.use_pchip and _HAVE_PCHIP) else _linear_interp_at(target_time, df_day["winddirection_cos"])
        # normalization, to make sure that information is only about the direction.
        mag = np.sqrt(sin_i.values**2 + cos_i.values**2)
        mag[mag == 0] = 1.0
        out["winddirection_sin"] = sin_i.values / mag
        out["winddirection_cos"] = cos_i.values / mag

    for col in df_day.columns:
        if have_wvec and col in ("winddirection_sin","winddirection_cos"):
            continue
        strat = strategy_map.get(col, "continuous") # default is continuous

        # When we change the time resolution from hourly to K-bins, we don't want to accidentally create or destroy the total solar energy received that day in our dataset.
        if strat == "irradiance":
            if cfg.irradiance_mode == "conservative":
                series = _conservative_remap_to_bins(df_day[col], K)
            elif cfg.irradiance_mode == "hold":
                series = _nearest_at(target_time, df_day[col])
            # PCHIP: create a smooth, physically realistic curve. But PCHIP is designed to preserve shape, not necessarily the total area under the curve.
            else:
                # Default is PCHIP which creates a smooth curve that is designed to look good and be physically plausible, but it doesn't have any built-in rule to make the total area underneath it match the original.
                series = _pchip_interp_at(target_time, df_day[col]) if (cfg.use_pchip and _HAVE_PCHIP) else _linear_interp_at(target_index=target_time, series=df_day[col])
                # Energy renormalization
                x_orig = np.arange(len(df_day), dtype=float)
                # Calculate the total "energy" of the original hourly data
                orig_total = np.trapezoid(df_day[col].values.astype(float), x_orig)
                x_k = np.linspace(0.0, day_len_hours, K)
                # Calculate the total "energy" under the new smooth curve
                up_total = np.trapezoid(series.values.astype(float), x_k)
                # Apply the correction
                if np.isfinite(orig_total) and np.isfinite(up_total) and up_total != 0:
                    # calculates a correction factor
                    series = pd.Series(series.values * (orig_total / up_total), index=series.index)
            if cfg.clamp_irradiance_nonneg:
                series = series.clip(lower=0.0)
            out[col] = series.values

        elif strat == "continuous":
            series = _pchip_interp_at(target_time, df_day[col]) if (cfg.use_pchip and _HAVE_PCHIP) else _linear_interp_at(target_time, df_day[col])
            out[col] = series.values

        elif strat == "categorical":
            out[col] = _nearest_at(target_time, df_day[col]).values

        elif strat == "temporal_recompute":
            # defer
            pass

        else:
            out[col] = _linear_interp_at(target_time, df_day[col]).values

    recompute_needed = [c for c, s in strategy_map.items() if s == "temporal_recompute" and (c in df_day.columns)]
    if recompute_needed:
        recomputed = _recompute_temporals(target_index=target_time, sunrise=sunrise, K=K, day_len_hours=day_len_hours)
        for c in recompute_needed:
            if c in recomputed:
                out[c] = recomputed[c].values
            else:
                out[c] = _nearest_at(target_time, df_day[c]).values

    if "is_daylight" in out.columns:
        out["is_daylight"] = 1
    if "day_boundary_flag" in out.columns:
        out["day_boundary_flag"] = 0

    if cfg.post_smooth_window and cfg.irradiance_mode == "conservative":
        for c in ("ghi","dni","CSI_ghi","CSI_dni"):
            if c in out.columns:
                base = out[c].values.astype(float)
                sm = _moving_average(base, window=int(cfg.post_smooth_window))
                if sm.sum() != 0:
                    sm *= base.sum() / sm.sum()
                out[c] = sm

    out.index.name = "target_time" 
    return out



def process_splits_to_kbins(
    splits: Union[Dict[str, pd.DataFrame], SplitData],
    cfg: KBinConfig,
    feature_cols: Optional[Iterable[str]] = None,
    strategy_map: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """models like Recurrent Neural Networks (RNNs) or Transformers that often expect fixed-size input sequences. 
    The function takes raw, continuous time-series data and resamples it into a fixed number of K bins for each day.

    •	Groups the time series by day.
	•	Maps each day's daylight to K equal bins (0…K-1).
	•	Rebuilds the DataFrame with MultiIndex (date, bin_id), sorted.
	•	Ensures the same per-day length (K) for all days.
	•	Returns a per-split normalized DataFrame that build_model_arrays can consume directly.

    Args:
        splits (Union[Dict[str, pd.DataFrame], SplitData]): This argument accepts time-series data, typically divided into sets like 'train', 'validation', and 'test'. 
        it can be a dictionary mapping split names to pandas DataFrames or a custom SplitData class.
        cfg (KBinConfig): A configuration object that holds key parameters for the binning process.
        feature_cols (Optional[Iterable[str]], optional): optional arguments for more granular control, likely used by the downstream function normalize_day_to_kbins. 
        it is the list of columns that will be used as model inputs (the predictors).Defaults to None.
        strategy_map (Optional[Dict[str, str]], optional): allow specifying different aggregation methods (e.g., 'mean' for a temperature sensor, 'last' for a state indicator) for different columns during binning.. Defaults to None.

    Raises:
        ValueError: TO BE FILLED

    Returns:
        Dict[str, pd.DataFrame]: return a dictionary where keys are the split names and values are the processed DataFrames.
    """
    # Adapter design pattern
    # 'splits' can be one of two types.
    # 1. A dictionary: {'train': df_train, 'val': df_val}
    # 2. A SplitData object: SplitData(train=df_train, val=df_val)
    if isinstance(splits, dict):
        data_dict = splits
    else:
        data_dict = {k: v for k, v in splits.__dict__.items() if v is not None}

    # Iteration and Validation
    out: Dict[str, pd.DataFrame] = {}
    for name, df in data_dict.items():
        # Emptiness Check
        if df is None or df.empty:
            out[name] = pd.DataFrame()
            continue
        # Index Type Check, crucial
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{name} split must have a DateTimeIndex.")
        # Timezone Standardization, UTC is the standard
        if cfg.tz is not None:
            df = ensure_datetime_index(df, tz=cfg.tz)

        if feature_cols is not None: 
            requested_cols = list(dict.fromkeys(feature_cols))
            missing_cols = [c for c in requested_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(
                    f"Columns missing in split '{name}': {missing_cols}. "
                    "Verify the dataset contains all requested features/target."
                )
            df = df[requested_cols].copy()


        # Daily Processing Loop
        blocks = []
        # guarantee chronological order before grouping. and groups all records by their calendar date.
        for date_key, df_day in df.sort_index().groupby(df.index.date, sort=True):
            # skips days with insufficient data.
            if len(df_day) < 2:
                continue
            # resampling the day's data into K bins
            day_block = normalize_day_to_kbins(df_day, cfg.K, cfg, strategy_map=strategy_map)

            # Indexing and Structuring
            # takes the binned data for a single day (day_block) and engineers a well-structured MultiIndex
            # The Date: Which day did this data come from?
            # The Bin ID: Which time bin within that day (e.g., bin 1, bin 2, ... bin K)?
            # The Target Time: What is the exact timestamp for that bin?
            # [date, bin_id, target_time]
            if not day_block.empty:
                # Ensure the time index is named so we can reorder by name
                if day_block.index.name is None:
                    day_block.index.name = "target_time"

                # Add BOTH bin_id and date into the index
                day_block.insert(0, "date", pd.to_datetime(date_key))
                day_block = day_block.set_index(["bin_id", "date"], append=True)

                # Reorder to [date, bin_id, target_time] exactly
                day_block.index = day_block.index.reorder_levels(["date", "bin_id", "target_time"])
                # Operations on a sorted MultiIndex can be orders of magnitude faster than on an unsorted one.
                day_block = day_block.sort_index()

                blocks.append(day_block)

        out[name] = pd.concat(blocks) if blocks else pd.DataFrame()

    return out


# ------ Validation ------
# The goal is to quantify how much information is lost when you compress the original high-resolution (e.g., hourly) data into a fixed number of bins (K).
def _reconstruct_hourly_from_kbins(k_series: pd.Series, n_hours: int) -> pd.Series:
    """
    Reconstruct an hourly series from a single day's K-bin profile.

    Assumes k_series corresponds to ONE day and has a MultiIndex with levels
    ['date','bin_id','target_time'] (as produced by process_splits_to_kbins),
    or a DatetimeIndex of bin centers. The reconstruction conservatively
    averages bins over hourly intervals.

    Parameters
    ----------
    k_series : pd.Series
        K-length series of bin values for one day.
    n_hours : int
        Number of original daylight hours in that day.

    Returns
    -------
    pd.Series
        Hourly-resolution reconstruction (length = n_hours) indexed at hour centers.
    """
    # Ensure bins are in ascending bin_id order (critical for overlap math)
    if isinstance(k_series.index, pd.MultiIndex):
        names = list(k_series.index.names)
        if "bin_id" in names:
            k_series = k_series.sort_index(level="bin_id")
        # Extract bin-center timestamps
        if "target_time" in names:
            target_times = pd.DatetimeIndex(k_series.index.get_level_values("target_time"))
        else:
            # Fallback: use the third element if unnamed
            target_times = pd.DatetimeIndex([ix[-1] for ix in k_series.index])
    else:
        # Simple DatetimeIndex case
        target_times = pd.DatetimeIndex(k_series.index)

    K = len(k_series)
    bin_vals = k_series.values.astype(float)

    # Equal-phase bin edges over [0, n_hours]
    edges = np.linspace(0.0, float(n_hours), K + 1)

    # Reconstruct per-hour values by overlap-weighted averaging
    hourly = np.zeros(n_hours, dtype=float)
    for j in range(n_hours):
        h0, h1 = float(j), float(j + 1)
        b_start = int(math.floor(j * K / n_hours))
        b_end   = int(math.ceil((j + 1) * K / n_hours) - 1)
        acc = 0.0
        for b in range(b_start, b_end + 1):
            b0, b1 = edges[b], edges[b + 1]
            o0, o1 = max(h0, b0), min(h1, b1)
            overlap = max(0.0, o1 - o0)
            if overlap > 0.0:
                acc += bin_vals[b] * overlap  # hour duration = 1
        hourly[j] = acc

    # Estimate sunrise from the earliest bin center minus half a bin width (in hours)
    bin_width_hours = n_hours / K
    sunrise_est = target_times.min() - pd.Timedelta(hours=0.5 * bin_width_hours)

    # Build hour-center timestamps: sunrise + (j + 0.5) hours
    hourly_index = sunrise_est + pd.to_timedelta(np.arange(n_hours) + 0.5, unit="h")
    return pd.Series(hourly, index=hourly_index)


def validate_roundtrip_day(df_day: pd.DataFrame, day_norm: pd.DataFrame,
                           check_cols: List[str], energy_cols: Optional[List[str]] = None) -> Dict[str, float]:
    df_day = df_day.sort_index()
    n_hours = len(df_day)
    results = {}
    mae_list, rmse_list = [], []
    for col in check_cols:
        if col not in df_day.columns or col not in day_norm.columns:
            continue
        k_series = pd.Series(day_norm[col].values, index=day_norm.index)
        rec = _reconstruct_hourly_from_kbins(k_series, n_hours=n_hours)
        y_true = df_day[col].values.astype(float)
        y_pred = rec.values.astype(float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        mae_list.append(mae)
        rmse_list.append(rmse)
    results["mae_mean"]  = float(np.mean(mae_list)) if mae_list else np.nan
    results["rmse_mean"] = float(np.mean(rmse_list)) if rmse_list else np.nan

    if energy_cols is not None:
        for c in energy_cols:
            if c in df_day.columns and c in day_norm.columns:
                orig_total = float(df_day[c].sum())
                rec = _reconstruct_hourly_from_kbins(pd.Series(day_norm[c].values, index=day_norm.index), n_hours=n_hours)
                rec_total = float(rec.sum())
                results[f"energy_ratio_{c}"] = rec_total / orig_total if orig_total != 0 else np.nan
    return results


def validate_roundtrip_split(original: pd.DataFrame, normalized: pd.DataFrame,
                             feature_cols: List[str], target_energy_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if original.empty or normalized.empty:
        return pd.DataFrame()
    metrics = []
    for date_key, df_day in original.sort_index().groupby(original.index.date, sort=True):
        try:
            day_block = normalized.xs(pd.to_datetime(date_key), level="date")
        except KeyError:
            continue
        res = validate_roundtrip_day(df_day=df_day, day_norm=day_block,
                                     check_cols=feature_cols, energy_cols=target_energy_cols)
        res["date"] = pd.to_datetime(date_key)
        metrics.append(res)
    if not metrics:
        return pd.DataFrame()
    rep = pd.DataFrame(metrics).set_index("date").sort_index()
    summary = rep.mean(numeric_only=True).to_frame().T
    summary.index = [pd.NaT]
    rep = pd.concat([rep, summary])
    return rep

# ------ Build model arrays ------

def _tensor_from_norm(norm_df: pd.DataFrame, feature_cols: List[str]):
    """
    Convert normalized DF with MultiIndex ['date','bin_id','target_time']
    into a tensor (num_days, K, F). For each (date, bin_id) we take the
    first value (unique by construction), avoiding the target_time level.

    Simple Analogy:
    - Dimension 1 (Samples/Days): How many different days of data do we have?
    - Dimension 2 (Timesteps/Bins): How many time bins (K) are there in each day?
    - Dimension 3 (Features): How many different sensors (F) were measured in each bin?

    Args:
        - norm_df: normalized DataFrame.
        - feature_cols: List of features.
    - 
    """
    # returns an empty 3D tensor and an empty list of dates
    if norm_df.empty:
        return np.empty((0,0,0)), []

    norm_df = norm_df.sort_index()

    # the first dimension of our output tensor.
    dates = list(norm_df.index.get_level_values("date").unique())
    # the second dimension of the tensor. finds the maximum bin ID (e.g., if you have 48 bins, the max ID would be 47), and adds 1 to get the total count.
    # Hours (Records) with in the day.
    K = int(norm_df.index.get_level_values("bin_id").max()) + 1
    # the number of features, the third dim.
    F = len(feature_cols)

    # (num_days, K_bins, F). Filled with (Not a Number)
    X = np.full((len(dates), K, F), np.nan, dtype=float)

    # builds the tensor one feature at a time.
    for j, col in enumerate(feature_cols):
        if col not in norm_df.columns:
            continue

        # (date, bin_id) -> value; drop target_time by taking first. 2D matrix (mat) where rows are dates and columns are bin IDs.
        mat = (
            norm_df[col]                                                                # 1. Select Series
            .groupby(level=["date", "bin_id"])                                          # 2. Group by (date, bin)
            .first()                         # one value per (date, bin_id)             # 3. Collapse time level
            .unstack("bin_id")               # rows: date, cols: bin_id                 # 4. Pivot bins to columns
            .reindex(index=dates, columns=range(K))                                     # 5. Enforce shape
        )

        X[:, :, j] = mat.values # takes the 2D matrix of data for the current feature and slots it perfectly into its designated "slice" of the final 3D tensor X.

    return X, dates



def build_model_arrays(norm_df: pd.DataFrame, 
                       feature_cols: List[str], 
                       target_col: str,
                       history_days: int = 7, 
                       horizon_days: int = 1
                       ):
    
    """sliding window generator: take a continuous block of time-series data and chop it up into many smaller, 
    overlapping samples that are suitable for training a sequence-to-sequence deep learning model (like an LSTM, GRU, or Transformer).
    The function slides this two-part window (Past, Future) one day at a time, generating a new training sample at each step.

    What does the function expect data to look like?
    1.	MultiIndex rows with levels:
        - Level 0: date (one row-group per day)
        - Level 1: bin axis (e.g., bin_id from 0…K-1 OR Record hour index )
    2.	Columns include all feature_cols plus the target_col.    
    3.	Rows are sorted by the MultiIndex.
    4. 

    Example of X:

    X: A 4D tensor of shape (samples, history_days, K, F) — e.g., (100, 7, 48, 15).
        - 100: We have 100 distinct training examples.
        - 7: Each example uses 7 days of historical data.
        - 48: Each day is divided into 48 time bins.
        - 15: Each time bin has 15 features.
    
    Y: A 3D tensor of shape (samples, horizon_days, K) — e.g., (100, 1, 48).
        - 100: For each of the 100 training examples...
        - 1: ...the target is to predict 1 day into the future.
        - 48: For that future day, we predict the target value for all 48 time bins.
    
        

    Args:
        - feature_cols:  is the list of columns that will be used as model inputs (the predictors).
        - history_days: A fixed number of past days (history_days) used as input features.
        - The Future (Prediction Horizon): A fixed number of future days (horizon_days) used as the target label to predict.

    Returns:
        - X: A 4D tensor of shape (samples, history_days, K, F)
        - Y: A 3D tensor of shape (samples, horizon_days, K)
        - labels: A list of 100 dates, where each date corresponds to the day being predicted for each sample.
    """

    # Once for all the input features (feature_cols) to create a 3D tensor X_all with the shape (num_days, K, F).
    X_all, dates = _tensor_from_norm(norm_df, feature_cols=feature_cols)
    # Once for just the single target variable (target_col) to create a tensor y_all.
    y_all, _ = _tensor_from_norm(norm_df, feature_cols=[target_col])
    # squeezes out the last dimension, making y_all a more convenient 2D matrix of shape (num_days, K).
    y_all = y_all[:,:,0]

    num_days = X_all.shape[0]
    if num_days == 0:
        return np.empty((0,history_days,0,0)), np.empty((0,horizon_days,0)), []
    K = X_all.shape[1]
    F = X_all.shape[2]
    # the total number of days leftover after accounting for one full history+horizon window. window can start at index 0, so we add 1.
    samples = max(0, num_days - history_days - horizon_days + 1)

    X_list, Y_list, labels = [], [], []
    for i in range(samples):
        past = slice(i, i+history_days)
        fut  = slice(i+history_days, i+history_days+horizon_days)
        X_i = X_all[past,:,:]
        Y_i = y_all[fut,:]
        X_list.append(X_i)
        Y_list.append(Y_i)
        labels.append(dates[i+history_days+horizon_days-1])

    # converts the list of individual samples into the final, single, massive tensor required by the deep learning model.
    X = np.stack(X_list, axis=0) if X_list else np.empty((0,history_days,K,F))
    Y = np.stack(Y_list, axis=0) if Y_list else np.empty((0,horizon_days,K))

    # X: A 4D tensor of shape (samples, history_days, K, F)
    return X, Y, labels


def to_fixedgrid_multiindex(df, timestamp_col="measurement_time", expected_T=None):
    """
    Convert a single-index time series (fixed steps per day) into the
    MultiIndex shape expected by build_model_arrays: index = [date, bin_id].
    """
    df = df.copy()
    # ensure timestamp column exists and is datetime
    if df.index.name == timestamp_col:
        df = df.reset_index()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.sort_values(timestamp_col)

    # derive the daily key and the within-day position
    df["date"] = df[timestamp_col].dt.tz_convert("UTC").dt.date
    df["bin_id"] = df.groupby("date").cumcount()

    # optional consistency check
    if expected_T is not None:
        per_day_last = df.groupby("date")["bin_id"].max()
        if not (per_day_last.eq(expected_T - 1)).all():
            bad = per_day_last[~per_day_last.eq(expected_T - 1)]
            raise ValueError(
                f"Inconsistent per-day length. Expected {expected_T}, "
                f"found {bad.add(1).unique().tolist()} for dates {bad.index.tolist()[:5]}..."
            )

    # set the expected MultiIndex and drop the raw timestamp
    df = df.set_index(["date", "bin_id"]).sort_index()
    return df.drop(columns=[timestamp_col])


__all__ = [
    "SplitData", "KBinConfig", "DEFAULT_STRATEGY",
    "ensure_datetime_index",
    "process_splits_to_kbins",
    "build_model_arrays",
    "validate_roundtrip_split",
    "normalize_day_to_kbins",
    
]

if __name__ == "__main__":

    pass

    
