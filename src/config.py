"""
Central configuration for the solar forecasting experiments.
Keeping all core settings in one place helps ensure reproducibility.
"""

from typing import List, Sequence, Union

# ---------------------------------------------------------------------------
# Data & feature settings
# ---------------------------------------------------------------------------

PROCESSED_DATA_PATH = "data/processed/dayTime_NAM_dayahead_features_processed.csv"
TIMESTAMP_COL = "measurement_time"

HISTORY_DAYS = 7
HORIZON_DAYS = 1
TARGET_COL = "CSI_ghi"

# Site metadata (used for clear-sky calculations and NAM comparisons)
SITE_LATITUDE = 38.642
SITE_LONGITUDE = -121.148
SITE_ALTITUDE = 0.0
SITE_TIMEZONE = "UTC"

# Columns available in the processed NAM day-ahead feature dataset.
AVAILABLE_FEATURES: List[str] = [
    "solar_zenith",
    "time_gap_hours",
    "time_gap_norm",
    "day_boundary_flag",
    "hour_progression",
    "absolute_hour",
    "CSI_ghi",
    "CSI_dni",
    "season_flag",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "nam_ghi",
    "nam_dni",
    "nam_cc",
]

# Curated feature groups to simplify experimentation.
FEATURE_SETS = {
    "all": AVAILABLE_FEATURES,
    "nam_forecast": ["nam_ghi", "nam_dni", "nam_cc"],
    "irradiance": ["CSI_ghi", "CSI_dni"],
    "temporal_geometry": [
        "solar_zenith",
        "time_gap_norm",
        "day_boundary_flag",
        "hour_progression",
        "absolute_hour",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ],
}

DEFAULT_FEATURE_SET = "all"


def resolve_feature_list(selection: Union[str, Sequence[str], None] = None) -> List[str]:
    """Return a concrete feature list from a named set or explicit sequence."""
    selection = DEFAULT_FEATURE_SET if selection is None else selection
    if isinstance(selection, str):
        if selection not in FEATURE_SETS:
            raise KeyError(
                f"Unknown feature set '{selection}'. "
                f"Available: {list(FEATURE_SETS.keys())}"
            )
        return list(FEATURE_SETS[selection])
    return [str(col) for col in selection]


FEATURE_COLS = resolve_feature_list(DEFAULT_FEATURE_SET)
DEFAULT_DATA_PREFIX = f"solar_kbins_{DEFAULT_FEATURE_SET}"

# Legacy defaults for any remaining scripts that rely on the old interface.
MODEL_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.1,
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
}

# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------

LSTM_CONFIG = {
    "experiment_name": "solar_lstm_kbins",
    "model_type": "LSTM",
    "model_config": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "bidirectional": True,
    },
    "data_prefix": DEFAULT_DATA_PREFIX,
    "splits_file": None,
    "feature_cols": FEATURE_COLS,
    "feature_selection": DEFAULT_FEATURE_SET,
    "target_col": TARGET_COL,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "early_stopping_patience": 20,
    "max_folds": 3,
}

MATNET_CONFIG = {
    "experiment_name": "solar_matnet_kbins",
    "model_type": "MATNet",
    "model_config": {
        "d_model": 256,
        "num_heads": 8,
        "num_layers": 4,
        "dropout": 0.2,
    },
    "data_prefix": DEFAULT_DATA_PREFIX,
    "feature_cols": FEATURE_COLS,
    "feature_selection": DEFAULT_FEATURE_SET,
    "target_col": TARGET_COL,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "early_stopping_patience": 20,
    "max_folds": 3,
}
