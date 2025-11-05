"""Data preparation entry point."""

import argparse
from typing import List, Optional

import pandas as pd

from src.config import (
    DEFAULT_DATA_PREFIX,
    FEATURE_SETS,
    HISTORY_DAYS,
    HORIZON_DAYS,
    K_BINS,
    USE_KBINS,
    PROCESSED_DATA_PATH,
    TARGET_COL,
    TIMESTAMP_COL,
    resolve_feature_list,
)
from src.data_preparation import load_data, rolling_origin_evaluation, save_splits_info
from src.preprocessing import KBinConfig, build_model_arrays, process_splits_to_kbins
from src.utils import DataManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare K-bin arrays for experiments.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=PROCESSED_DATA_PATH,
        help="Path to the processed feature CSV.",
    )
    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=TIMESTAMP_COL,
        help="Name of the timestamp column in the CSV.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=sorted(FEATURE_SETS.keys()),
        help="Predefined feature group to use.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        help="Explicit list of feature columns to use (overrides --feature-set).",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        help="Prefix for saved numpy arrays. Defaults to feature-set specific name.",
    )
    parser.add_argument(
        "--k-bins",
        type=int,
        default=K_BINS,
        help="Number of bins per day.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=HISTORY_DAYS,
        help="Number of historic days per sample.",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=HORIZON_DAYS,
        help="Number of forecast days per sample.",
    )
    return parser.parse_args()


def determine_features(feature_set: Optional[str], features: Optional[List[str]]) -> List[str]:
    if features:
        return [f.strip() for f in features]
    return resolve_feature_list(feature_set)


def infer_data_prefix(feature_set: Optional[str], explicit_prefix: Optional[str]) -> str:
    if explicit_prefix:
        return explicit_prefix
    if feature_set:
        return f"solar_kbins_{feature_set}"
    return DEFAULT_DATA_PREFIX


def main():
    args = parse_args()
    feature_cols = determine_features(args.feature_set, args.features)
    data_prefix = infer_data_prefix(args.feature_set, args.data_prefix)
    selected_columns = sorted(set(feature_cols + [TARGET_COL]))

    if not feature_cols:
        raise ValueError("Feature list is empty. Provide --feature-set or --features with valid columns.")

    print("\n--- Step 1: Loading processed data and creating splits ---")
    df = load_data(args.input_csv, date_col=args.timestamp_col)

    missing = sorted(set(selected_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    rolling_splits = rolling_origin_evaluation(df)
    save_splits_info({}, rolling_splits)


    if USE_KBINS:
        print("\n--- Step 2: Converting full dataset to K-Bins format ---")
        cfg = KBinConfig(K=args.k_bins)
        processed = process_splits_to_kbins({"full_data": df}, cfg, feature_cols=selected_columns)
        norm_df = processed.get("full_data")
        if norm_df is None or norm_df.empty:
            raise ValueError("Normalized dataframe is empty after K-bin processing.")

        print("\n--- Step 3: Building model arrays (X, Y) ---")
        X, Y, labels_list = build_model_arrays(
            norm_df,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            history_days=args.history_days,
            horizon_days=args.horizon_days,
        )
    else:
        print("\n--- Step 2: Building model arrays (X, Y) ---")
        X, Y, labels_list = build_model_arrays(
            df,
            feature_cols=selected_columns,
            target_col=TARGET_COL)
        

    labels_index = pd.to_datetime(labels_list)
    if getattr(labels_index, "tz", None) is None:
        labels_index = labels_index.tz_localize("UTC")
    labels_df = pd.DataFrame(index=labels_index)

    print("\n--- Step 4: Saving processed arrays for training ---")
    data_manager = DataManager()
    data_manager.save_arrays(
        X,
        Y,
        labels_df,
        filename_prefix=data_prefix,
        feature_cols=feature_cols,
        target_col=TARGET_COL,
        metadata={
            "input_csv": args.input_csv,
            "timestamp_col": args.timestamp_col,
            "feature_set": args.feature_set,
            "history_days": args.history_days,
            "horizon_days": args.horizon_days,
            "k_bins": args.k_bins,
        },
    )

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
