"""
Purpose: This module is responsible for the initial one-time setup of loading raw data and defining the training/validation/test splits. 
Its output is the JSON files in the splits/ directory.

Author: Muhammad hassan
Date: 30.10.2025
Project: MPISSM (Multimodal Physics Informed State Space Models)


This code is typically run only once at the beginning of a project or when the raw data changes. 

Content
- load_data()
- fixed_holdout_split()
- rolling_origin_evaluation()
- save_splits_info()

"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Tuple, List, Dict
import json

# Set global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def load_data(csv_path: str, date_col: str = 'timeStamp') -> pd.DataFrame:
    """
    Load CSV data with proper UTC timestamp handling.
    
    Parameters:
    -----------
    csv_path : str - Path to CSV file
    date_col : str - Name of timestamp column
    
    Returns:
    --------
    DataFrame with DatetimeIndex in UTC
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found. Available: {list(df.columns)}")
    
    # Parse timestamps and ensure UTC timezone
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
    
    # Remove any rows with invalid timestamps
    invalid_dates = df[date_col].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: Removed {invalid_dates} rows with invalid timestamps")
        df = df.dropna(subset=[date_col])
    
    # Set as index and sort
    df = df.set_index(date_col).sort_index()
    
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Timezone: {df.index.tz}")
    
    return df


def fixed_holdout_split(
    df: pd.DataFrame,
    train_start: str = '2014-01-01',
    train_end: str = '2015-09-30',
    val_start: str = '2015-10-01',
    val_end: str = '2015-12-31',
    test_start: str = '2016-01-01',
    test_end: str = '2016-12-31'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Perform a fixed chronological train/validation/test split.
    Assumes df has a UTC-aware DatetimeIndex.
    
    IMPORTANT: Validation period should NOT overlap with training period!
    
    Parameters:
    -----------
    df : pd.DataFrame with DatetimeIndex in UTC
    train_start, train_end : str - Training period boundaries (inclusive)
    val_start, val_end : str - Validation period boundaries (inclusive)
    test_start, test_end : str - Test period boundaries (inclusive)
    
    Returns:
    --------
    train_df, val_df, test_df : DataFrames for each split
    split_indices : Dict with timestamp strings (for reproducibility)
    """
    df = df.copy()
    
    # Ensure UTC-aware timestamps for boundaries
    train_start = pd.Timestamp(train_start, tz='UTC')
    train_end = pd.Timestamp(train_end, tz='UTC')
    val_start = pd.Timestamp(val_start, tz='UTC')
    val_end = pd.Timestamp(val_end, tz='UTC')
    test_start = pd.Timestamp(test_start, tz='UTC')
    test_end = pd.Timestamp(test_end, tz='UTC')
    
    # Validate no overlap
    if val_start <= train_end:
        raise ValueError(
            f"Validation start ({val_start.date()}) must be AFTER "
            f"training end ({train_end.date()}) to avoid data leakage!"
        )
    if test_start <= val_end:
        raise ValueError(
            f"Test start ({test_start.date()}) must be AFTER "
            f"validation end ({val_end.date()}) to avoid data leakage!"
        )
    
    # Filter by date ranges
    train_df = df.loc[train_start:train_end].copy()
    val_df = df.loc[val_start:val_end].copy()
    test_df = df.loc[test_start:test_end].copy()
    
    # Store indices as timestamp strings (more robust than integer indices)
    split_indices = {
        'train': train_df.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist(),
        'val': val_df.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist(),
        'test': test_df.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist()
    }
    
    # Print split summary
    print("\n" + "="*60)
    print("FIXED HOLDOUT SPLIT")
    print("="*60)
    print(f"Training:   {train_start.date()} to {train_end.date()} → {len(train_df):,} records")
    print(f"Validation: {val_start.date()} to {val_end.date()} → {len(val_df):,} records")
    print(f"Test:       {test_start.date()} to {test_end.date()} → {len(test_df):,} records")
    print(f"Total:      {len(train_df) + len(val_df) + len(test_df):,} records")
    print(f"Original:   {len(df):,} records")
    
    # Check for gaps
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("\n WARNING: One or more splits are empty!")
    
    print("="*60)
    
    return train_df, val_df, test_df, split_indices


def rolling_origin_evaluation(
    df: pd.DataFrame,
    start_train: str = '2014-01-31',
    end_train: str = '2015-09-30',
    freq: str = 'MS'  # Month Start
) -> List[Dict[str, any]]:
    """
    Perform rolling-origin evaluation with EXPANDING window.
    
    At each step:
    - Training: All data from beginning up to cutoff date
    - Validation: Next month after cutoff
    
    This is the standard expanding-window approach for time series.
    
    Parameters:
    -----------
    df : pd.DataFrame with DatetimeIndex in UTC
    start_train : str - First training cutoff date
    end_train : str - Last training cutoff date
    freq : str - Frequency for rolling (default: 'MS' = month start)
    
    Returns:
    --------
    List of dicts with train/val periods and timestamp indices
    """
    df = df.copy()
    
    # Create UTC-aware date range for training cutoff points
    start_train = pd.Timestamp(start_train, tz='UTC')
    end_train = pd.Timestamp(end_train, tz='UTC')
    
    date_range = pd.date_range(start=start_train, end=end_train, freq=freq, tz='UTC')
    
    results = []
    data_start = df.index.min()
    
    for train_end_date in date_range:
        # Validation: NEXT month after train_end_date
        val_start = train_end_date + pd.DateOffset(days=1)
        val_end = val_start + pd.offsets.MonthEnd(0)
        
        # Check if validation period exists in data
        if val_start > df.index.max():
            break
        
        # Training: All data from start to train_end_date
        train_mask = df.index <= train_end_date
        val_mask = (df.index >= val_start) & (df.index <= val_end)
        
        train_subset = df[train_mask]
        val_subset = df[val_mask]
        
        if len(train_subset) == 0 or len(val_subset) == 0:
            print(f"Skipped fold: train_end={train_end_date.date()}, "
                  f"train_size={len(train_subset)}, val_size={len(val_subset)}")
            continue
        
        results.append({
            'fold_id': len(results) + 1,

            'train_period': {
                'start': str(data_start),
                'end': str(train_end_date)
                            },

            'val_period': {
                'start': str(val_start),
                'end': str(val_end)
                            },

            'train_size': len(train_subset),
            'val_size': len(val_subset),
            'train_indices': train_subset.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist(), # trining indees for different folds
            'val_indices': val_subset.index.strftime('%Y-%m-%d %H:%M:%S%z').tolist()
        })
    
    # Print summary
    print("\n" + "="*60)
    print("ROLLING ORIGIN EVALUATION")
    print("="*60)
    print(f"Total folds: {len(results)}")
    print(f"Frequency: {freq}")
    print(f"Data range: {data_start.date()} to {df.index.max().date()}")
    print("\nFold Summary:")
    for fold in results:
        print(f"  Fold {fold['fold_id']}: "
              f"Train [{fold['train_period']['start'][:10]} to {fold['train_period']['end'][:10]}] "
              f"({fold['train_size']:,} records) → "
              f"Val [{fold['val_period']['start'][:10]} to {fold['val_period']['end'][:10]}] "
              f"({fold['val_size']:,} records)")
    print("="*60)
    
    return results


def save_splits_info(
    fixed_splits: Dict[str, List[str]],
    rolling_splits: List[Dict],
    experiment_name: str = 'exp-001',
    output_dir: str = 'splits'
):
    """
    Save split metadata to JSON for reproducibility.
    
    Parameters:
    -----------
    fixed_splits : Dict with train/val/test timestamp indices
    rolling_splits : List of rolling origin fold information
    output_dir : str - Directory to save JSON files
    """
    output_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Add metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'timezone': 'UTC'
    }
    
    fixed_output = {
        'metadata': metadata,
        'splits': fixed_splits
    }
    
    rolling_output = {
        'metadata': metadata,
        'folds': rolling_splits
    }
    
    with open(os.path.join(output_dir, f'{experiment_name}fixed_holdout_splits.json'), 'w') as f:
        json.dump(fixed_output, f, indent=2)
    
    with open(os.path.join(output_dir, f'{experiment_name}rolling_origin_splits.json'), 'w') as f:
        json.dump(rolling_output, f, indent=2)
    
    print(f"\nSplit metadata saved to '{output_dir}/' directory")




