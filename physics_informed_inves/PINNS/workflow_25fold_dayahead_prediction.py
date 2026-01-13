"""
25-Fold Cross-Validation with Day-Ahead Prediction Workflow

This script implements a comprehensive experiment with:
- 25 patches/folds for training (instead of standard 2-4 folds)
- Day-ahead prediction (28-52 hours, similar to NWP forecasts)
- Main station (station 7) as target
- Comparison against ground truth and benchmarks (Smart Persistence, NWP)

This demonstrates the full capability of the refactored modular package
for production-ready solar power forecasting research.
"""

import os
import sys
import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from pvlib import location, pvsystem, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Add path to modular package
sys.path.insert(0, '/home/muhammadhassan/App_v02/physics_informed_inves/PINNS')

from physics_residual_mamba import (
    # Configuration
    PhysicsResidualMambaConfigs,
    
    # Data
    MultiStepDataset,
    prepare_rolling_folds,
    add_cyclic_features,
    add_past_rolling_stats,
    
    # Models
    PhysicsResidualMamba,
    MMPISSM_Model_01,
    VanillaLSTM,
    VanillaMamba,
    
    # Losses
    PhysicsPVLoss,
    
    # Evaluation
    evaluate_fold_physics,
    evaluate_fold_base,
    calculate_metrics,
    calculate_improvement,
    calculate_confidence_interval,
    
    # Visualization
    plot_model_performance,
    plot_benchmark_summary,
    plot_multi_model_comparison,
    
    # Utils
    set_random_seed,
    get_device,
    get_random_seed,
    setup_logging,
    ExperimentLogger
)

# Import training functions
from physics_residual_mamba.training.trainers import (
    train_one_epoch_physics,
    train_one_epoch_base,
)

import torch
import torch.optim as optim


# =============================================================================
# PART 1: Station Metadata Management
# =============================================================================

def get_station_metadata(station_num: int, df: pd.DataFrame) -> Dict[str, any]:
    """
    Retrieves and parses metadata for a specific station from DataFrame.
    
    Args:
        station_num: The station number (e.g., 7 for station07)
        df: DataFrame containing station metadata
        
    Returns:
        Dictionary with clean numerical values ready for simulation
    """
    station_id_str = f"station{str(station_num).zfill(2)}"
    
    # Filter dataframe
    station_row = df[df['Station_ID'] == station_id_str]
    
    if station_row.empty:
        raise ValueError(f"Station ID {station_id_str} not found in DataFrame.")
    
    row = station_row.iloc[0]

    # Helper: Parse "Key:Value" blocks
    def parse_kv_string(text_block):
        data = {}
        if isinstance(text_block, str):
            for item in text_block.split('\n'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    data[k.strip()] = v.strip()
        return data

    # Helper: Extract numeric values safely
    def extract_num(val, default=0.0):
        if pd.isna(val):
            return default
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else default

    # Parse complex text columns
    module_items = parse_kv_string(row.get('Module', ''))
    inverter_items = parse_kv_string(row.get('Inverters', ''))
    layout_items = parse_kv_string(row.get('Layout', ''))

    # Build final clean dictionary
    metadata = {
        # Identity & Location
        'Station_ID': station_id_str,
        'Longitude': float(row['Longitude']),
        'Latitude': float(row['Latitude']),
        'Array_Tilt': row['Array_Tilt'],
        
        # System Capacity & Dimensions
        'Capacity': float(row['Capacity']),
        'Panel_Size': float(row.get('Panel_Size', 1.62)),
        'Total_Panel_Number': int(row.get('Panel_Number', 0)),
        'PV_Technology': row['PV_Technology'],

        # Module Specs (cleaned to float)
        'Module_Pmax': extract_num(module_items.get('Pmax'), default=250),
        'Module_Vmpp': extract_num(module_items.get('Vmpp'), default=30),
        'Module_Impp': extract_num(module_items.get('Impp'), default=8),
        
        # Inverter Specs (cleaned to float)
        'Inverter_Rated_Power': extract_num(inverter_items.get('Rated power'), default=500),
        'Inverter_Max_DC_Voltage': extract_num(inverter_items.get('Max. DC voltage'), default=1000),
        
        # Layout Specs (cleaned to int)
        'Modules_per_String': int(extract_num(layout_items.get('modules per string'), default=20)),
        'Strings_per_Inverter': int(extract_num(layout_items.get('strings per inverter'), default=100)),
    }
    
    return metadata


def calculate_clearsky_indices(metadata: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates both Irradiance (K_CS) and Power (K_PV) Clear Sky Indices.
    
    Args:
        metadata: Station metadata from get_station_metadata()
        df: Time-series data with 'lmd_totalirrad' and 'power' columns
        
    Returns:
        DataFrame with added columns: 'GHI_clr', 'K_CS', 'P_CLR', 'K_PV'
    """
    station_id = metadata.get('Station_ID', 'Unknown')
    print(f"--- Processing {station_id} ---")
    
    # 1. SETUP LOCATION & TIME
    lat = metadata['Latitude']
    lon = metadata['Longitude']
    
    site = location.Location(lat, lon, tz='UTC')
    
    # Ensure index is Datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date_time'] = pd.to_datetime(df['date_time'])
        df.set_index('date_time', inplace=True)
    
    # 2. CALCULATE CLEAR SKY IRRADIANCE (GHI_clr)
    print("Calculating Clear Sky Irradiance (Ineichen model)...")
    
    cs = site.get_clearsky(df.index, model='ineichen')
    df['GHI_clr'] = cs['ghi']
    
    # 3. CALCULATE IRRADIANCE CLEAR SKY INDEX (K_CS)
    if 'lmd_totalirrad' in df.columns:
        meas_ghi = df['lmd_totalirrad']
        
        # Calculate Index (Filter: Only when model expects sun > 10 W/m²)
        df['K_CS_day'] = np.where(
            df['GHI_clr'] > 10, 
            meas_ghi / df['GHI_clr'], 
            0.0
        )
        df['K_CS_dayNight'] = meas_ghi / df['GHI_clr']
        df['K_CS'] = df['K_CS_day']
        
        # Clip outliers
        df['K_CS'] = df['K_CS'].clip(lower=0.0, upper=1.25)
    else:
        print("Warning: 'lmd_totalirrad' column missing. K_CS not calculated.")
    
    # 4. SETUP PV SYSTEM MODEL (For P_CLR)
    print("Setting up PV System Model...")
    
    # Extract Tilt
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except Exception as e:
        print(f"Error extracting tilt: {e}. Assuming 33 degrees")
        tilt = 33.0
    
    # Default Azimuth: 180 (South) for Northern Hemisphere
    azimuth = 180 
    
    # Module: Yingli YL250P-29b
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    
    # Inverter: Advanced Energy 500kW
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_']
    ivt_para["Pdco"], ivt_para['Vdco'], ivt_para["Vdcmax"], ivt_para['Idcmax'] = 567000, 315, 1000, 1134
    ivt_para["Mppt_low"], ivt_para['Mppt_high'] = 460, 950
    inverter_parameters = ivt_para
    
    # Temperature Model (Open Rack)
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        module_parameters=module_params,
        inverter_parameters=inverter_parameters,
        temperature_model_parameters=temp_model,
        modules_per_string=metadata['Modules_per_String'],
        strings_per_inverter=metadata['Strings_per_Inverter']
    )
    
    # 5. RUN MODEL CHAIN FOR POWER
    print("Running PV Power Simulation...")
    
    mc = modelchain.ModelChain(
        system, site, 
        transposition_model='perez',
        solar_position_method='nrel_numpy',
        aoi_model='physical', 
        spectral_model='no_loss'
    )
    
    mc.run_model(cs)
    
    # 6. SCALE & CALCULATE POWER INDEX (K_PV)
    station_total_capacity_watts = metadata['Capacity'] * 1000  # kW -> Watts
    block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    scaling_factor = station_total_capacity_watts / block_dc_watts
    
    p_clr_watts = mc.results.ac.fillna(0) * scaling_factor
    df['P_CLR'] = p_clr_watts / 1_000_000  # Convert to MW
    
    # 7. CALCULATE POWER CLEAR SKY INDEX (K_PV)
    if 'power' in df.columns:
        meas_power = df['power']
        
        # Check units: If max > 500, likely kW, divide by 1000
        if meas_power.max() > 500:
            meas_power = meas_power / 1000.0
            
        # Calculate Index (Filter: Only when model expects > 0.05 MW)
        df['K_PV'] = np.where(
            df['P_CLR'] > 1.0, 
            meas_power / df['P_CLR'], 
            0.0
        )
        
        # Clip for cleanliness
        df['K_PV'] = df['K_PV'].clip(lower=0.0, upper=1.5)
    else:
        print("Warning: 'power' column missing. K_PV not calculated.")
    
    print(f"Done. GHI_clr mean: {df['GHI_clr'].mean():.1f} W/m²")
    print(f"Done. P_CLR mean: {df['P_CLR'].mean():.4f} MW")
    
    return df


def calculate_nwp_power(metadata: Dict, df: pd.DataFrame) -> pd.Series:
    """
    Calculates expected PV Power output (MW) based purely on NWP Weather Forecasts.
    
    This acts as a "Physics-Based Baseline" to compare against the AI model.
    
    Args:
        metadata: Station metadata from get_station_metadata()
        df: DataFrame containing 'nwp_' columns
        
    Returns:
        Series: The calculated power in MW, aligned with df.index
    """
    station_id = metadata.get('Station_ID', 'Unknown')
    print(f"--- Calculating NWP Physical Power for {station_id} ---")
    
    # 1. SETUP LOCATION & TIME
    lat = metadata['Latitude']
    lon = metadata['Longitude']
    
    site = location.Location(lat, lon, tz='UTC')
    
    # Ensure DataFrame index is Datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 2. PREPARE WEATHER DATA (Map NWP -> PVLib Standard)
    weather_nwp = pd.DataFrame(index=df.index)
    
    # Map available columns
    weather_nwp['ghi'] = df['nwp_globalirrad']
    weather_nwp['dni'] = df['nwp_directirrad']
    weather_nwp['temp_air'] = df['nwp_temperature']
    weather_nwp['wind_speed'] = df['nwp_windspeed']
    
    # Handle Pressure (Default to 101325 Pa if missing)
    if 'nwp_pressure' in df.columns:
        weather_nwp['pressure'] = df['nwp_pressure'] * 100  # hPa -> Pa
    else:
        weather_nwp['pressure'] = 101325.0
    
    # 3. CALCULATE MISSING DHI (Diffuse Horizontal Irradiance)
    print("Calculating DHI from GHI and DNI...")
    
    solpos = site.get_solarposition(weather_nwp.index)
    zenith_rad = np.radians(solpos['zenith'])
    
    weather_nwp['dhi'] = weather_nwp['ghi'] - (weather_nwp['dni'] * np.cos(zenith_rad))
    
    # Physics Check: Irradiance cannot be negative
    weather_nwp['dhi'] = weather_nwp['dhi'].clip(lower=0.0)
    weather_nwp['dni'] = weather_nwp['dni'].clip(lower=0.0)
    weather_nwp['ghi'] = weather_nwp['ghi'].clip(lower=0.0)
    
    # 4. DEFINE PV SYSTEM (Hardware Specs)
    print("Setting up PV System for NWP simulation...")
    
    # Extract Tilt
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except:
        tilt = 33.0
        
    # Module: Yingli YL250P-29b
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    
    # Inverter: Advanced Energy 500kW
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_']
    ivt_para["Pdco"], ivt_para['Vdco'], ivt_para["Vdcmax"], ivt_para['Idcmax'] = 567000, 315, 1000, 1134
    ivt_para["Mppt_low"], ivt_para['Mppt_high'] = 460, 950
    inverter_parameters = ivt_para
    
    # Temp Model
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=180,
        module_parameters=module_params,
        inverter_parameters=inverter_parameters,
        temperature_model_parameters=temp_model,
        modules_per_string=metadata['Modules_per_String'],
        strings_per_inverter=metadata['Strings_per_Inverter']
    )
    
    # 5. RUN PHYSICAL SIMULATION (ModelChain)
    print("Running ModelChain with NWP Weather...")
    
    mc = modelchain.ModelChain(
        system, site, 
        transposition_model='perez',
        solar_position_method='nrel_numpy',
        aoi_model='physical', 
        spectral_model='no_loss'
    )
    
    mc.run_model(weather_nwp)
    
    # 6. SCALE TO FULL STATION CAPACITY
    block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    station_total_capacity_watts = metadata['Capacity'] * 1000
    
    if block_dc_watts > 0:
        scaling_factor = station_total_capacity_watts / block_dc_watts
    else:
        scaling_factor = 1.0
        
    nwp_power_mw = (mc.results.ac.fillna(0) * scaling_factor) / 1_000_000
    
    # Final Clip (Power cannot be negative)
    nwp_power_mw = nwp_power_mw.clip(lower=0.0)
    
    print(f"Done. Mean Predicted Power: {nwp_power_mw.mean():.4f} MW")
    
    return nwp_power_mw


# =============================================================================
# PART 2: Feature Engineering
# =============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training.
    
    Adds:
    - Cyclic time encodings (hour, day, month, season)
    - Rolling statistics (causal - past only)
    - Ensures DatetimeIndex
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added features
    """
    data = df.copy()
    
    # 1. Ensure DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date_time' in data.columns:
            data['date_time'] = pd.to_datetime(data['date_time'])
            data.set_index('date_time', inplace=True)
    
    # 2. Cyclic Encoding for Hour (0-23)
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    
    # 3. Cyclic Encoding for Day of Year (1-365)
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
    
    # 4. Months
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    
    # 5. Seasons
    data['season_sin'] = np.sin(2 * np.pi * data.index.month / 4)
    data['season_cos'] = np.cos(2 * np.pi * data.index.month / 4)
    
    # 6. Rolling Statistics (causal - past only)
    rolling_hours = [3, 5, 8]
    cols_to_roll = ["power", "lmd_totalirrad", "lmd_temperature", "lmd_windspeed"]
    
    for col in cols_to_roll:
        if col in data.columns:
            for h in rolling_hours:
                data[f'{col}_mean_{h}h'] = data[col].shift(1).rolling(f'{h}h', min_periods=1).mean()
                data[f'{col}_std_{h}h'] = data[col].shift(1).rolling(f'{h}h', min_periods=1).std()
    
    # Fill NaN values with 0.0
    data = data.fillna(0.0)
    
    print(f"Prepared features: {len(data.columns)} columns")
    
    return data


# =============================================================================
# PART 3: 25-Fold Cross-Validation with Day-Ahead Prediction
# =============================================================================

def run_25fold_dayahead_experiment(
    target_station: int = 7,
    n_patches: int = 25,
    prediction_horizon_hours: int = 48,  # 48 hours = 2 days
    epochs_per_patch: int = 5,
    random_seed: int = 42,
    use_spatial: bool = False,
    output_dir: str = './results_25fold_dayahead'
):
    """
    Run 25-fold cross-validation with day-ahead prediction.
    
    This experiment uses 25 patches/folds where each fold is trained
    for a few epochs (5) and then used to predict day-ahead.
    
    Args:
        target_station: Target station number (default 7)
        n_patches: Number of training patches/folds (default 25)
        prediction_horizon_hours: Prediction horizon in hours (default 48)
        epochs_per_patch: Training epochs per patch (default 5)
        random_seed: Random seed for reproducibility
        use_spatial: Whether to add spatial features (default False)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all results and metrics
    """
    print("=" * 80)
    print("25-FOLD CROSS-VALIDATION WITH DAY-AHEAD PREDICTION")
    print("=" * 80)
    
    # Set reproducibility
    set_random_seed(random_seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Random Seed: {random_seed}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # 1. LOAD METADATA
    print("\n" + "=" * 80)
    print("STEP 1: Loading Station Metadata")
    print("=" * 80)
    
    abs_path = os.path.abspath("/home/muhammadhassan/App_v02/physics_informed_inves/PVODdatasets_v1")
    metadata_path = os.path.join(abs_path, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path)
    
    station_metadata = get_station_metadata(target_station, metadata_df)
    print(f"\nStation: {station_metadata['Station_ID']}")
    print(f"Capacity: {station_metadata['Capacity']} kW")
    print(f"Location: ({station_metadata['Latitude']:.4f}, {station_metadata['Longitude']:.4f})")
    print(f"Tilt: {station_metadata['Array_Tilt']}")
    print(f"Module: {station_metadata['Module_Pmax']} Wp")
    print(f"Inverter: {station_metadata['Inverter_Rated_Power']} kW")
    
    # 2. LOAD STATION DATA
    print("\n" + "=" * 80)
    print("STEP 2: Loading Station Data")
    print("=" * 80)
    
    station_path = os.path.join(abs_path, f"station{str(target_station).zfill(2)}.csv")
    df = pd.read_csv(station_path)
    print(f"Loaded {len(df)} rows from {station_path}")
    
    # 3. CALCULATE CLEAR SKY INDICES
    print("\n" + "=" * 80)
    print("STEP 3: Calculating Clear Sky Indices")
    print("=" * 80)
    
    df = calculate_clearsky_indices(station_metadata, df)
    
    # 4. CALCULATE NWP POWER
    print("\n" + "=" * 80)
    print("STEP 4: Calculating NWP Physical Power")
    print("=" * 80)
    
    df['NWP_Power_MW'] = calculate_nwp_power(station_metadata, df)
    
    # 5. PREPARE FEATURES
    print("\n" + "=" * 80)
    print("STEP 5: Preparing Features")
    print("=" * 80)
    
    df = prepare_features(df)
    
    # 6. CONFIGURE MODELS FOR DAY-AHEAD PREDICTION
    print("\n" + "=" * 80)
    print("STEP 6: Configuring Models")
    print("=" * 80)
    
    # Physics-Residual Mamba Configuration
    phys_configs = PhysicsResidualMambaConfigs(n_splits=2)  # Use 2 folds for CV
    phys_configs.seq_len = 672  # ~7 days lookback
    phys_configs.pred_len = prediction_horizon_hours * 4  # 48 hours = 192 steps at 15-min intervals
    phys_configs.batch_size = 64
    phys_configs.epochs = epochs_per_patch
    phys_configs.T_ref = 25.0  # Reference temperature
    phys_configs.G_night_thr = 10.0  # Night threshold
    
    # Base Mamba Configuration (for comparison)
    base_configs = PhysicsResidualMambaConfigs(n_splits=2)
    base_configs.seq_len = 672
    base_configs.pred_len = prediction_horizon_hours * 4
    base_configs.batch_size = 64
    base_configs.epochs = epochs_per_patch
    
    # Update feature columns for day-ahead prediction
    # Basic features
    basic_features = [
        'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure',
        'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
        'power', 'P_CLR', 'K_PV', 'NWP_Power_MW',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'season_sin', 'season_cos'
    ]
    
    # Add rolling features
    rolling_features = []
    for col in ['power', 'lmd_totalirrad', 'lmd_temperature', 'lmd_windspeed']:
        for h in [3, 5, 8]:
            if f'{col}_mean_{h}h' in df.columns:
                rolling_features.append(f'{col}_mean_{h}h')
            if f'{col}_std_{h}h' in df.columns:
                rolling_features.append(f'{col}_std_{h}h')
    
    all_features = basic_features + rolling_features
    
    # Update configurations using the new method to recalculate derived attributes
    # Add time features to FUTURE_INPUT_COLS for physics layer geometric gating
    future_features = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
                       'hour_sin', 'hour_cos', 'season_sin', 'season_cos']
    
    phys_configs.update_feature_columns(all_features, future_features)
    base_configs.update_feature_columns(all_features, future_features)
    
    print(f"Past features: {len(phys_configs.PAST_INPUT_COLS)}")
    print(f"Future features: {len(phys_configs.FUTURE_INPUT_COLS)}")
    print(f"Common features: {len(phys_configs.common_features)}")
    print(f"Total input dimension (enc_in): {phys_configs.enc_in}")
    print(f"Prediction horizon: {phys_configs.pred_len} steps ({prediction_horizon_hours} hours)")
    
    # 7. RUN 25-FOLD CROSS-VALIDATION
    print("\n" + "=" * 80)
    print("STEP 7: Running 25-Fold Cross-Validation")
    print("=" * 80)
    
    # Create 25-fold splits (simulate 25 patches)
    from sklearn.model_selection import KFold
    
    kfold = KFold(n_splits=n_patches, shuffle=True, random_state=random_seed)
    
    all_results = []
    all_predictions = []
    all_actuals = []
    
    fold_num = 1
    for train_idx, val_idx in kfold.split(df):
        print(f"\n--- Fold {fold_num}/{n_patches} ---")
        
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Prepare train data
        train_df = prepare_features(train_df)
        val_df = prepare_features(val_df)
        
        # Create dataloaders using prepare_rolling_folds
        # Note: We use n_splits=2 for each fold to get single train/val split
        # TimeSeriesSplit requires at least 2 splits
        train_loader, val_loader, _ = list(prepare_rolling_folds(
            train_df,
            phys_configs.PAST_INPUT_COLS,
            ['power'],  # TARGET_COL
            phys_configs.FUTURE_INPUT_COLS,
            n_splits=2,
            seq_len=phys_configs.seq_len,
            pred_len=phys_configs.pred_len,
            batch_size=phys_configs.batch_size
        ))[0]
        
        # Train Physics-Residual Mamba
        print(f"  Training Physics-Residual Mamba (Fold {fold_num})...")
        phys_model = PhysicsResidualMamba(phys_configs).to(device)
        criterion = PhysicsPVLoss(phys_configs, phys_configs.FUTURE_INPUT_COLS).to(device)
        
        for epoch in range(phys_configs.epochs):
            train_loss, train_logs = train_one_epoch_physics(
                phys_model, train_loader, 
                optim.Adam(phys_model.parameters(), lr=5e-4),
                criterion, device
            )
            
            if (epoch + 1) % 5 == 0:
                phys = phys_model.get_physics_params()
                msg = (
                    f"Epoch {epoch+1:03d} | "
                    f"Train={train_loss:.4f} | "
                    f"L_data={train_logs['L_data']:.4f} "
                    f"L_night={train_logs['L_night']:.4f} "
                    f"L_mono={train_logs['L_mono']:.4f} | "
                    f"η={phys['eta']:.4f} U0={phys['U0']:.2f}"
                )
                print(msg)
        
        # Evaluate on validation set
        print(f"  Evaluating Physics-Residual Mamba (Fold {fold_num})...")
        val_loss, val_rmse, val_mae, val_logs, preds, actuals, baseline, nwp = evaluate_fold_physics(
            phys_model, val_loader, criterion, device, phys_configs
        )
        
        # Calculate Smart Persistence Metrics
        sp_rmse = np.sqrt(np.nanmean((baseline - actuals.squeeze())**2))
        sp_mae = np.nanmean(np.abs(baseline - actuals.squeeze()))
        
        # Calculate NWP Metrics
        nwp_rmse = np.sqrt(np.nanmean((nwp - actuals.squeeze())**2))
        
        print(f"  Physics-Residual: RMSE={val_rmse:.4f}, MAE={val_mae:.4f}")
        print(f"  Smart Persistence: RMSE={sp_rmse:.4f}, MAE={sp_mae:.4f}")
        print(f"  NWP Forecast: RMSE={nwp_rmse:.4f}")
        
        # Store results
        all_results.append({
            'fold': fold_num,
            'phys_rmse': val_rmse,
            'phys_mae': val_mae,
            'sp_rmse': sp_rmse,
            'sp_mae': sp_mae,
            'nwp_rmse': nwp_rmse,
            'preds': preds,
            'actuals': actuals,
            'baseline': baseline,
            'nwp': nwp
        })
        
        all_predictions.append(preds)
        all_actuals.append(actuals)
        
        fold_num += 1
    
    # 8. GENERATE COMPREHENSIVE REPORTS
    print("\n" + "=" * 80)
    print("STEP 8: Generating Reports")
    print("=" * 80)
    
    # Calculate average metrics
    avg_phys_rmse = np.mean([r['phys_rmse'] for r in all_results])
    avg_phys_mae = np.mean([r['phys_mae'] for r in all_results])
    avg_sp_rmse = np.mean([r['sp_rmse'] for r in all_results])
    avg_nwp_rmse = np.mean([r['nwp_rmse'] for r in all_results])
    
    # Calculate improvements
    imp_vs_sp = (1 - avg_phys_rmse / avg_sp_rmse) * 100
    imp_vs_nwp = (1 - avg_phys_rmse / avg_nwp_rmse) * 100
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    print(f"\nPhysics-Residual Mamba (25-Fold CV):")
    print(f"  Average RMSE: {avg_phys_rmse:.4f} MW")
    print(f"  Average MAE: {avg_phys_mae:.4f} MW")
    print(f"  Improvement vs Smart Persistence: {imp_vs_sp:.1f}%")
    print(f"  Improvement vs NWP: {imp_vs_nwp:.1f}%")
    
    print(f"\nBaselines:")
    print(f"  Smart Persistence RMSE: {avg_sp_rmse:.4f} MW")
    print(f"  NWP Forecast RMSE: {avg_nwp_rmse:.4f} MW")
    
    # 9. VISUALIZE RESULTS
    print("\n" + "=" * 80)
    print("STEP 9: Visualizing Results")
    print("=" * 80)
    
    # Use last fold for detailed visualization
    last_result = {
        'preds': all_results[-1]['preds'],
        'actuals': all_results[-1]['actuals'],
        'baseline': all_results[-1]['baseline'],
        'nwp': all_results[-1]['nwp'],
        'rmse': avg_phys_rmse
    }
    
    # Plot 1: Model Performance (3 plots)
    plot_model_performance(last_result)
    
    # Plot 2: Benchmark Summary
    base_results_for_plot = [
        {'rmse': r['sp_rmse'], 'mae': r['sp_mae'], 'model': 'Smart Persistence'}
        for r in all_results
    ]
    phys_results_for_plot = [
        {'rmse': r['phys_rmse'], 'mae': r['phys_mae'], 'model': 'Physics-Residual'}
        for r in all_results
    ]
    plot_benchmark_summary(base_results_for_plot, phys_results_for_plot)
    
    # Plot 3: Multi-Model Comparison (first 500 steps)
    predictions_dict = {
        'Ground Truth': all_results[-1]['actuals'][:500].flatten(),
        'Physics-Residual': all_results[-1]['preds'][:500].flatten(),
        'Smart Persistence': all_results[-1]['baseline'][:500].flatten(),
        'NWP Forecast': all_results[-1]['nwp'][:500].flatten(),
    }
    plot_multi_model_comparison(all_results[-1]['actuals'], predictions_dict, limit=500)
    
    # 10. SAVE RESULTS
    print("\n" + "=" * 80)
    print("STEP 10: Saving Results")
    print("=" * 80)
    
    import json
    
    results_summary = {
        'experiment_config': {
            'target_station': target_station,
            'n_patches': n_patches,
            'prediction_horizon_hours': prediction_horizon_hours,
            'epochs_per_patch': epochs_per_patch,
            'random_seed': random_seed,
            'use_spatial': use_spatial,
        },
        'metrics': {
            'avg_phys_rmse': float(avg_phys_rmse),
            'avg_phys_mae': float(avg_phys_mae),
            'avg_sp_rmse': float(avg_sp_rmse),
            'avg_nwp_rmse': float(avg_nwp_rmse),
            'imp_vs_sp': float(imp_vs_sp),
            'imp_vs_nwp': float(imp_vs_nwp),
        },
        'fold_results': all_results,
    }
    
    results_file = os.path.join(output_dir, f"results_station{target_station}_25fold.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Save predictions for each fold
    for i, result in enumerate(all_results):
        fold_file = os.path.join(output_dir, f"fold_{i+1}_predictions.npz")
        np.savez(fold_file, {
            'preds': result['preds'],
            'actuals': result['actuals'],
            'baseline': result['baseline'],
            'nwp': result['nwp'],
        })
    
    print(f"Predictions saved for {len(all_results)} folds")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    return results_summary


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run 25-fold cross-validation with day-ahead prediction
    results = run_25fold_dayahead_experiment(
        target_station=7,
        n_patches=25,
        prediction_horizon_hours=48,
        epochs_per_patch=5,
        random_seed=42,
        use_spatial=False,
        output_dir='./results_25fold_dayahead'
    )
    
    print("\n✓ Experiment completed successfully!")
    print("\nTo run with different settings:")
    print("  results = run_25fold_dayahead_experiment(")
    print("      target_station=7,")
    print("      n_patches=25,")
    print("      prediction_horizon_hours=48,")
    print("      epochs_per_patch=5,")
    print("      random_seed=42,")
    print("      use_spatial=False,")
    print("      output_dir='./results_25fold_dayahead',")
    print("  )")
