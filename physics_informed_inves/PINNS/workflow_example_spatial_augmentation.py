"""
Unified Workflow for Physics-Residual Power Mamba with Spatial Augmentation

This script demonstrates:
1. Loading station-specific metadata from metadata.csv
2. Calculating clear sky indices (GHI_clr, K_CS, P_CLR, K_PV)
3. Calculating NWP power from NWP weather forecasts
4. Preparing features (cyclic encodings, rolling statistics)
5. Adding spatial information from adjacent stations
6. Training and evaluating models
7. Visualizing results with publication-quality plots

The workflow shows how to use the refactored modular package for
end-to-end solar power forecasting with spatial augmentation.
"""

import os
import sys
import pandas as pd
import numpy as np
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
    calculate_errors_by_time,
    
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

# =============================================================================
# PART 1: Station Metadata Management
# =============================================================================

def get_station_metadata(station_num: int, df: pd.DataFrame) -> Dict[str, any]:
    """
    Retrieves and parses metadata for a specific station from the DataFrame.
    
    Args:
        station_num: The station number (e.g., 7 for station07)
        df: DataFrame containing station metadata
        
    Returns:
        Dictionary with clean numerical values ready for simulation
        
    Example:
        >>> metadata = get_station_metadata(7, metadata_df)
        >>> print(metadata['Capacity'])  # 20.0
    """
    # Format station ID
    station_id_str = f"station{str(station_num).zfill(2)}"
    
    # Filter dataframe
    station_row = df[df['Station_ID'] == station_id_str]
    
    if station_row.empty:
        raise ValueError(f"Station ID {station_id_str} not found in DataFrame.")
    
    # Get single row as Series
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
        
    Example:
        >>> df = calculate_clearsky_indices(metadata, df)
        >>> print(df[['GHI_clr', 'K_CS', 'P_CLR', 'K_PV']].head())
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
    
    # Extract Tilt safely from string
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
    
    Args:
        metadata: Station metadata from get_station_metadata()
        df: DataFrame containing 'nwp_' columns
        
    Returns:
        Series: The calculated power in MW, aligned with df.index
        
    Example:
        >>> nwp_power = calculate_nwp_power(metadata, df)
        >>> print(f"Mean NWP Power: {nwp_power.mean():.4f} MW")
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
# PART 2: Spatial Augmentation (Multi-Station Support)
# =============================================================================

def load_multiple_stations(
    station_nums: List[int],
    metadata_df: pd.DataFrame,
    data_dir: str
) -> Dict[int, pd.DataFrame]:
    """
    Load data for multiple stations.
    
    Args:
        station_nums: List of station numbers (e.g., [5, 6, 7, 8])
        metadata_df: DataFrame with station metadata
        data_dir: Directory containing station CSV files
        
    Returns:
        Dictionary mapping station number to DataFrame
        
    Example:
        >>> stations = load_multiple_stations([5, 6, 7], metadata_df, '/path/to/data')
        >>> print(f"Loaded {len(stations)} stations")
    """
    stations_data = {}
    
    for station_num in station_nums:
        station_path = os.path.join(data_dir, f"station{str(station_num).zfill(2)}.csv")
        
        if os.path.exists(station_path):
            stations_data[station_num] = pd.read_csv(station_path)
            print(f"✓ Loaded station {station_num}")
        else:
            print(f"✗ Station {station_num} not found at {station_path}")
    
    return stations_data


def add_spatial_features(
    df: pd.DataFrame,
    stations_data: Dict[int, pd.DataFrame],
    target_station: int,
    adjacent_stations: List[int],
    feature_cols: List[str] = ['power', 'lmd_totalirrad']
) -> pd.DataFrame:
    """
    Add spatial features from adjacent stations to improve predictions.
    
    This implements a simple spatial augmentation approach:
    - Calculate rolling statistics from adjacent stations
    - Add as additional features to target station
    
    Args:
        df: Target station DataFrame
        stations_data: Dictionary of all station DataFrames
        target_station: Target station number
        adjacent_stations: List of adjacent station numbers
        feature_cols: Features to aggregate from adjacent stations
        
    Returns:
        DataFrame with added spatial features
        
    Example:
        >>> df_augmented = add_spatial_features(df, stations_data, 7, [6, 8])
    """
    print(f"Adding spatial features from stations: {adjacent_stations}")
    
    df_aug = df.copy()
    
    for adj_station in adjacent_stations:
        if adj_station in stations_data:
            adj_df = stations_data[adj_station]
            
            # Align indices (use intersection)
            common_idx = df.index.intersection(adj_df.index)
            
            if len(common_idx) > 0:
                # Calculate rolling statistics from adjacent station
                for col in feature_cols:
                    if col in adj_df.columns:
                        # 3-hour rolling mean
                        adj_rolling_3h = adj_df[col].rolling('3H', min_periods=1).mean()
                        df_aug[f'station{adj_station}_{col}_mean_3h'] = adj_rolling_3h.reindex(df.index)
                        
                        # 3-hour rolling std
                        adj_rolling_3h_std = adj_df[col].rolling('3H', min_periods=1).std()
                        df_aug[f'station{adj_station}_{col}_std_3h'] = adj_rolling_3h_std.reindex(df.index)
                        
                        # Current value (for real-time spatial correlation)
                        df_aug[f'station{adj_station}_{col}_current'] = adj_df[col].reindex(df.index)
            else:
                print(f"  Station {adj_station} not available in stations_data")
    
    # Count spatial features added
    spatial_cols = [col for col in df_aug.columns if 'station' in col]
    print(f"Added {len(spatial_cols)} spatial features")
    
    return df_augmented


def get_adjacent_stations(target_station: int, total_stations: int = 10, num_adjacent: int = 2) -> List[int]:
    """
    Get adjacent station numbers based on spatial proximity.
    
    This is a simplified approach - in production, you would use
    actual geographic coordinates to find nearest neighbors.
    
    Args:
        target_station: Target station number
        total_stations: Total number of stations available
        num_adjacent: Number of adjacent stations to include
        
    Returns:
        List of adjacent station numbers
        
    Example:
        >>> adjacent = get_adjacent_stations(7, total_stations=10, num_adjacent=2)
        >>> print(adjacent)  # [6, 8]
    """
    # Simple approach: use station numbers before and after
    adjacent = []
    
    for offset in range(-num_adjacent, num_adjacent + 1):
        adj_station = target_station + offset
        
        if 1 <= adj_station <= total_stations:
            adjacent.append(adj_station)
    
    return adjacent


# =============================================================================
# PART 3: Feature Engineering
# =============================================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training.
    
    Adds:
    - Cyclic time encodings (hour, day, month, season)
    - Rolling statistics (3h, 5h, 8h windows)
    - Ensures DatetimeIndex
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added features
        
    Example:
        >>> df_prepared = prepare_features(df)
        >>> print(df_prepared.columns)
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
                data[f'{col}_mean_{h}h'] = data[col].shift(1).rolling(f'{h}H', min_periods=1).mean()
                data[f'{col}_std_{h}h'] = data[col].shift(1).rolling(f'{h}H', min_periods=1).std()
    
    # Fill NaN values with 0.0
    data = data.fillna(0.0)
    
    print(f"Prepared features: {len(data.columns)} columns")
    
    return data


# =============================================================================
# PART 4: Complete Workflow Example
# =============================================================================

def run_complete_workflow(
    target_station: int = 7,
    use_spatial: bool = True,
    n_splits: int = 2,
    random_seed: int = 42,
    epochs: int = 20
):
    """
    Run complete workflow from data loading to model evaluation.
    
    Args:
        target_station: Target station number
        use_spatial: Whether to add spatial features from adjacent stations
        n_splits: Number of cross-validation folds
        random_seed: Random seed for reproducibility
        epochs: Number of training epochs
        
    Example:
        >>> results = run_complete_workflow(
        ...     target_station=7,
        ...     use_spatial=True,
        ...     n_splits=2
        ... )
    """
    print("=" * 80)
    print("PHYSICS-RESIDUAL POWER MAMBA - COMPLETE WORKFLOW")
    print("=" * 80)
    
    # Set reproducibility
    set_random_seed(random_seed)
    device = get_device()
    print(f"\nDevice: {device}")
    print(f"Random Seed: {random_seed}")
    
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
    
    # 6. ADD SPATIAL FEATURES (Optional)
    if use_spatial:
        print("\n" + "=" * 80)
        print("STEP 6: Adding Spatial Features")
        print("=" * 80)
        
        # Load adjacent stations
        adjacent_stations = get_adjacent_stations(target_station, total_stations=10, num_adjacent=2)
        stations_data = load_multiple_stations([target_station] + adjacent_stations, metadata_df, abs_path)
        
        # Add spatial features
        df = add_spatial_features(df, stations_data, target_station, adjacent_stations)
        
        print(f"Spatial features added from {len(adjacent_stations)} adjacent stations")
    
    # 7. CONFIGURE MODELS
    print("\n" + "=" * 80)
    print("STEP 7: Configuring Models")
    print("=" * 80)
    
    # Physics-Residual Mamba Configuration
    phys_configs = PhysicsResidualMambaConfigs(n_splits=n_splits)
    phys_configs.seq_len = 672  # ~7 days
    phys_configs.pred_len = 96   # 24 hours
    phys_configs.batch_size = 64
    phys_configs.epochs = epochs
    
    # Update feature columns based on what's available
    # Basic features
    basic_features = [
        'lmd_totalirrad', 'lmd_diffuseirrad', 'lmd_temperature', 'lmd_pressure',
        'nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed',
        'power', 'P_CLR', 'K_PV', 'NWP_Power_MW',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'month_sin', 'month_cos', 'season_sin', 'season_cos'
    ]
    
    # Add rolling features if they exist
    rolling_features = []
    for col in ['power', 'lmd_totalirrad', 'lmd_temperature', 'lmd_windspeed']:
        for h in [3, 5, 8]:
            if f'{col}_mean_{h}h' in df.columns:
                rolling_features.append(f'{col}_mean_{h}h')
            if f'{col}_std_{h}h' in df.columns:
                rolling_features.append(f'{col}_std_{h}h')
    
    # Add spatial features if used
    if use_spatial:
        spatial_cols = [col for col in df.columns if 'station' in col]
        basic_features.extend(spatial_cols)
    
    # Update configuration
    phys_configs.PAST_INPUT_COLS = basic_features + rolling_features
    phys_configs.FUTURE_INPUT_COLS = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed', 'NWP_Power_MW']
    
    print(f"Past features: {len(phys_configs.PAST_INPUT_COLS)}")
    print(f"Future features: {len(phys_configs.FUTURE_INPUT_COLS)}")
    
    # 8. TRAIN AND EVALUATE
    print("\n" + "=" * 80)
    print("STEP 8: Training and Evaluating Models")
    print("=" * 80)
    
    # Run physics-informed model
    print("\n--- Training Physics-Residual Mamba ---")
    phys_results, phys_history = run_physics_residual_cv(
        df, phys_configs, use_scaler=True, n_splits=n_splits, random_seed=random_seed
    )
    
    # Run base Mamba for comparison
    print("\n--- Training Base Mamba ---")
    base_configs = PhysicsResidualMambaConfigs(n_splits=n_splits)
    base_configs.seq_len = 672
    base_configs.pred_len = 96
    base_configs.batch_size = 64
    base_configs.epochs = epochs
    base_configs.PAST_INPUT_COLS = phys_configs.PAST_INPUT_COLS
    base_configs.FUTURE_INPUT_COLS = ['nwp_globalirrad', 'nwp_temperature', 'nwp_windspeed']
    
    base_results = run_base_mamba_cv(
        df, base_configs, n_splits=n_splits, random_seed=random_seed
    )
    
    # 9. VISUALIZE RESULTS
    print("\n" + "=" * 80)
    print("STEP 9: Visualizing Results")
    print("=" * 80)
    
    # Plot 1: Model Performance (last fold)
    print("\n--- Plotting Physics-Residual Model Performance ---")
    plot_model_performance(phys_results[-1])
    
    # Plot 2: Benchmark Summary
    print("\n--- Plotting Benchmark Summary ---")
    plot_benchmark_summary(base_results, phys_results)
    
    # Plot 3: Multi-Model Comparison
    print("\n--- Plotting Multi-Model Comparison ---")
    predictions_dict = {
        'Ground Truth': phys_results[-1]['actuals'],
        'Physics-Residual': phys_results[-1]['preds'],
        'Base Mamba': base_results[-1]['preds'],
        'Smart Persistence': phys_results[-1]['baseline'],
        'NWP Forecast': phys_results[-1]['nwp']
    }
    plot_multi_model_comparison(phys_results[-1]['actuals'], predictions_dict, limit=500)
    
    # 10. PRINT SUMMARY
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    # Calculate average metrics
    avg_phys_rmse = np.mean([r['rmse'] for r in phys_results])
    avg_base_rmse = np.mean([r['rmse'] for r in base_results])
    avg_sp_rmse = np.mean([r['sp_rmse'] for r in phys_results])
    avg_nwp_rmse = np.mean([r['nwp_rmse'] for r in phys_results])
    
    # Calculate improvements
    imp_vs_base = (1 - avg_phys_rmse / avg_base_rmse) * 100
    imp_vs_sp = (1 - avg_phys_rmse / avg_sp_rmse) * 100
    imp_vs_nwp = (1 - avg_phys_rmse / avg_nwp_rmse) * 100
    
    print(f"\nPhysics-Residual Mamba:")
    print(f"  RMSE: {avg_phys_rmse:.4f} MW")
    print(f"  Improvement vs Base Mamba: {imp_vs_base:.1f}%")
    print(f"  Improvement vs Smart Persistence: {imp_vs_sp:.1f}%")
    print(f"  Improvement vs NWP: {imp_vs_nwp:.1f}%")
    
    print(f"\nBase Mamba:")
    print(f"  RMSE: {avg_base_rmse:.4f} MW")
    
    print(f"\nBaselines:")
    print(f"  Smart Persistence RMSE: {avg_sp_rmse:.4f} MW")
    print(f"  NWP Forecast RMSE: {avg_nwp_rmse:.4f} MW")
    
    # Print learned physics parameters
    print(f"\nLearned Physics Parameters (Last Fold):")
    final_params = phys_results[-1]['physics_params']
    for k, v in final_params.items():
        print(f"  {k}: {v:.6f}")
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    
    return {
        'phys_results': phys_results,
        'base_results': base_results,
        'configs': {
            'phys': phys_configs.to_dict(),
            'base': base_configs.to_dict()
        }
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run complete workflow
    results = run_complete_workflow(
        target_station=7,
        use_spatial=True,
        n_splits=2,
        random_seed=42,
        epochs=20
    )
    
    print("\n✓ Workflow completed successfully!")
    print("\nTo run with different settings:")
    print("  results = run_complete_workflow(")
    print("      target_station=7,")
    print("      use_spatial=True,")
    print("      n_splits=4,")
    print("      random_seed=42,")
    print("      epochs=30")
    print("  )")
