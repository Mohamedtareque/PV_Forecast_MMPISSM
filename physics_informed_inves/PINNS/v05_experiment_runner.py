"""
V05 Experiment Runner - Spatial vs Single Station Comparison
============================================================

This script automates the full A/B test for V05 Physics-Residual Mamba models:
1. Data Preparation (Station 07 + Station 08 + Spatial Deltas).
2. Running Spatial Benchmark (St07 + St08 + Deltas).
3. Running Single Station Benchmark (St07 Only).
4. Generating Comparison Metrics and Graphs.

Usage:
    python v05_experiment_runner.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import re
import matplotlib.pyplot as plt
from typing import Dict, List
from pvlib import location, pvsystem, modelchain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

# Import Benchmarks
from physics_residual_mamba_spatial_v05 import run_comparative_benchmark
from physics_residual_mamba_SINGLE_v05 import run_single_station_benchmark

# =============================================================================
# DATA PREPARATION HELPERS (From Workflow)
# =============================================================================

# Delegated to station_meta_data_manipulation.py
def get_station_metadata(station_num: int, df: pd.DataFrame) -> Dict[str, any]:
    station_id_str = f"station{str(station_num).zfill(2)}"
    station_row = df[df['Station_ID'] == station_id_str]
    if station_row.empty: raise ValueError(f"Station ID {station_id_str} not found in DataFrame.")
    row = station_row.iloc[0]
    
    def parse_kv_string(text_block):
        data = {}
        if isinstance(text_block, str):
            for item in text_block.split('\\n'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    data[k.strip()] = v.strip()
        return data
    
    def extract_num(val, default=0.0):
        if pd.isna(val): return default
        match = re.search(r"[-+]?\\d*\\.\\d+|\\d+", str(val))
        return float(match.group()) if match else default
    
    module_items = parse_kv_string(row.get('Module', ''))
    inverter_items = parse_kv_string(row.get('Inverters', ''))
    layout_items = parse_kv_string(row.get('Layout', ''))
    
    return {
        'Station_ID': station_id_str,
        'Longitude': float(row['Longitude']),
        'Latitude': float(row['Latitude']),
        'Array_Tilt': row['Array_Tilt'],
        'Capacity': float(row['Capacity']),
        'Modules_per_String': int(extract_num(layout_items.get('modules per string'), default=20)),
        'Strings_per_Inverter': int(extract_num(layout_items.get('strings per inverter'), default=100)),
        'Module_Pmax': extract_num(module_items.get('Pmax'), default=250),
        'Inverter_Rated_Power': extract_num(inverter_items.get('Rated power'), default=500),
    }


# calculate clearsky indices based on site 
def calculate_clearsky_indices(latitude:float, longitude:float, email:str, df: pd.DataFrame) -> pd.DataFrame:
    site = location.Location(latitude, longitude, tz='UTC')
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'],  utc=True) 
            df.set_index('date_time', inplace=True)
            df = df.sort_index()
    
        # Try McClear first (more accurate satellite-based model)
    try:
        from pvlib.iotools import get_cams
        cams_data, cams_meta = get_cams(
            latitude=latitude,
            longitude=longitude,
            start=df.index.min(),
            end=df.index.max(),
            email=email,
            identifier='mcclear',
            time_step='15min',
            time_ref='UT',
            integrated=False,
            map_variables=True
        )
        
        cs = cams_data.rename(columns={
            'ghi_clear': 'ghi',
            'dni_clear': 'dni',
            'dhi_clear': 'dhi',
        })
        # CRITICAL: Reindex CAMS data to match original df index
        cs = cs.reindex(df.index, method='nearest')
        print("✓ Using McClear (satellite-based, more accurate)")
        
    except Exception as e:
        print(f"⚠ McClear failed ({e}), falling back to Ineichen")
        cs = site.get_clearsky(df.index, model='ineichen')

    cs = cs.reindex(df.index, method='nearest')
    df['GHI_clr'] = cs['ghi'].values  # Use .values to avoid index issues
    df['dni_clr'] = cs['dni'].values  # Use .values to avoid index issues
    df['dhi_clr'] = cs['dhi'].values  # Use .values to avoid index issues

    return df


    
    

# def calculate_clearsky_indices(metadata: Dict, df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculates both Irradiance (K_CS) and Power (K_PV) Clear Sky Indices.
#     Uses McClear (satellite-based) with Ineichen fallback.
#     """
#     print(f"--- Processing {metadata.get('Station_ID', 'Unknown')} ---")
    
#     lat = metadata['Latitude']
#     lon = metadata['Longitude']
#     site = location.Location(lat, lon, tz='UTC')
    
#     if not isinstance(df.index, pd.DatetimeIndex):
#         if 'date_time' in df.columns:
#             df['date_time'] = pd.to_datetime(df['date_time']) 
#             df.set_index('date_time', inplace=True)
        
#     print("Calculating Clear Sky Irradiance...")
    
#     # Try McClear first (more accurate satellite-based model)
#     try:
#         from pvlib.iotools import get_cams
#         email = 'mohamedtareck95@gmail.com'
        
#         cams_data, cams_meta = get_cams(
#             latitude=metadata['Latitude'],
#             longitude=metadata['Longitude'],
#             start=df.index.min(),
#             end=df.index.max(),
#             email=email,
#             identifier='mcclear',
#             time_step='15min',
#             time_ref='UT',
#             integrated=False,
#             map_variables=True
#         )
        
#         cs = cams_data.rename(columns={
#             'ghi_clear': 'ghi',
#             'dni_clear': 'dni',
#             'dhi_clear': 'dhi',
#         })
#         # CRITICAL: Reindex CAMS data to match original df index
#         cs = cs.reindex(df.index, method='nearest')
#         print("✓ Using McClear (satellite-based, more accurate)")
        
#     except Exception as e:
#         print(f"⚠ McClear failed ({e}), falling back to Ineichen")
#         cs = site.get_clearsky(df.index, model='ineichen')

#     df['GHI_clr'] = cs['ghi'].values  # Use .values to avoid index issues

#     if 'lmd_totalirrad' in df.columns:
#         meas_ghi = df['lmd_totalirrad']
#         df['K_CS_day'] = np.where(df['GHI_clr'] > 10, meas_ghi / df['GHI_clr'], 0.0)
#         df['K_CS_dayNight'] = meas_ghi / df['GHI_clr']
#         df['K_CS'] = df['K_CS_day']
#         df['K_CS'] = df['K_CS'].clip(lower=0.0, upper=1.25)
#     else:
#         print("Warning: 'lmd_totalirrad' column missing. K_CS not calculated.")

#     # PV System Model
#     try:
#         tilt_str = str(metadata.get('Array_Tilt'))
#         tilt_match = re.search(r"[\d.]+", tilt_str)
#         tilt = float(tilt_match.group()) if tilt_match else 31.0
#     except:
#         tilt = 31.0
    
#     module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
#     block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    
#     # BYD 500kW Inverter Parameters
#     inverter_parameters = {
#         'Vac': 480,
#         'Pso': 4549.66,
#         'Paco': 501756,
#         'Pdco': 521992,
#         'Vdco': 890,
#         'C0': -3.45645e-08,
#         'C1': 2.12268e-05,
#         'C2': 0.000603471,
#         'C3': 0.000463011,
#         'Pnt': 2.0,
#         'Vdcmax': 1000,
#         'Idcmax': 586.508,
#         'Mppt_low': 780,
#         'Mppt_high': 1000
#     }
    
#     temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
#     system = pvsystem.PVSystem(
#         surface_tilt=tilt,
#         surface_azimuth=180,
#         module_parameters=module_params,
#         inverter_parameters=inverter_parameters,
#         temperature_model_parameters=temp_model,
#         modules_per_string=metadata['Modules_per_String'],
#         strings_per_inverter=metadata['Strings_per_Inverter']
#     )

#     print("Running PV Power Simulation...")
#     mc = modelchain.ModelChain(system, site, transposition_model='perez',
#                                solar_position_method='nrel_numpy',
#                                aoi_model='physical', spectral_model='no_loss')
#     mc.run_model(cs)

#     station_total_capacity_watts = metadata['Capacity'] * 1000
#     scaling_factor = station_total_capacity_watts / block_dc_watts
    
#     p_clr_watts = mc.results.ac.fillna(0) * scaling_factor
#     df['P_CLR'] = p_clr_watts / 1_000_000 

#     if 'power' in df.columns:
#         meas_power = df['power']
#         if meas_power.max() > 500:
#             meas_power = meas_power / 1000.0
            
#         df['K_PV'] = np.where(df['P_CLR'] > 1.0, meas_power / df['P_CLR'], 0.0)
#         df['K_PV'] = df['K_PV'].clip(lower=0.0, upper=1.25)
#     else:
#         print("Warning: 'power' column missing. K_PV not calculated.")

#     return df


def calculate_nwp_power(metadata: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates expected PV Power output (MW) based on NWP Weather Forecasts.
    Uses BYD 500kW parameters.
    """
    station_id = metadata.get('Station_ID', 'Unknown')
    print(f"--- Calculating NWP Physical Power for {station_id} ---")
    
    lat = float(metadata['Latitude'])
    lon = float(metadata['Longitude'])
    site = location.Location(lat, lon, tz='UTC')
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time']) 
            df.set_index('date_time', inplace=True)
    
    # NOTE: Removed timezone localization to avoid merge issues with Station 08

    # Sun Elevation & Weather Prep
    solpos = site.get_solarposition(df.index)
    df['sun_elevation'] = solpos['elevation']

    weather_nwp = pd.DataFrame(index=df.index)
    weather_nwp['ghi'] = df['nwp_globalirrad']
    weather_nwp['dni'] = df['nwp_directirrad']
    weather_nwp['temp_air'] = df['nwp_temperature']
    weather_nwp['wind_speed'] = df['nwp_windspeed']
    weather_nwp['pressure'] = df['nwp_pressure'] * 100 if 'nwp_pressure' in df.columns else 101325.0

    zenith_rad = np.radians(solpos['zenith'])
    weather_nwp['dhi'] = (weather_nwp['ghi'] - (weather_nwp['dni'] * np.cos(zenith_rad))).clip(lower=0)

    # Define PV System
    tilt_match = re.search(r"(\d+(\.\d+)?)", str(metadata.get('Array_Tilt', '31')))
    tilt = float(tilt_match.group(1)) if tilt_match else 31.0
        
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    
    # BYD 500kW Inverter Parameters
    inv_para = {
        'Vac': 480,
        'Pso': 4549.66,
        'Paco': 501756,
        'Pdco': 521992,
        'Vdco': 890,
        'C0': -3.45645e-08,
        'C1': 2.12268e-05,
        'C2': 0.000603471,
        'C3': 0.000463011,
        'Pnt': 2.0,
        'Vdcmax': 1000,
        'Idcmax': float(metadata.get('Inverter_Idcmax', 586.5)),
        'Mppt_low': 780,
        'Mppt_high': 1000
    }

    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=180, 
        module_parameters=module_params,
        inverter_parameters=inv_para,
        temperature_model_parameters=temp_model,
        modules_per_string=int(metadata['Modules_per_String']),
        strings_per_inverter=int(metadata['Strings_per_Inverter'])
    )

    # Run Simulation
    mc = modelchain.ModelChain(system, site, transposition_model='perez',
                               aoi_model='physical', spectral_model='no_loss')
    mc.run_model(weather_nwp)
    
    # Scale to Station Capacity
    block_dc_watts = (int(metadata['Modules_per_String']) * int(metadata['Strings_per_Inverter']) * float(metadata['Module_Pmax']))
    station_capacity_watts = float(metadata['Capacity']) * 1000 
    theoretical_scaling = station_capacity_watts / block_dc_watts if block_dc_watts > 0 else 1.0
        
    # Convert to MW
    df['NWP_Power_MW'] = (mc.results.ac.fillna(0) * theoretical_scaling / 1_000_000).clip(lower=0.0)
    
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date_time' in data.columns:
            data['date_time'] = pd.to_datetime(data['date_time'])
            data.set_index('date_time', inplace=True)
            
    data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
    data['day_sin'] = np.sin(2 * np.pi * data.index.dayofyear / 365)
    data['day_cos'] = np.cos(2 * np.pi * data.index.dayofyear / 365)
    data['month_sin'] = np.sin(2 * np.pi * data.index.month / 12)
    data['month_cos'] = np.cos(2 * np.pi * data.index.month / 12)
    data['season_sin'] = np.sin(2 * np.pi * data.index.month / 4)
    data['season_cos'] = np.cos(2 * np.pi * data.index.month / 4)
    
    # Calculate Sun Elevation
    # (Simplified approximation for feature prep, usually should come from site object but we need lat/lon)
    # Here we skip it if not passed or assume it's added later. 
    # Actually, let's assume valid lat/lon for St07 (34.0, 100.0 approx) if needed, 
    # but `calculate_nwp_power` etc usually handle solar pos internally. 
    # For Feature Prep, we just create the cyclics.
    return data


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# NOTE: SpatialMambaConfigs is now imported from physics_residual_mamba_spatial_v05
# to ensure consistent parameter definitions and geometry methods.
from physics_residual_mamba_spatial_v05 import SpatialMambaConfigs

# =============================================================================
# LOSS CURVES PLOTTING
# =============================================================================

def plot_loss_curves(results: Dict, benchmark_name: str):
    """
    Plots loss curves for each model/fold from benchmark results.
    Expects results dict with 'loss_history' key containing per-fold losses.
    """
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), squeeze=False)
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        ax = axes[0, idx]
        
        if 'loss_history' in model_results and model_results['loss_history']:
            loss_history = model_results['loss_history']
            
            # If loss_history is a list of per-fold losses
            if isinstance(loss_history, list):
                for fold_idx, fold_losses in enumerate(loss_history):
                    if isinstance(fold_losses, list):
                        epochs = range(1, len(fold_losses) + 1)
                        ax.plot(epochs, fold_losses, label=f'Fold {fold_idx + 1}', alpha=0.7)
            # If loss_history is a single list (aggregated)
            elif isinstance(loss_history, dict):
                for fold_name, fold_losses in loss_history.items():
                    epochs = range(1, len(fold_losses) + 1)
                    ax.plot(epochs, fold_losses, label=fold_name, alpha=0.7)
            else:
                epochs = range(1, len(loss_history) + 1)
                ax.plot(epochs, loss_history, label='Training Loss')
        
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Loss Curves - {benchmark_name}', fontsize=12)
    plt.tight_layout()
    
    save_path = f'results/{benchmark_name.lower().replace(" ", "_")}_loss_curves.png'
    plt.savefig(save_path, dpi=150)
    print(f"Loss curves saved to {save_path}")
    plt.close()

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main():
    abs_path = os.path.abspath("/home/muhammadhassan/App_v02/physics_informed_inves/PVODdatasets_v1")
    metadata_path = os.path.join(abs_path, "metadata.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.csv not found at {metadata_path}")
        return

    print("Loading Metadata...")
    metaData = pd.read_csv(metadata_path)

    # 1. Load Station 07
    station07_path = os.path.join(abs_path, "station07.csv")
    print(f"Loading Station 07 from {station07_path}...")
    station07_data = pd.read_csv(station07_path)
    if 'date_time' in station07_data.columns:
        station07_data['date_time'] = pd.to_datetime(station07_data['date_time'])
        station07_data.set_index('date_time', inplace=True)
        
    s07_meta = get_station_metadata(7, metaData)
    print("Pre-processing Station 07...")
    station07_df = calculate_clearsky_indices(s07_meta, station07_data)
    station07_df = calculate_nwp_power(s07_meta, station07_df)  # Now returns df with NWP_Power_MW
    station07_df = prepare_features(station07_df)

    # Add Sun Elevation (needed for V05 features)
    print("Adding Sun Elevation...")
    lat, lon = s07_meta['Latitude'], s07_meta['Longitude']
    site = location.Location(lat, lon, tz='UTC')
    solpos = site.get_solarposition(station07_df.index)
    station07_df['sun_elevation'] = solpos['elevation']

    # 2. Load Station 08
    station08_path = os.path.join(abs_path, "station08.csv")
    print(f"Loading Station 08 from {station08_path}...")
    station08_data = pd.read_csv(station08_path)
    if 'date_time' in station08_data.columns:
        station08_data['date_time'] = pd.to_datetime(station08_data['date_time'])
        station08_data.set_index('date_time', inplace=True)

    s08_meta = get_station_metadata(8, metaData)
    print("Pre-processing Station 08...")
    station08_df = calculate_clearsky_indices(s08_meta, station08_data)
    station08_df = calculate_nwp_power(s08_meta, station08_df)  # Now returns df with NWP_Power_MW
    station08_df = prepare_features(station08_df)

    # 3. Rename & Merge
    print("Merging Spatial Data...")
    s08_rename_cols = {
        'nwp_globalirrad': 's08_nwp_globalirrad',
        'nwp_temperature': 's08_nwp_temperature',
        'nwp_windspeed': 's08_nwp_windspeed',
        'K_PV': 's08_K_PV',
    }
    station08_spatial = station08_df[list(s08_rename_cols.keys())].rename(columns=s08_rename_cols)
    merged_df = station07_df.join(station08_spatial, how='inner')

    # 4. Deltas
    merged_df['spatial_delta_irrad'] = merged_df['nwp_globalirrad'] - merged_df['s08_nwp_globalirrad']
    merged_df['spatial_delta_temp'] = merged_df['nwp_temperature'] - merged_df['s08_nwp_temperature']
    merged_df['spatial_delta_wind'] = merged_df['nwp_windspeed'] - merged_df['s08_nwp_windspeed']
    
    merged_df = merged_df.fillna(0)
    print(f"Final Merged Shape: {merged_df.shape}")

    # 5. Run Benchmarks
    configs = SpatialMambaConfigs(n_splits=4)
    configs.set_station_coordinates(s07_meta, {**s08_meta, 'Longitude': 113.69999}); configs.set_station_coordinates = lambda *args: None
    configs.epochs = 30 
    configs.warmup_epochs = 15 
    
    print("\n" + "="*50)
    print("RUNNING SPATIAL BENCHMARK")
    print("="*50)
    res_spatial = run_comparative_benchmark(merged_df, n_splits=4, custom_config=configs)
    
    # Extract and plot loss curves if available
    if isinstance(res_spatial, dict) and 'loss_history' in res_spatial.get(list(res_spatial.keys())[0], {}):
        plot_loss_curves(res_spatial, 'Spatial Benchmark')

    print("\n" + "="*50)
    print("RUNNING SINGLE STATION BENCHMARK")
    print("="*50)
    res_single = run_single_station_benchmark(merged_df, n_splits=4, custom_config=configs)
    
    # Extract and plot loss curves if available
    if isinstance(res_single, dict) and 'loss_history' in res_single.get(list(res_single.keys())[0], {}):
        plot_loss_curves(res_single, 'Single Station Benchmark')

    # 6. Compare & Report
    if 'res_spatial' in locals() and 'res_single' in locals():
        print("\n" + "="*50)
        print("FINAL COMPARISON: SPATIAL GAIN")
        print("="*50)
        
        summary_data = []
        models = res_spatial.keys()
        
        for m in models:
            if m not in res_single: continue
            rmse_spatial = res_spatial[m]['rmse']
            rmse_single = res_single[m]['rmse']
            imp_abs = rmse_single - rmse_spatial
            imp_rel = (imp_abs / rmse_single) * 100 if rmse_single > 0 else 0
            
            summary_data.append({
                'Model': m,
                'Single RMSE': rmse_single,
                'Spatial RMSE': rmse_spatial,
                'Gain (MW)': imp_abs,
                'Gain (%)': imp_rel
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False, float_format="%.3f"))
        
        # Plotting
        labels = df_summary['Model']
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        rects1 = ax.bar(x - width/2, df_summary['Single RMSE'], width, label='Single Station', color='gray')
        rects2 = ax.bar(x + width/2, df_summary['Spatial RMSE'], width, label='Spatial Augmentation', color='skyblue')
        
        ax.set_ylabel('RMSE (MW)')
        ax.set_title('Spatial Augmentation Gain by Model (V05)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        
        ax.bar_label(rects1, padding=3, fmt='%.2f')
        ax.bar_label(rects2, padding=3, fmt='%.2f')
        
        plt.tight_layout()
        plt.savefig('results/v05_spatial_gain_chart.png')
        print("\nChart saved to results/v05_spatial_gain_chart.png")
    else:
        print("\nSkipping Comparison (One or both benchmarks disabled)")

if __name__ == "__main__":
    main()
