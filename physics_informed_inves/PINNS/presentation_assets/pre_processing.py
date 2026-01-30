import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import pvlib
from pvlib import location, pvsystem, modelchain
from pvlib.modelchain import ModelChain

from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import re



def get_station_metadata(station_num, df):
    """
    Retrieves and parses metadata for a specific station from the DataFrame.
    
    Args:
        station_num (int or str): The station number (e.g., 5 or '05').
        df (pd.DataFrame): The source dataframe containing station metadata.
        
    Returns:
        dict: A clean dictionary with numerical values ready for simulation.
    """
    
    # 1. Format the station ID
    station_id_str = f"station{str(station_num).zfill(2)}"
    
    # 2. Filter the dataframe
    station_row = df[df['Station_ID'] == station_id_str]
    
    if station_row.empty:
        raise ValueError(f"Station ID {station_id_str} not found in DataFrame.")
    
    # Get the single row as a Series
    row = station_row.iloc[0]

    # --- Helper 1: Parse "Key:Value" blocks (lines separated by \n) ---
    def parse_kv_string(text_block):
        data = {}
        if isinstance(text_block, str):
            for item in text_block.split('\n'):
                if ':' in item:
                    k, v = item.split(':', 1)
                    data[k.strip()] = v.strip()
        return data

    # --- Helper 2: Extract numeric values safely (e.g., "250 Wp" -> 250.0) ---
    def extract_num(val, default=0.0):
        if pd.isna(val): return default
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else default

    # 3. Parse complex text columns into dictionaries
    module_items = parse_kv_string(row.get('Module', ''))
    inverter_items = parse_kv_string(row.get('Inverters', ''))
    layout_items = parse_kv_string(row.get('Layout', ''))

    # 4. Build the final clean dictionary
    # We use extract_num to ensure we pass numbers (floats/ints) to the model, not strings.
    metadata = {
        # --- Identity & Location ---
        'Station_ID': station_id_str,
        'Longitude': float(row['Longitude']),
        'Latitude': float(row['Latitude']),
        'Array_Tilt': row['Array_Tilt'], # Kept as string (e.g., "South 30") for parsing in main model
        
        # --- System Capacity & Dimensions ---
        'Capacity': float(row['Capacity']),          # Total Station Capacity (kW)
        'Panel_Size': float(row.get('Panel_Size', 1.62)), # Panel Area in m^2
        'Total_Panel_Number': int(row.get('Panel_Number', 0)),
        'PV_Technology': row['PV_Technology'],

        # --- Extracted Module Specs (Cleaned to Float) ---
        'Module_Pmax': extract_num(module_items.get('Pmax'), default=250),
        'Module_Vmpp': extract_num(module_items.get('Vmpp'), default=30),
        'Module_Impp': extract_num(module_items.get('Impp'), default=8),
        
        # --- Extracted Inverter Specs (Cleaned to Float) ---
        'Inverter_Rated_Power': extract_num(inverter_items.get('Rated power'), default=500),
        'Inverter_Max_DC_Voltage': extract_num(inverter_items.get('Max. DC voltage'), default=1000),
        
        # --- Extracted Layout Specs (Cleaned to Int) ---
        # Note: Layout strings can look like "modules per string:20"
        'Modules_per_String': int(extract_num(layout_items.get('modules per string'), default=20)),
        'Strings_per_Inverter': int(extract_num(layout_items.get('strings per inverter'), default=100)),
    }
    
    return metadata



def calculate_nwp_power(metadata, df):
    """
    Calculates expected PV Power output (MW) based purely on NWP Weather Forecasts.
    Includes fix for date_time NaN and incorrect index parsing.
    """
    station_id = metadata.get('Station_ID', 'Unknown')
    print(f"--- Calculating NWP Physical Power & Sun Elevation for {station_id} ---")
    
    # ---------------------------------------------------------
    # 1. SETUP LOCATION & TIME
    # ---------------------------------------------------------
    lat = metadata['Latitude']
    lon = metadata['Longitude']
    
    # Define Location (UTC to align with solar position)
    site = location.Location(lat, lon, tz='UTC')
    
    # --- FIX: Robust Datetime Indexing ---
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check if 'date_time' column exists (it does in CSV)
        if 'date_time' in df.columns:
            # FIX: Remove 'format' argument to allow pandas to infer if seconds (00:00:00) exist
            df['date_time'] = pd.to_datetime(df['date_time']) 
            df.set_index('date_time', inplace=True)
        else:
            # Fallback if column is missing
            df.index = pd.to_datetime(df.index)
    elif df.index.tz is None:
        # If it is a DatetimeIndex but naive, localize it
        df.index = df.index.tz_localize('UTC')

    # ---------------------------------------------------------
    # 2. CALCULATE SUN ELEVATION
    # ---------------------------------------------------------
    solpos = site.get_solarposition(df.index)
    df['sun_elevation'] = solpos['elevation']

    # ---------------------------------------------------------
    # 3. PREPARE WEATHER DATA
    # ---------------------------------------------------------
    weather_nwp = pd.DataFrame(index=df.index)
    
    weather_nwp['ghi'] = df['nwp_globalirrad']
    weather_nwp['dni'] = df['nwp_directirrad']
    weather_nwp['temp_air'] = df['nwp_temperature']
    weather_nwp['wind_speed'] = df['nwp_windspeed']
    
    if 'nwp_pressure' in df.columns:
        weather_nwp['pressure'] = df['nwp_pressure'] * 100 
    else:
        weather_nwp['pressure'] = 101325.0

    zenith_rad = np.radians(solpos['zenith'])
    weather_nwp['dhi'] = weather_nwp['ghi'] - (weather_nwp['dni'] * np.cos(zenith_rad))
    
    weather_nwp['dhi'] = weather_nwp['dhi'].clip(lower=0.0)
    weather_nwp['dni'] = weather_nwp['dni'].clip(lower=0.0)
    weather_nwp['ghi'] = weather_nwp['ghi'].clip(lower=0.0)

    # ---------------------------------------------------------
    # 4. DEFINE PV SYSTEM
    # ---------------------------------------------------------
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except:
        tilt = 33.0
        
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_']
    ivt_para["Pdco"], ivt_para['Vdco'], ivt_para["Vdcmax"], ivt_para['Idcmax'] = 567000, 315, 1000, 1134
    ivt_para["Mppt_low"], ivt_para['Mppt_high'] = 460, 950
    
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=180, 
        module_parameters=module_params,
        inverter_parameters=ivt_para,
        temperature_model_parameters=temp_model,
        modules_per_string=metadata['Modules_per_String'],
        strings_per_inverter=metadata['Strings_per_Inverter']
    )

    # ---------------------------------------------------------
    # 5. RUN PHYSICAL SIMULATION
    # ---------------------------------------------------------
    print("Running ModelChain with NWP Weather...")
    
    mc = ModelChain(system, site, 
                    transposition_model='perez',
                    solar_position_method='nrel_numpy',
                    aoi_model='physical', 
                    spectral_model='no_loss')

    mc.run_model(weather_nwp)
    
    # ---------------------------------------------------------
    # 6. SCALE TO FULL STATION CAPACITY
    # ---------------------------------------------------------
    block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    station_total_capacity_watts = metadata['Capacity'] * 1000
    
    if block_dc_watts > 0:
        scaling_factor = station_total_capacity_watts / block_dc_watts
    else:
        scaling_factor = 1.0 
        
    nwp_power_mw = (mc.results.ac.fillna(0) * scaling_factor) / 1_000_000
    nwp_power_mw = nwp_power_mw.clip(lower=0.0)
    
    df['NWP_Power_MW'] = nwp_power_mw
    
    print(f"Done. Mean Predicted Power: {nwp_power_mw.mean():.4f} MW | Sun Elevation calculated.")
    
    return df


def calculate_clearsky_indices(metadata, df):
    """
    Calculates both Irradiance (K_CS) and Power (K_PV) Clear Sky Indices.
    Includes fix for date_time NaN caused by strict format mismatch.
    """
    
    print(f"--- Processing {metadata.get('Station_ID', 'Unknown')} ---")
    
    # ---------------------------------------------------------
    # 1. SETUP LOCATION & TIME
    # ---------------------------------------------------------
    lat = metadata['Latitude']
    lon = metadata['Longitude']
    
    site = location.Location(lat, lon, tz='UTC')
    
    # --- FIX: Ensure index is Datetime and Localized to UTC ---
    if not isinstance(df.index, pd.DatetimeIndex):
        # FIX: Remove strict 'format' argument to handle seconds (e.g. 00:00:00)
        df['date_time'] = pd.to_datetime(df['date_time']) 
        df.set_index('date_time', inplace=True)
        
        # FIX: Explicitly localize to UTC
        df.index = df.index.tz_localize('UTC')
    elif df.index.tz is None:
        # If it is a DatetimeIndex but naive, localize it
        df.index = df.index.tz_localize('UTC')
        
    # ---------------------------------------------------------
    # 2. CALCULATE CLEAR SKY IRRADIANCE (GHI_clr)
    # ---------------------------------------------------------
    print("Calculating Clear Sky Irradiance (Ineichen model)...")
    
    cs = site.get_clearsky(df.index, model='ineichen')
    df['GHI_clr'] = cs['ghi']

    if 'lmd_totalirrad' in df.columns:
        meas_ghi = df['lmd_totalirrad']
        df['K_CS_day'] = np.where(
            df['GHI_clr'] > 10, 
            meas_ghi / df['GHI_clr'], 
            0.0
        )
        df['K_CS_dayNight'] = meas_ghi / df['GHI_clr']
        df['K_CS'] = df['K_CS_day']
        df['K_CS'] = df['K_CS'].clip(lower=0.0, upper=1.25)
    else:
        print("Warning: 'lmd_totalirrad' column missing. K_CS not calculated.")

    # ---------------------------------------------------------
    # 3. SETUP PV SYSTEM MODEL (For P_CLR)
    # ---------------------------------------------------------
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except Exception as e:
        print(f"Error extracting tilt: {e} .... Assuming 33 degrees")
        tilt = 33.0
    
    azimuth = 180 

    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    block_dc_watts = (metadata['Modules_per_String'] * metadata['Strings_per_Inverter'] * metadata['Module_Pmax'])
    
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_']
    ivt_para["Pdco"], ivt_para['Vdco'], ivt_para["Vdcmax"], ivt_para['Idcmax'] = 567000, 315, 1000, 1134
    ivt_para["Mppt_low"], ivt_para['Mppt_high'] = 460, 950
    inverter_parameters = ivt_para
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

    # ---------------------------------------------------------
    # 4. RUN MODEL CHAIN FOR POWER
    # ---------------------------------------------------------
    print("Running PV Power Simulation...")

    mc = ModelChain(system, site, transposition_model='perez',
                        solar_position_method='nrel_numpy',
                        aoi_model='physical', spectral_model='no_loss')

    mc.run_model(cs)

    # ---------------------------------------------------------
    # 5. SCALE & CALCULATE POWER INDEX (K_PV)
    # ---------------------------------------------------------
    station_total_capacity_watts = metadata['Capacity'] * 1000
    scaling_factor = station_total_capacity_watts / block_dc_watts
    
    p_clr_watts = mc.results.ac.fillna(0) * scaling_factor
    df['P_CLR'] = p_clr_watts / 1_000_000 

    if 'power' in df.columns:
        meas_power = df['power']
        if meas_power.max() > 500:
            meas_power = meas_power / 1000.0
            
        df['K_PV'] = np.where(
            df['P_CLR'] > 1.0, 
            meas_power / df['P_CLR'], 
            0.0
        )
        df['K_PV'] = df['K_PV'].clip(lower=0.0, upper=1.5)
    else:
        print("Warning: 'power' column missing. K_PV not calculated.")

    return df


def plot_nwp_vs_actual_interactive(df, confidence_mw=None):
    """
    Plots NWP_Power_MW vs actual power (MW) using Plotly with an Office Color Scheme.
    Includes a shaded Confidence Interval (CI) around the NWP prediction.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'NWP_Power_MW' and 'power'.
                           Index must be datetime.
        confidence_mw (float, optional): The fixed margin for the confidence interval in MW.
                                         If None, calculates 95% CI (1.96 * STD of residuals).
    
    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure.
    """
    
    # --- 1. PREPARE DATA ---
    # Handle date_time: could be index or column
    if 'date_time' in df.columns:
        # date_time is a column (e.g. after reset_index)
        df = df.set_index('date_time')
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Convert to Asia/Shanghai timezone for display
    if df.index.tz is not None:
        dates = df.index.tz_convert('Asia/Shanghai')
    else:
        # If naive, assume UTC and convert
        dates = df.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
    nwp = df['NWP_Power_MW']
    
    # Handle 'power' column (Actual Power)
    if 'power' in df.columns:
        actual = df['power']
        # SAFEGUARD: If 'power' mean is > 50 (likely kW for large plant), divide by 1000.
        if actual.mean() > 50: 
            actual = actual / 1000.0
    else:
        # Fallback if no actual power
        actual = pd.Series([0]*len(df), index=df.index)

    # --- 2. DYNAMIC METRICS CALCULATION (ROLLING WINDOW) ---
    # Calculate Residuals (NWP - Actual)
    residuals = nwp - actual
    
    # Global Metrics (for annotation)
    mae = residuals.abs().mean()
    
    total_actual = actual.sum()
    if total_actual > 0:
        wape = (residuals.abs().sum() / total_actual) * 100.0
    else:
        wape = 0.0
        
    # Confidence Interval Width Calculation
    if confidence_mw is None:
        # Dynamic: Rolling Standard Deviation (Window = 96 points, ~24 hours at 15-min res)
        # Using min_periods=1 to ensure we have values at the start/end
        rolling_std = residuals.rolling(window=96, min_periods=1, center=True).std()
        
        # 95% Confidence Interval
        ci_width = 1.96 * rolling_std
        
        # Zero-Power Constraint (Nighttime Masking)
        # If both NWP and Actual are effectively zero, the CI should be zero.
        # Using a small threshold (e.g. 0.01 MW) to account for floating point noise
        is_night = (nwp <= 0.01) & (actual <= 0.01)
        ci_width[is_night] = 0.0
        
        ci_label_text = "Dynamic 95% CI (Window=96)"
        
        # For annotation, show average width
        avg_ci_width = ci_width.mean()
        metrics_ci_text = f"Avg CI Width: +/- {avg_ci_width:.2f} MW"
    else:
        # Fixed
        ci_width = float(confidence_mw)
        ci_label_text = f"Fixed CI (+/- {ci_width:.2f} MW)"
        metrics_ci_text = f"CI Width: +/- {ci_width:.2f} MW"

    # Calculate Confidence Bounds (Vectorized)
    upper_bound = nwp + ci_width
    lower_bound = (nwp - ci_width).clip(lower=0.0)
    
    # --- 3. DEFINE OFFICE COLOR SCHEME ---
    COLOR_ORANGE = '#ED7D31'   # Office Orange (NWP)
    COLOR_BLUE   = '#2F5597'   # Office Blue (Actual)
    # Faded Orange for Confidence Band
    COLOR_ORANGE_FADE = 'rgba(237, 125, 49, 0.2)' 

    # --- 4. CREATE PLOT ---
    fig = go.Figure()

    # Trace 1: Confidence Band (Lower Bound - Transparent, for filling)
    fig.add_trace(go.Scatter(
        x=dates, 
        y=lower_bound,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
        name='Lower Bound'
    ))

    # Trace 2: Confidence Band (Upper Bound - Filled to Lower)
    fig.add_trace(go.Scatter(
        x=dates, 
        y=upper_bound,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor=COLOR_ORANGE_FADE,
        name=ci_label_text,
        hoverinfo='skip'
    ))

    # Trace 3: Actual Power (Blue)
    fig.add_trace(go.Scatter(
        x=dates, 
        y=actual,
        mode='lines',
        name='Actual Power (MW)',
        line=dict(color=COLOR_BLUE, width=2)
    ))

    # Trace 4: NWP Power (Orange)
    fig.add_trace(go.Scatter(
        x=dates, 
        y=nwp,
        mode='lines',
        name='NWP Power (MW)',
        line=dict(color=COLOR_ORANGE, width=3, dash='solid')
    ))

    # --- 5. SECONDARY AXIS TRACES (POA-BASED ANALYSIS) ---
    # Check if POA columns exist
    if 'poa_global_calc' in df.columns:
        poa_raw = df['poa_global_calc']
        
        # Use poa_global_calc_norm if it exists, otherwise normalize
        if 'poa_global_calc_norm' in df.columns:
            poa_norm = df['poa_global_calc_norm']
        else:
            # Min-Max Normalize
            poa_min, poa_max = poa_raw.min(), poa_raw.max()
            if poa_max > poa_min:
                poa_norm = (poa_raw - poa_min) / (poa_max - poa_min)
            else:
                poa_norm = poa_raw * 0.0  # All zeros if no variance
        
        # Normalize actual power for comparison
        actual_min, actual_max = actual.min(), actual.max()
        if actual_max > actual_min:
            power_norm = (actual - actual_min) / (actual_max - actual_min)
        else:
            power_norm = actual * 0.0
        
        # Calculate Daily Accumulated Error
        # Error = poa_norm - power_norm (how much POA overestimates or underestimates)
        instantaneous_error = poa_norm - power_norm
        
        # Create "daylight segments" - reset accumulation when power is ~0 (night)
        # Detect transitions: when power goes from 0 to non-zero (sunrise)
        is_night = (actual <= 0.01)  # Night mask (power effectively 0)
        
        # Create segment IDs: increment when transitioning FROM night TO day
        # A new segment starts each time we go from night to daytime
        sunrise_mask = (~is_night) & is_night.shift(1, fill_value=True)
        segment_id = sunrise_mask.cumsum()
        
        # Accumulate within each daytime segment
        df_temp = pd.DataFrame({
            'error': instantaneous_error.values,
            'segment': segment_id.values
        }, index=dates)
        
        # Cumulative sum within each segment, resetting at segment boundaries (sunrise)
        daily_accumulated_error = df_temp.groupby('segment')['error'].cumsum()
        
        # Force night hours to 0 (no accumulation during night)
        daily_accumulated_error[is_night.values] = 0.0
        
        # Colors (Office Palette)
        COLOR_GREEN_OFFICE = '#70AD47'   # POA Raw
        COLOR_PURPLE_OFFICE = '#7030A0'  # POA Normalized
        COLOR_RED_OFFICE = '#C00000'     # Accumulated Error
        
        # Trace 5: POA Raw (Green Dotted)
        fig.add_trace(go.Scatter(
            x=dates,
            y=poa_raw,
            mode='lines',
            name='POA Global (W/m²)',
            line=dict(color=COLOR_GREEN_OFFICE, width=2, dash='dot'),
            yaxis='y2',
            visible='legendonly'  # Hidden by default
        ))
        
        # Trace 6: POA Normalized (Purple Dotted)
        fig.add_trace(go.Scatter(
            x=dates,
            y=poa_norm,
            mode='lines',
            name='POA Normalized',
            line=dict(color=COLOR_PURPLE_OFFICE, width=2, dash='dot'),
            yaxis='y2'
        ))
        
        # Trace 7: Daily Accumulated Error (Red Solid)
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_accumulated_error.values,
            mode='lines',
            name='Daily Accum. Error (POA_norm - Power_norm)',
            line=dict(color=COLOR_RED_OFFICE, width=2, dash='solid'),
            yaxis='y2'
        ))
        # --- HOURLY VARIABILITY RED OVERLAY ---
        # Calculate mean absolute change per hour
        poa_diff = poa_raw.diff().abs()  # Absolute instantaneous change
        
        # Create a DataFrame for hourly aggregation
        hourly_df = pd.DataFrame({
            'poa_diff': poa_diff.values,
            'poa_raw': poa_raw.values
        }, index=df.index)
        
        # Group by hour and calculate mean variability per hour
        hourly_variability = hourly_df['poa_diff'].resample('1h').mean()
        
        # Calculate rolling 3-hour mean of variability (shift by 1 to exclude current hour)
        prev_3h_mean = hourly_variability.shift(1).rolling(window=3, min_periods=1).mean()
        
        # Flag hours where current variability > 75% greater than prev 3H mean
        # i.e., current > 1.75 * prev_3h_mean
        is_high_variability_hour = hourly_variability > (1.75 * prev_3h_mean)
        
        # Also require some minimum variability to avoid flagging calm periods
        min_variability = 20  # W/m² per step minimum to consider
        is_high_variability_hour = is_high_variability_hour & (hourly_variability > min_variability)
        
        # Get the flagged hour timestamps
        flagged_hours = is_high_variability_hour[is_high_variability_hour].index
        
        if len(flagged_hours) > 0:
            # Create mask for original data: True if timestamp falls within a flagged hour
            hour_labels = df.index.floor('1h')
            is_flagged = hour_labels.isin(flagged_hours)
            
            # Mask POA values: show only flagged hours, NaN for others
            poa_flagged = poa_raw.where(is_flagged)
            
            # Trace 8: Red Overlay for High Variability Hours
            fig.add_trace(go.Scatter(
                x=dates,
                y=poa_flagged,
                mode='lines',
                name=f'High Variability ({len(flagged_hours)} hrs)',
                line=dict(color='#C00000', width=3),
                yaxis='y2',
                hovertemplate='<b>High Variability Hour</b><br>Time: %{x}<br>POA: %{y:.1f} W/m²<extra></extra>'
            ))

    # --- 6. LAYOUT STYLING & ANNOTATIONS ---
    
    # Metrics Text for Annotation
    metrics_text = (
        f"<b>Model Performance</b><br>"
        f"Mean Dev (MAE): {mae:.2f} MW<br>"
        f"Deviation (WAPE): {wape:.1f}%<br>"
        f"{metrics_ci_text}"
    )
    
    fig.update_layout(
        title=dict(
            text="NWP Power vs Actual Power (with Dynamic Confidence Intervals)",
            font=dict(size=20, family="Arial")
        ),
        xaxis=dict(
            title="Time",
            showgrid=True,
            gridcolor='#E5E5E5'
        ),
        yaxis=dict(
            title="Power (MW)",
            showgrid=True,
            gridcolor='#E5E5E5'
        ),
        yaxis2=dict(
            title="POA / Normalized / Accum. Error",
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=True,
            zerolinecolor='#E5E5E5'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            orientation="v",  # Vertical legend
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.15  # Positioned to the right of the plot
        ),
        margin=dict(t=100, r=200), # Increase right margin for legend
        hovermode="x unified",
        # Add Annotation for Metrics
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=metrics_text,
                showarrow=False,
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12, color="black")
            )
        ]
    )

    return fig

