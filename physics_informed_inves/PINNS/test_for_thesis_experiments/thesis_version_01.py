import pandas as pd
import numpy as np
import re
from pvlib import location, pvsystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

def calculate_clearsky_indices(metadata, df):
    """
    Calculates both Irradiance (K_CS) and Power (K_PV) Clear Sky Indices.
    
    This function computes two critical indices for solar resource and system performance analysis:
    
    1. K_CS (Clear Sky Index for Irradiance): 
       - Definition: Ratio of Measured GHI to Modeled Clear Sky GHI.
       - Interpretation: strictly an ATMOSPHERIC index. It describes sky brightness relative to a 
         clear sky model. Values near 1.0 indicate clear sky; values << 1.0 indicate clouds; 
         values > 1.0 indicate cloud-edge enhancement. 
       - Note: It does NOT measure array performance, shading, or soiling.
       
    2. K_PV (Clear Sky Performance Index for Power):
       - Definition: Ratio of Measured AC Power to Modeled Clear Sky AC Power.
       - Interpretation: A normalized plant performance metric. 
         - 0.95 - 1.05: Normal operation (close to model).
         - < 0.95: Underperformance (shading, soiling, faults).
         - > 1.05: Warning flag. Suggests modeling errors (wrong capacity/tilt) or measurement issues, 
           rather than physical "overperformance".
         - > 1.25: Hard clipping cap (likely unphysical/data error).

    Handling of Zeros and NaNs:
    - Zeros are assigned when the denominator (Clear Sky GHI or Power) is below a threshold 
      (nighttime or very low signal).
    - These 0.0 values are useful for maintaining a continuous time series for plotting/masking.
    - CRITICAL: For statistics (means, quantiles, distributions), these 0.0 values must be TREATED AS INVALID 
      and excluded. A value of 0.0 does not mean "zero quality", it means the index is undefined/undefined physics.

    Corrections Applied during calculation:
    1. Fixed Azimuth: Changed from Tracking (solar_azimuth) to Fixed South (180).
    2. Weather Injection: Added measured temperature/wind to Clear Sky data for accurate cell temp modeling.
    3. Robust Indexing: Ensures UTC localization without errors.
    """
    
    station_id = metadata.get('Station_ID', 'Unknown')
    print(f"--- Processing {station_id} ---")
    
    # ---------------------------------------------------------
    # 1. SETUP LOCATION & TIME
    # ---------------------------------------------------------
    lat = float(metadata['Latitude'])
    lon = float(metadata['Longitude'])
    
    site = location.Location(lat, lon, tz='UTC')
    
    # --- Ensure index is Datetime and Localized to UTC ---
    # We work on a copy to avoid SettingWithCopy warnings on the original df
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df['date_time'] = pd.to_datetime(df['date_time']) 
        df.set_index('date_time', inplace=True)
    
    # Ensure UTC awareness
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    # ---------------------------------------------------------
    # 2. CALCULATE CLEAR SKY IRRADIANCE (GHI_clr)
    # ---------------------------------------------------------
    print("Calculating Clear Sky Irradiance (Ineichen model)...")
    
    # Calculate clear sky components (ghi, dni, dhi)
    # Ineichen–Perez clear-sky irradiance model
    cs = site.get_clearsky(df.index, model='ineichen')
    df['GHI_clr'] = cs['ghi']

    # Calculate K_CS (Irradiance Index)
    if 'lmd_totalirrad' in df.columns:
        meas_ghi = df['lmd_totalirrad']
        
        # Avoid division by zero at night or low light
        # Only calculate K_CS when clear sky GHI > 10 W/m^2
        # This threshold excludes nighttime/twilight noise.
        # NOTE: The resulting 0.0 values at night are for MASKING ONLY.
        # Exclude them from any statistical analysis or aggregation.
        df['K_CS'] = np.where(
            df['GHI_clr'] > 10, 
            meas_ghi / df['GHI_clr'], 
            0.0
        )
        # Clip to physical limits (allow slightly >1 for cloud enhancement)
        # Values > 1.25 are truncated as they exceed physical cloud enhancement limits.
        df['K_CS'] = df['K_CS'].clip(lower=0.0, upper=1.25)
    else:
        print("Warning: 'lmd_totalirrad' column missing. K_CS not calculated.")

    # ---------------------------------------------------------
    # 3. SETUP PV SYSTEM MODEL (For P_CLR)
    # ---------------------------------------------------------
    # A. Parse Tilt
    try:
        tilt_str = str(metadata.get('Array_Tilt'))
        tilt_match = re.search(r"[\d.]+", tilt_str)
        tilt = float(tilt_match.group()) if tilt_match else 33.0
    except Exception as e:
        print(f"Error extracting tilt: {e} .... Assuming 33 degrees")
        tilt = 33.0
    
    # B. Define Hardware Parameters
    module_params = pvsystem.retrieve_sam('CECMod')['Yingli_Energy__China__YL250P_29b']
    
    # Inverter Parameters (Updated for TBEA TC500KH)
    # Metadata: "TC500KH", Max DC Input 618kW, Max DC Voltage 1000V, Max DC Current 1344A, Rated 500kW
    # Since specific coefficients for TBEA TC500KH might not be in the publicly available CEC database,
    # we adapt the generic 500kW profile but Strictly Enforce the provided datasheet limits.
    ivt_para = pvsystem.retrieve_sam('cecinverter')['Advanced_Energy_Industries__Solaron_500kW__3159500_XXXX___480V_'].copy()
    
    # Overwrite with datasheet specifics provided by User
    ivt_para["Paco"] = 500000    # Rated AC Power (500 kW)
    ivt_para["Pdco"] = 618000    # Max DC Input (618 kW) - dictated by metadata
    ivt_para["Vdcmax"] = 1000    # Max DC Voltage (1000 V)
    ivt_para['Idcmax'] = 1344    # Max DC Current (1344 A)
    
    # Keep reasonable defaults for efficiency curve if not provided
    ivt_para['Vdco'] = 315       # Nominal DC voltage (kept from reference if not specified)
    ivt_para["Mppt_low"] = 460   # MPPT window (Standard range)
    ivt_para['Mppt_high'] = 950
    
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    
    # C. Build System (FIXED TILT)
    # Critical Fix: surface_azimuth=180 (South), NOT df['solar_azimuth'] (Tracking)
    system = pvsystem.PVSystem(
        surface_tilt=tilt,
        surface_azimuth=180, 
        module_parameters=module_params,
        inverter_parameters=ivt_para,
        temperature_model_parameters=temp_model,
        modules_per_string=int(metadata['Modules_per_String']),
        strings_per_inverter=int(metadata['Strings_per_Inverter'])
    )

    # ---------------------------------------------------------
    # 4. RUN MODEL CHAIN FOR POWER
    # ---------------------------------------------------------
    print("Running PV Power Simulation...")

    # Critical Fix: Inject Measured Temperature/Wind into Clear Sky Data
    # The 'cs' dataframe only has irradiance. The model needs temp to calculate efficiency losses.
    # We assume "Clear Sky Sun" but "Actual Ambient Temperature".
    cs['temp_air'] = df.get('lmd_temperature', 20)  # Default to 20C if missing
    cs['wind_speed'] = df.get('lmd_windspeed', 0)   # Default to 0 m/s if missing

    mc = ModelChain(system, site, 
                    transposition_model='perez',
                    solar_position_method='nrel_numpy',
                    aoi_model='physical', 
                    spectral_model='no_loss')

    # Run the model using the ENRICHED clear sky data
    mc.run_model(cs)

    # ---------------------------------------------------------
    # 5. SCALE & CALCULATE POWER INDEX (K_PV)
    # ---------------------------------------------------------
    # Calculate Scaling Factor (Total Station Capacity / One Inverter Block)
    block_dc_watts = (int(metadata['Modules_per_String']) * int(metadata['Strings_per_Inverter']) * float(metadata['Module_Pmax']))
    station_total_capacity_watts = float(metadata['Capacity']) * 1000
    scaling_factor = station_total_capacity_watts / block_dc_watts if block_dc_watts > 0 else 1.0
    
    # Save P_CLR (Predicted Clear Sky Power in MW)
    p_clr_watts = mc.results.ac.fillna(0) * scaling_factor
    df['P_CLR'] = p_clr_watts / 1_000_000 

    # Calculate K_PV
    if 'power' in df.columns:
        meas_power = df['power']
        
        # Determine if power is in kW or MW. If max > 500, likely kW (convert to MW).
        if meas_power.max() > 500:
            meas_power = meas_power / 1000.0
            
        # K_PV = Measured Power / Clear Sky Power
        # Calculated only when P_CLR is significant (> 0.1 MW).
        # Intepretation Guide:
        # - Near 1.0 (0.95-1.05): Optimal Operation
        # - > 1.05: Flag for potential model mismatch or sensor error.
        # - 0.0 (Mask): Nighttime or low model power. EXCLUDE from stats.
        df['K_PV'] = np.where(
            df['P_CLR'] > 0.1, 
            meas_power / df['P_CLR'], 
            0.0
        )
        # Clip to physical limits
        # We enforce a hard cap at 1.25 to remove gross outliers.
        # Values between 1.05 and 1.25 should be investigated as anomalies.
        df['K_PV'] = df['K_PV'].clip(lower=0.0, upper=1.25)
    else:
        print("Warning: 'power' column missing. K_PV not calculated.")

    return df
