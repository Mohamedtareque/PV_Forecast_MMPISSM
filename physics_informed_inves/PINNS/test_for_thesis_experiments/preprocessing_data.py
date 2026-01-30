import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def check_sensor_misalignment(df, irradiance_col='lmd_totalirrad', power_col='power', station_name="Station"):
    """
    Replicates Figure 6a: Power vs Irradiance Scatter Plot with Linear Regression.
    
    Args:
        df (pd.DataFrame): Dataframe containing the data.
        irradiance_col (str): Column name for In-Plane Irradiance (X-axis).
        power_col (str): Column name for Power Output (Y-axis).
        station_name (str): Name of the station for the title.
    """
    print(f"\n--- [Step 1] Quality Check: Sensor Misalignment for {station_name} ---")
    
    # 1. CLEAN DATA
    # Create a clean copy dropping NaNs in the relevant columns
    # We use .copy() to avoid SettingWithCopyWarning
    df_clean = df[[irradiance_col, power_col]].dropna().copy()
    
    # Filter out very low irradiance to avoid noise (e.g., night time or sensor noise close to 0)
    # This helps the regression fit the main trend better.
    df_clean = df_clean[df_clean[irradiance_col] > 10]
    
    if df_clean.empty:
        print("Warning: No valid data found after filtering (NaNs or low irradiance).")
        return

    x = df_clean[irradiance_col].values
    y = df_clean[power_col].values
    
    # 2. FIT LINEAR REGRESSION
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    print(f"Linear Regression Fit:")
    print(f"  Slope: {slope:.4f}")
    print(f"  Intercept: {intercept:.4f}")
    print(f"  R-squared: {r_squared:.4f}")
    
    # 3. INTERPRETATION & FLAGGING
    print("\nValidation Check:")
    print("  -> Look for a 'split' pattern in the plot (two distinct slopes).")
    print("  -> If found, it implies: Sensor Tilt/Azimuth Mismatch OR Timestamp Offset.")
    
    if r_squared < 0.90:
         print(f"  [WARNING] Low R² ({r_squared:.2f}). Potential misalignment, timestamp offset, or noisy data detected.")
    else:
         print(f"  [PASS] High R² ({r_squared:.2f}). Data indicates consistent alignment (visual check still recommended).")

    # 4. PLOTTING
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=10, label='Measured Data', c='royalblue', edgecolors='none')
    
    # Plot Regression Line
    # Create a line from min to max x
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Fit (R²={r_squared:.2f})')
    
    plt.title(f"Quality Check 1: Sensor Misalignment Detection ({station_name})\nPower vs. In-Plane Irradiance")
    plt.xlabel(f"In-Plane Irradiance ({irradiance_col}) [W/m²]")
    plt.ylabel(f"Power Output ({power_col}) [MW]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

def check_seasonal_consistency(df, irradiance_col='lmd_totalirrad', power_col='power', time_col='date_time', station_name="Station"):
    """
    Plots Power vs Irradiance grouped by Season using Plotly for interactivity.
    Seasons (Northern Hemisphere standard):
      - Winter: Dec, Jan, Feb
      - Spring: Mar, Apr, May
      - Summer: Jun, Jul, Aug
      - Autumn: Sep, Oct, Nov
    """
    print(f"\n--- [Step 2] Quality Check: Seasonal Consistency for {station_name} ---")
    
    # Clean data
    df_clean = df[[irradiance_col, power_col, time_col]].dropna().copy()
    
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        
    df_clean['month'] = df_clean[time_col].dt.month
    
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    }
    df_clean['season'] = df_clean['month'].map(season_map)
    
    # Colors for seasons
    colors = {'Winter': 'blue', 'Spring': 'green', 'Summer': 'red', 'Autumn': 'orange'}
    
    # Create Plotly Figure
    fig = go.Figure()

    # Plot each season
    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        subset = df_clean[df_clean['season'] == season]
        if subset.empty:
            continue
            
        x = subset[irradiance_col]
        y = subset[power_col]
        
        # 1. Scatter Points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name=f'{season} Data',
            marker=dict(color=colors[season], size=4, opacity=0.4),
            legendgroup=season
        ))
        
        # 2. Fit Regression
        if len(subset) > 50:
            try:
                slope, intercept, _, _, _ = stats.linregress(x, y)
                # Create line points
                x_range = np.linspace(x.min(), x.max(), 100)
                y_range = slope * x_range + intercept
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_range,
                    mode='lines',
                    name=f'{season} Fit (Slope={slope:.3f})',
                    line=dict(color=colors[season], width=2, dash='dash'),
                    legendgroup=season
                ))
            except Exception as e:
                print(f"Could not fit regression for {season}: {e}")

    fig.update_layout(
        title=f"Quality Check 2: Seasonal Consistency ({station_name})<br>Click legend to toggle seasons",
        xaxis_title=f"In-Plane Irradiance ({irradiance_col}) [W/m²]",
        yaxis_title=f"Power Output ({power_col}) [MW]",
        template='plotly_white',
        height=700,
        legend=dict(groupclick="toggleitem") # Clicking checks/unchecks the whole group
    )
    
    fig.show()

def investigate_low_power_anomalies(df, power_col='power', poa_col='poa_global_calc', time_col='date_time', threshold_mw=1.0, min_irrad=50):
    """
    Interactive Plotly Graph to investigate Low Power Anomalies.
    STRICTLY Filters for times where Power < threshold_mw BUT Irradiance > min_irrad.
    PLOTS NORMALIZED VALUES (0-1) for easy comparison.
    HOVER shows ORIGINAL units AND Shanghai Time (Asia/Shanghai).
    
    Args:
        df: DataFrame
        power_col: Power column name (assumed kW or MW, label accordingly)
        poa_col: Plane of Array Irradiance column (W/m^2)
        time_col: Date time column
        threshold_mw: Upper limit for 'Low Power' (default 1.0)
        min_irrad: Lower limit for Irradiance to consider (default 50 W/m^2)
    """
    print(f"\n--- [Step 3] Quality Check: Low Power Anomalies (Interactive) ---")
    print(f"Filtering for Power < {threshold_mw} AND Irradiance > {min_irrad}...")
    
    # Prepare data
    df_clean = df.copy()
    
    # Handle index if needed
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
        
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # 1. Calculate Max for Normalization (Use GLOBAL max to preserve context)
    max_power = df_clean[power_col].max()
    max_poa = df_clean[poa_col].max()
    
    if max_power == 0 or pd.isna(max_power): max_power = 1.0
    if max_poa == 0 or pd.isna(max_poa): max_poa = 1.0

    # 2. STRICT FILTERING (using original values)
    mask = (df_clean[power_col] < threshold_mw) & (df_clean[poa_col] > min_irrad)
    df_anomalies = df_clean[mask].copy()
    
    if df_anomalies.empty:
        print("No anomalies found with current settings.")
        return

    # Sort by time
    df_anomalies = df_anomalies.sort_values(by=time_col)
    
    print(f"Found {len(df_anomalies)} anomaly records.")
    
    # 3. Create Normalized Columns for Plotting
    df_anomalies['power_norm'] = df_anomalies[power_col] / max_power
    df_anomalies['poa_norm'] = df_anomalies[poa_col] / max_poa
    
    # 4. Handle Timezone for Display (Convert UTC -> Shanghai)
    # Ensure it's localized to UTC first if naive (assuming input is UTC based on context)
    if df_anomalies[time_col].dt.tz is None:
        df_anomalies['time_utc'] = df_anomalies[time_col].dt.tz_localize('UTC')
    else:
        df_anomalies['time_utc'] = df_anomalies[time_col].dt.tz_convert('UTC')
        
    # Convert to Shanghai
    df_anomalies['time_shanghai'] = df_anomalies['time_utc'].dt.tz_convert('Asia/Shanghai')
    # Format as string for clean tooltip
    df_anomalies['time_shanghai_str'] = df_anomalies['time_shanghai'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Create the figure (Single Y-axis is sufficient since both are 0-1)
    fig = go.Figure()

    # Trace 1: Power Output
    fig.add_trace(
        go.Scatter(
            x=df_anomalies[time_col],
            y=df_anomalies['power_norm'],
            mode='markers', 
            name=f'Low Power (<{threshold_mw})',
            marker=dict(color='red', size=8, symbol='circle'),
            # Custom Data: [Original Power, Original POA, Shanghai Time]
            customdata=df_anomalies[[power_col, poa_col, 'time_shanghai_str']],
            hovertemplate=(
                "<b>UTC Time</b>: %{x}<br>" +
                "<b>Shanghai Time</b>: %{customdata[2]}<br>" +
                "<b>Norm Power</b>: %{y:.2f}<br>" +
                "<b>Original Power</b>: %{customdata[0]:.2f} kW<br>" + 
                "<extra></extra>"
            )
        )
    )

    # Trace 2: POA Irradiance
    fig.add_trace(
        go.Scatter(
            x=df_anomalies[time_col],
            y=df_anomalies['poa_norm'],
            mode='markers',
            name=f'High Irradiance (>{min_irrad})',
            marker=dict(color='orange', size=6, symbol='x'),
            opacity=0.7,
            # Custom Data: [Original Power, Original POA, Shanghai Time]
            customdata=df_anomalies[[power_col, poa_col, 'time_shanghai_str']],
            hovertemplate=(
                "<b>UTC Time</b>: %{x}<br>" +
                "<b>Shanghai Time</b>: %{customdata[2]}<br>" +
                "<b>Norm Irradiance</b>: %{y:.2f}<br>" +
                "<b>Original Irradiance</b>: %{customdata[1]:.2f} W/m²<br>" + 
                "<extra></extra>"
            )
        )
    )

    # Layout Configuration
    fig.update_layout(
        title=f'Quality Check 3: Low Power Anomalies (Normalized View)<br>Power < {threshold_mw} & Irrad > {min_irrad}',
        xaxis_title='Date Time (UTC)',
        yaxis_title='Normalized Value (0-1)',
        hovermode='closest', 
        template='plotly_white',
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    
    fig.show()

import pvlib

def check_shading_analysis_position(df, metadata=None, lat=None, lon=None, power_col='power', time_col='date_time', threshold_mw=0.1, min_irrad=10):
    """
    Shading Analysis: Plots 'Zero Power' (or low power) events on an Azimuth vs. Elevation chart.
    This helps identify fixed obstacles (trees, buildings) causing shading at specific times of year.
    
    Args:
        df: DataFrame
        metadata: Dictionary containing station metadata (optional).
        lat (float): Latitude (required if not in metadata and columns missing).
        lon (float): Longitude (required if not in metadata and columns missing).
        power_col: Power column
        threshold_mw: Threshold for "Zero/Low Power" (default 0.1 MW).
        min_irrad: Minimum irradiance to consider (default 10 W/m²) to avoid night noise.
    """
    print(f"\n--- [Step 4] Quality Check: Shading Analysis (Azimuth vs Elevation) ---")
    
    # Extract from metadata if provided
    if metadata:
        if lat is None: lat = float(metadata.get('Latitude', 0))
        if lon is None: lon = float(metadata.get('Longitude', 0))
    
    df_clean = df.copy()
    
    # Ensure datetime
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # 1. Ensure Solar Position Columns
    if 'azimuth' not in df_clean.columns or 'elevation' not in df_clean.columns:
        if lat is None or lon is None:
            print("[Error] 'azimuth' and 'elevation' columns missing, and Lat/Lon not provided.")
            return
        
        print(f"Calculating Solar Position for Lat:{lat}, Lon:{lon}...")
        try:
            site = pvlib.location.Location(lat, lon, tz='UTC')
            solpos = site.get_solarposition(df_clean[time_col])
            df_clean['azimuth'] = solpos['azimuth'].values
            df_clean['elevation'] = solpos['elevation'].values
        except Exception as e:
            print(f"[Error] Could not calculate solar position: {e}")
            return
    
    # 2. Filter for "Zero Power" Events (Anomaly)
    # We look for times where Power is LOW but Sun is UP (Elevation > 10 usually, or min_irrad checks that)
    # User said "Zero Power Events", so threshold should be small.
    mask_anomaly = (df_clean[power_col] < threshold_mw) & (df_clean['elevation'] > 10)
    
    # Optional: If irradiance col exists, allow filtering by that too? 
    # User didn't strictly ask, but it's good practice. 
    # "check if zero-power events happen... where power was less than 1 MW"
    
    df_events = df_clean[mask_anomaly].copy()
    
    if df_events.empty:
        print("No low-power events found during daylight (Elevation > 10°).")
        return

    print(f"Found {len(df_events)} shading/outage candidates.")
    
    # 3. Plotly Scatter (Azimuth vs Elevation)
    fig = go.Figure()
    
    # Add "Background" of all sun positions? (Optional, maybe too heavy)
    # Let's just plot the events.
    
    fig.add_trace(go.Scatter(
        x=df_events['azimuth'],
        y=df_events['elevation'],
        mode='markers',
        marker=dict(
            size=6,
            color=df_events[time_col].dt.month, # Color by Month to show seasonality
            colorscale='Viridis',
            colorbar=dict(title="Month"),
            showscale=True
        ),
        # Custom hover info
        text=df_events[time_col].dt.strftime('%Y-%m-%d %H:%M'),
        customdata=df_events[[power_col]],
        hovertemplate=(
            "<b>Time</b>: %{text}<br>" +
            "<b>Azimuth</b>: %{x:.1f}°<br>" +
            "<b>Elevation</b>: %{y:.1f}°<br>" +
            "<b>Power</b>: %{customdata[0]:.3f} MW<br>" +
            "<extra></extra>" 
        )
    ))
    
    fig.update_layout(
        title=f"Quality Check 4: Shading Analysis<br>Low Power Events (<{threshold_mw} MW) in Solar Coordinates",
        xaxis_title="Solar Azimuth (deg)",
        yaxis_title="Solar Elevation (deg)",
        xaxis=dict(range=[0, 360]),
        yaxis=dict(range=[0, 90]),
        template="plotly_white",
        height=600,
        width=800
    )
    
    fig.show()


def apply_horizon_mask(df, metadata=None, lat=None, lon=None, 
                       power_col='power', poa_col='poa_global_calc', time_col='date_time',
                       poa_threshold=400, near_zero_power_mw=0.02, max_shading_elev_cap=25.0):
    """
    Non-destructive Geometry-Locked Mismatch Detection (Potential Horizon Shading).
    
    Identifies data points where:
    1. The sun is CLEARLY present (POA is high, not marginal dawn/dusk).
    2. Power is near ZERO (not just "low"), indicating severe underperformance.
    3. The event is GEOMETRY-LOCKED: concentrated at specific low elevations and azimuths.
    
    This flags potential horizon shading, but could also include outages or snow cover
    that happen to correlate with geometry. The flag is named accordingly.
    
    Instead of deleting rows, this function adds a boolean flag 'is_geometry_locked_mismatch'.
    
    Args:
        df: DataFrame containing power, irradiance, and timestamps.
        metadata: Dictionary containing station metadata (optional).
        lat, lon: Location (needed if azimuth/elevation missing).
        power_col: Name of power column.
        poa_col: Name of POA irradiance column.
        time_col: Name of datetime column.
        poa_threshold: Minimum POA (W/m²) to consider "sun clearly present" (default 400).
                       Higher values reduce false positives from marginal light.
        near_zero_power_mw: Power threshold (MW) below which output is "near zero" (default 0.02).
                            This should be close to zero to capture true outages/shading.
        max_shading_elev_cap: Maximum elevation angle (degrees) for horizon shading (default 25).
                              True horizon obstructions rarely block sun above 25°.
    
    Returns:
        pd.DataFrame: Original dataframe with added 'is_geometry_locked_mismatch' column.
    """
    print(f"\n--- [Step 4] Quality Check: Geometry-Locked Mismatch Detection ---")
    print(f"Parameters: POA > {poa_threshold} W/m², Power < {near_zero_power_mw} MW, Elev Cap: {max_shading_elev_cap}°")

    # Extract from metadata if provided
    if metadata:
        if lat is None: lat = float(metadata.get('Latitude', 0))
        if lon is None: lon = float(metadata.get('Longitude', 0))
    
    df_clean = df.copy()
    
    # Ensure datetime
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # 1. Ensure Solar Position (Elevation & Azimuth)
    if 'azimuth' not in df_clean.columns or 'elevation' not in df_clean.columns:
        if lat is None or lon is None:
            print("[Error] Lat/Lon required to calculate Solar Position.")
            df_clean['is_geometry_locked_mismatch'] = False
            return df_clean
        
        print("Calculating Solar Position...")
        try:
            site = pvlib.location.Location(lat, lon, tz='UTC')
            solpos = site.get_solarposition(df_clean[time_col])
            df_clean['azimuth'] = solpos['azimuth'].values
            df_clean['elevation'] = solpos['elevation'].values
        except Exception as e:
            print(f"[Error] Solar Calc failed: {e}")
            df_clean['is_geometry_locked_mismatch'] = False
            return df_clean

    # 2. Identify STRICT Anomalies (High Irradiance, Near-Zero Power)
    # -----------------------------------------------------------------
    # Condition 1: POA is HIGH (sun is unambiguously present, not marginal)
    # Using user-specified threshold (default 400 W/m²)
    is_high_poa = df_clean[poa_col] > poa_threshold
    
    # Condition 2: Power is NEAR ZERO (not just "low", but truly abnormal)
    # Using absolute threshold (default 0.02 MW) to capture true shading/outage
    is_near_zero_power = df_clean[power_col] < near_zero_power_mw
    
    # Strict Mismatch: High POA but Near-Zero Power
    # This excludes normal low-sun behavior and captures true anomalies
    mismatch_mask = is_near_zero_power & is_high_poa
    
    n_mismatches = mismatch_mask.sum()
    if n_mismatches < 10:
        print(f"Only {n_mismatches} strict mismatches found. Insufficient for pattern analysis.")
        df_clean['is_geometry_locked_mismatch'] = False
        return df_clean
    
    print(f"Found {n_mismatches} strict mismatches (near-zero power with high POA)")

    # 3. Derive Geometric Thresholds from Data
    # -----------------------------------------
    mismatch_data = df_clean[mismatch_mask]
    
    # A. Elevation Threshold
    # Hypothesis: Horizon shading occurs at LOW elevations only.
    # We derive from data but CAP at a physically plausible value.
    # True horizon obstructions rarely block sun above 15-25°.
    data_driven_elev = np.percentile(mismatch_data['elevation'], 75)  # Use 75th, not 90th
    max_shading_elev = min(data_driven_elev, max_shading_elev_cap)
    
    print(f"Derived Max Shading Elevation: {max_shading_elev:.1f}° (data: {data_driven_elev:.1f}°, cap: {max_shading_elev_cap}°)")
    
    # B. Azimuth Analysis (Identify Problematic Sectors)
    # Focus on low-elevation mismatches only (geometry-consistent with horizon)
    low_elev_mismatches = mismatch_data[mismatch_data['elevation'] < max_shading_elev]
    
    if low_elev_mismatches.empty:
        print("No mismatches found at low elevation. No geometry-locked pattern detected.")
        df_clean['is_geometry_locked_mismatch'] = False
        return df_clean
    
    print(f"Analyzing {len(low_elev_mismatches)} mismatches at elevation < {max_shading_elev:.1f}°")

    # Histogram of mismatches vs all data per azimuth bin (10° bins)
    az_bins = np.arange(0, 361, 10)
    
    # Compare against ALL low-elevation data (normalized failure rate)
    low_elev_all = df_clean[df_clean['elevation'] < max_shading_elev]
    
    if low_elev_all.empty:
        print("No data at low elevation. Skipping analysis.")
        df_clean['is_geometry_locked_mismatch'] = False
        return df_clean
    
    hist_all, _ = np.histogram(low_elev_all['azimuth'], bins=az_bins)
    hist_fail, _ = np.histogram(low_elev_mismatches['azimuth'], bins=az_bins)
    
    # Avoid division by zero
    hist_all_safe = np.where(hist_all == 0, 1, hist_all)
    failure_rates = hist_fail / hist_all_safe
    
    # C. Statistical Outlier Detection for Azimuth Sectors
    # Instead of fixed 5% threshold, use mean + 2*std to identify outliers
    # This adapts to the baseline failure rate of the dataset
    mean_rate = np.mean(failure_rates[failure_rates > 0]) if (failure_rates > 0).any() else 0
    std_rate = np.std(failure_rates[failure_rates > 0]) if (failure_rates > 0).any() else 0
    outlier_threshold = mean_rate + 2 * std_rate
    
    # Ensure threshold is at least 10% to avoid flagging near-normal sectors
    outlier_threshold = max(outlier_threshold, 0.10)
    
    print(f"Azimuth Failure Rate: mean={mean_rate:.2%}, std={std_rate:.2%}, threshold={outlier_threshold:.2%}")
    
    # Identify problematic bins
    problematic_bins = np.where(failure_rates > outlier_threshold)[0]
    if len(problematic_bins) > 0:
        problematic_azimuths = [(b*10, (b+1)*10) for b in problematic_bins]
        print(f"Problematic Azimuth Sectors (rate > {outlier_threshold:.1%}): {problematic_azimuths}")
    else:
        print("No azimuth sectors exceed the outlier threshold.")
        df_clean['is_geometry_locked_mismatch'] = False
        return df_clean

    # 4. Apply Flagging Logic
    # -----------------------
    # Flag points that are:
    # - Strict Mismatch (near-zero power when POA is high)
    # - Low Elevation (< derived threshold, capped)
    # - In a Problematic Azimuth Sector (statistically outlier)
    
    bin_indices = (df_clean['azimuth'] // 10).astype(int).clip(0, len(failure_rates)-1)
    row_failure_rates = failure_rates[bin_indices]
    
    geometry_locked_mask = (
        mismatch_mask & 
        (df_clean['elevation'] < max_shading_elev) & 
        (row_failure_rates > outlier_threshold)
    )
    
    df_clean['is_geometry_locked_mismatch'] = geometry_locked_mask
    
    num_flagged = geometry_locked_mask.sum()
    
    # 5. Summary Statistics
    # ---------------------
    print(f"\n=== Summary ===")
    print(f"Total rows: {len(df_clean)}")
    print(f"Flagged as geometry-locked mismatch: {num_flagged} ({num_flagged/len(df_clean):.2%})")
    print(f"Criteria:")
    print(f"  - Power < {near_zero_power_mw} MW (near-zero)")
    print(f"  - POA > {poa_threshold} W/m² (sun clearly present)")
    print(f"  - Elevation < {max_shading_elev:.1f}° (horizon-level)")
    print(f"  - Azimuth in outlier sector (failure rate > {outlier_threshold:.1%})")
    print(f"\nNote: This flag captures potential horizon shading, but may also include")
    print(f"      geometry-correlated outages or snow cover. Use for loss masking, not deletion.")
    
    return df_clean


def remove_outages(df, power_col='power', irradiance_col='lmd_totalirrad', min_power=0.01, min_irrad=10):
    """
    Removes specific 'Outage' events where Power is effectively Zero but Irradiance is significant.
    This differs from 'Low Power Anomalies' in that it destructively cleans the dataframe 
    for subsequent high-quality analysis (like regression).
    
    Args:
        df: DataFrame
        power_col: Power column
        irradiance_col: Irradiance column (In-Plane or Global)
        min_power: Threshold below which is considered 'Zero Power' (default 0.01)
        min_irrad: Threshold above which we expect power (default 10 W/m²)
        
    Returns:
        pd.DataFrame: Cleaned dataframe with outages removed.
    """
    print(f"\n--- [Step 0.5] Data Cleaning: Removing Outages ---")
    
    df_clean = df.copy()
    
    # Logic: Drop if Power < min_power AND Irradiance > min_irrad
    is_outage = (df_clean[power_col] < min_power)  & (df_clean[irradiance_col] > min_irrad)
    
    num_removed = is_outage.sum()
    total = len(df_clean)
    
    if num_removed > 0:
        print(f"Applying Filter: Excluding records where Power < {min_power} AND Irradiance > {min_irrad}")
        print(f"Removed {num_removed} / {total} records ({num_removed/total:.1%}) detected as outages.")
        return df_clean[~is_outage].copy()
    else:
        print("No Outages removed.")
        return df_clean




def check_inverter_clipping(df, metadata=None, power_col='power', time_col='date_time', irradiance_col='lmd_totalirrad', inverter_capacity_mw=None, window_minutes=15, min_irrad=800):
    """
    Checks for Inverter Clipping (Flat Tops) using CONSECUTIVE plateau logic.
    Plots daily power curves for days where power stays near the capacity for an extended continuous period,
    AND irradiance is high (to rule out curtailment or cloudy days).
    
    Args:
        df: DataFrame
        metadata: Dictionary containing station metadata (optional).
        power_col: Power column
        time_col: Datetime column
        irradiance_col: Irradiance column (used to verify it's a sunny day)
        inverter_capacity_mw: Known inverter capacity. If None, estimated from metadata or data.
        window_minutes: Minimum CONSECUTIVE duration of "flat top" to flag as clipping (default 15 mins).
        min_irrad: Minimum max-irradiance (W/m²) for the day to confirm it's a high-irradiance day (default 800).
    """
    print(f"\n--- [Step 2] Quality Check: Inverter Clipping (Flat Top Detection) ---")

    # Extract from metadata if provided
    if metadata and inverter_capacity_mw is None:
        # Metadata 'Capacity' is typically in kW. Convert to MW.
        inverter_capacity_mw = float(metadata.get('Capacity', 0)) / 1000.0

    
    df_clean = df.copy()
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # 1. Estimate Capacity if not provided
    if inverter_capacity_mw is None:
        # Use 99.5th percentile to avoid outliers but catch the peak
        estimated_cap = df_clean[power_col].quantile(0.995)
        print(f"Inverter capacity not provided. Estimating from data (99.5% tile): {estimated_cap:.3f} MW")
        capacity = estimated_cap
    else:
        capacity = inverter_capacity_mw

    # 2. Iterate by Day to find candidates
    df_clean['date'] = df_clean[time_col].dt.date
    daily_groups = df_clean.groupby('date')
    
    clipping_candidates = []
    
    # Infer sample interval from the first chunk of data
    sample_interval = pd.Timedelta(minutes=15) # Default
    if len(df_clean) > 2:
        diffs = df_clean[time_col].diff().median()
        if not pd.isna(diffs):
            sample_interval = diffs
            
    # Gap threshold for "consecutive" (allow 1.5x interval to handle minor jitter)
    gap_threshold = sample_interval * 1.5
    print(f"Inferred sample interval: {sample_interval}. Using {gap_threshold} gap for continuity.")
    
    for date, day_data in daily_groups:
        if day_data.empty: continue
        
        # Check 1: High Irradiance Day? (Ensure clearly sunny)
        if irradiance_col in day_data.columns:
            if day_data[irradiance_col].max() < min_irrad:
                continue
        
        # Check 2: Values near Capacity
        # We look for points >= 99% of Capacity
        near_peak_mask = day_data[power_col] >= 0.99 * capacity
        if not near_peak_mask.any():
            continue
            
        near_peak_points = day_data[near_peak_mask].sort_values(time_col)
        
        # Check 3: Consecutive Duration
        if len(near_peak_points) > 1:
            # Calculate time difference between these selected points
            t_diffs = near_peak_points[time_col].diff()
            
            # A 'break' in the chain is where diff > gap_threshold
            # cumsum() creates unique IDs for each continuous run
            run_ids = (t_diffs > gap_threshold).cumsum()
            
            # Find max duration of any single run
            max_duration = 0
            for _, run_grp in near_peak_points.groupby(run_ids):
                if len(run_grp) > 1:
                    # Duration = Time Span of the run
                    run_dur = (run_grp[time_col].max() - run_grp[time_col].min()).total_seconds() / 60
                    max_duration = max(max_duration, run_dur)
            
            if max_duration >= window_minutes:
                clipping_candidates.append({
                    'date': date,
                    'duration_mins': max_duration,
                    'data': day_data
                })

    print(f"Found {len(clipping_candidates)} potential clipping days (High Power + Flat Top > {window_minutes} mins).")
    
    if not clipping_candidates:
        print("No clipping detected with current thresholds.")
        return

    # Sort by duration (longest flat top first)
    clipping_candidates.sort(key=lambda x: x['duration_mins'], reverse=True)
    
    # 3. Plot Top Candidates (Max 5)
    top_n = min(5, len(clipping_candidates))
    
    fig = go.Figure()
    
    for i in range(top_n):
        cand = clipping_candidates[i]
        day_df = cand['data']
        date_str = str(cand['date'])
        
        fig.add_trace(go.Scatter(
            x=day_df[time_col],
            y=day_df[power_col],
            mode='lines+markers',
            name=f"{date_str} (Dur: {cand['duration_mins']:.0f}m)",
            marker=dict(size=4)
        ))

    # Add Capacity Line
    fig.add_hline(y=capacity, line_dash="dash", line_color="red", annotation_text=f"Capacity ({capacity:.2f} MW)")

    fig.update_layout(
        title=f"Quality Check 2: Inverter Clipping Detection<br>Top {top_n} Days with Flat Tops (> {window_minutes} min)",
        xaxis_title="Time",
        yaxis_title=f"Power ({power_col})",
        template="plotly_white",
        height=600,
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.8)")
    )
    
    fig.show()


    fig.show()


def diagnose_inverter_clipping(df, metadata=None, power_col='power', time_col='date_time', irradiance_col='lmd_totalirrad', capacity_mw=None):
    """
    Diagnostic Visualization for Inverter Clipping.
    Use this when the automated check finds 0 days, to visually inspect why.
    
    Creates a Single Plot:
    1. Scatter of ALL Power data vs Time, colored by Irradiance. Shows if we ever hit capacity.
    
    Args:
        df: DataFrame
        metadata: Dictionary containing station metadata (optional).
        power_col: Power column
        time_col: Datetime column
        irradiance_col: Irradiance column (for coloring)
        capacity_mw: Expected Inverter Capacity (default None, uses metadata or fallback 20 MW) for reference line.
    """
    print(f"\n--- [Step 2.5] Diagnostic: Inverter Clipping Visualization ---")

    if capacity_mw is None:
        if metadata:
            capacity_mw = float(metadata.get('Capacity', 0)) / 1000.0
        else:
            capacity_mw = 20.0 # Default fallback

    
    df_clean = df.copy()
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
    if not pd.api.types.is_datetime64_any_dtype(df_clean[time_col]):
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])

    # --- Full Power History ---
    # Downsample for big datasets if needed? Plotly WebGL handles ~100k ok.
    # We will use Scattergl for performance.
    
    fig = go.Figure()
    
    # Trace 1: Full History
    fig.add_trace(
        go.Scattergl(
            x=df_clean[time_col],
            y=df_clean[power_col],
            mode='markers',
            name='Power History',
            marker=dict(
                size=3,
                color=df_clean[irradiance_col] if irradiance_col in df_clean.columns else 'blue',
                colorscale='Solar',
                showscale=True,
                colorbar=dict(len=0.8, title="Irrad")
            ),
            # Custom Hover
            customdata=df_clean[[irradiance_col]] if irradiance_col in df_clean.columns else None,
            hovertemplate=(
                "<b>Time</b>: %{x}<br>" +
                "<b>Power</b>: %{y:.2f} MW<br>" +
                ("<b>Irrad</b>: %{customdata[0]:.0f} W/m²<br>" if irradiance_col in df_clean.columns else "") +
                "<extra></extra>"
            )
        )
    )
    
    # Add Capacity Line
    fig.add_hline(
        y=0.99 * capacity_mw, 
        line_dash="dash", line_color="red", 
        annotation_text="99% Cap"
    )

    # Formatting
    fig.update_layout(
        title=f"Diagnostic: Inverter Clipping & Capacity Check (Cap={capacity_mw} MW)<br>Full Power History",
        template="plotly_white",
        height=600,
        hovermode="closest",
        xaxis_title="Time",
        yaxis_title="Power (MW)"
    )
    
    fig.show()



def check_negative_power(df, power_col='power', time_col='date_time', fix_action=None):
    """
    Step 3: Detect Negative Power & Polarity Issues.
    Flags negative values which might indicate sensor calibration errors or polarity switches.
    
    Args:
        df: DataFrame
        power_col: Power column
        time_col: Datetime column
        fix_action: Action to take on negative values.
            - None: Report only (default).
            - 'zero': Clamp negative values to 0.
            - 'nan': Set negative values to NaN.
            
    Returns:
        pd.DataFrame: Modified dataframe (if fix_action is set), or original df.
    """
    print(f"\n--- [Step 3] Quality Check: Negative Power (Polarity Check) ---")
    
    df_clean = df.copy()
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
        
    # Check for negative values
    neg_mask = df_clean[power_col] < 0
    neg_count = neg_mask.sum()
    total_count = len(df_clean)
    
    if neg_count == 0:
        print("PASS: No negative power values detected.")
        return df_clean
        
    neg_pct = (neg_count / total_count) * 100
    min_val = df_clean[power_col].min()
    
    print(f"FAIL: Found {neg_count} negative values ({neg_pct:.2f}%).")
    print(f"Min Value: {min_val:.4f} MW")
    
    # Visualization: Plot ONLY the negative values to see pattern
    fig = go.Figure()
    
    neg_data = df_clean[neg_mask]
    
    fig.add_trace(go.Scatter(
        x=neg_data[time_col],
        y=neg_data[power_col],
        mode='markers',
        name='Negative Power',
        marker=dict(color='red', size=5)
    ))
    
    fig.update_layout(
        title=f"Quality Check 3: Negative Power Events (Count={neg_count})",
        xaxis_title="Time",
        yaxis_title="Power (MW)",
        template="plotly_white",
        height=400
    )
    fig.show()
    
    # Fix Action
    if fix_action == 'zero':
        print("Action: Clamping negative values to 0.0")
        df_clean.loc[neg_mask, power_col] = 0.0
        return df_clean
    elif fix_action == 'nan':
        print("Action: Setting negative values to NaN")
        df_clean.loc[neg_mask, power_col] = float('nan')
        return df_clean
    else:
        print("Action: None taken (use fix_action='zero' or 'nan' to correct).")
        return df_clean
        return df_clean


def check_data_shifts(df, power_col='power', time_col='date_time', window_days=30, threshold=0.10):
    """
    Step 4: Check for Data Shifts (Step Changes in Baseline).
    Plots Daily Energy Yield and its rolling mean to detect sudden shifts in performance.
    
    Args:
        df: DataFrame
        power_col: Power column
        time_col: Datetime column
        window_days: Rolling window size for baseline (default 30 days).
        threshold: Percentage change to flag as a shift (default 0.10 = 10%).
    """
    print(f"\n--- [Step 4] Quality Check: Data Shifts (Rolling Mean) ---")
    
    df_clean = df.copy()
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
        
    # 1. Calculate Daily Energy Yield
    # Approximate integration: Sum(Power) * TimeStep. Assuming ~15m data but grouping by day makes it safer.
    # For relative shift detection, Sum(Power) is sufficient (Energy metric).
    df_clean['date'] = df_clean[time_col].dt.date
    daily_yield = df_clean.groupby('date')[power_col].sum()
    
    # 2. Rolling Mean
    rolling_mean = daily_yield.rolling(window=window_days, min_periods=window_days//2).mean()
    
    # 3. Detect Shifts
    # Calculate % change from yesterday's rolling mean to today's rolling mean?
    # Actually, we want to find where the *mean itself* shifts rapidly, or where daily values consistently deviate.
    # Paper suggests: sudden jump in rolling mean.
    pct_change = rolling_mean.pct_change()
    
    # Filter for significant shifts (e.g., > 10% change in the Trend Line in one day is HUGE for a 30-day avg)
    # A 10% jump in a 30-day moving average usually implies a massive discontinuity.
    shifts = pct_change[abs(pct_change) > threshold]
    
    print(f"Daily Yield Analysis: {len(daily_yield)} days.")
    print(f"Shift Detection (Threshold={threshold:.0%}): Found {len(shifts)} potential shifts.")
    
    if len(shifts) > 0:
        print("Potential Step Changes detected on:", shifts.index.tolist())

    # 4. Plot
    fig = go.Figure()
    
    # Scatter: Daily Yield
    fig.add_trace(go.Scatter(
        x=daily_yield.index,
        y=daily_yield.values,
        mode='markers',
        name='Daily Yield',
        marker=dict(size=4, color='gray', opacity=0.5)
    ))
    
    # Line: Rolling Mean
    fig.add_trace(go.Scatter(
        x=rolling_mean.index,
        y=rolling_mean.values,
        mode='lines',
        name=f'{window_days}-Day Rolling Mean',
        line=dict(color='blue', width=2)
    ))
    
    # Vertical Lines for Shifts
    for date in shifts.index:
        fig.add_vline(x=date, line_dash="dash", line_color="red", annotation_text="Shift")

    fig.update_layout(
        title=f"Quality Check 4: Data Shifts & Baseline Consistency<br>(Rolling Window: {window_days} Days, Threshold: {threshold:.0%})",
        xaxis_title="Date",
        yaxis_title="Daily Energy Yield (Arbitrary Units)",
        template="plotly_white",
        height=500
    )
    
    fig.show()


def detect_inverter_failures_pr(df, p_nom=None, pac_col='power', poa_col='poa_global_calc', time_col='date_time', 
                               window_days=30, pr_drop_factor=0.5, min_daily_irrad=2000):
    """
    Step 6: Detect Inverter Failures via Daily Performance Ratio (PR).
    Identifies days where PR drops significantly below the baseline on sunny days.
    
    Args:
        df: DataFrame
        p_nom: Plant Nominal AC Power (MW or kW). Must match unit of `pac_col`.
        pac_col: Power Column (AC Output).
        poa_col: Plane-of-Array Irradiance Column.
        time_col: Datetime column.
        window_days: Rolling median window for baseline PR (default 30).
        pr_drop_factor: Threshold ratio to flag failure (PR < factor * Baseline).
        min_daily_irrad: Min daily irradiation (Wh/m²) to consider valid day (default 2000 = 2 kWh/m²).
        
    Returns:
        pd.DataFrame: Daily summary with columns [E_AC, H_POA, PR, Baseline, Flag].
    """
    print(f"\n--- [Step 6] Quality Check: Inverter Failures (Daily PR) ---")
    
    if p_nom is None:
        print("Error: `p_nom` (System Capacity) is required to calculate PR.")
        return None

    df_clean = df.copy()
    if time_col not in df_clean.columns and isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean[time_col] = df_clean.index
    
    # 1. Compute Timestep (dt in hours)
    # Robust method: Calculate per-row diff, forward fill last
    df_clean = df_clean.sort_values(time_col)
    df_clean['dt_h'] = df_clean[time_col].diff().shift(-1).dt.total_seconds() / 3600
    
    # Fill last row & Clip to valid range (e.g., max 1 hour) to handle gaps
    median_dt = df_clean['dt_h'].median()
    df_clean['dt_h'] = df_clean['dt_h'].fillna(median_dt).clip(0, 1.5)
    
    # 2. Daily Integration
    df_clean['date'] = df_clean[time_col].dt.date
    
    # E_AC (Wh) if Power is W; (MWh) if Power is MW
    daily_stats = df_clean.groupby('date').apply(
        lambda x: pd.Series({
            'E_AC': (x[pac_col] * x['dt_h']).sum(),      # Daily Energy
            'H_POA': (x[poa_col] * x['dt_h']).sum()      # Daily Irradiation (Wh/m²)
        })
    )
    
    # 3. Compute Daily PR
    # PR = (E_AC / P_nom) / (H_POA / 1000)
    # Ensure H_POA > 0 to avoid div/0
    daily_stats['PR'] = (daily_stats['E_AC'] / p_nom) / (daily_stats['H_POA'] / 1000)
    
    # Clean up infinite/nan PRs (e.g. night-only data)
    daily_stats.loc[daily_stats['H_POA'] < 10, 'PR'] = float('nan')
    
    # 4. Baseline Calculation (Rolling Median)
    # Shift by 1 to exclude current day from its own baseline
    daily_stats['PR_Baseline'] = daily_stats['PR'].rolling(window=window_days, min_periods=5).median().shift(1)
    
    # 5. Flag Failures
    # Conditions: Sufficient Sun AND PR < Threshold * Baseline
    daily_stats['Irrad_OK'] = daily_stats['H_POA'] >= min_daily_irrad
    daily_stats['Failure_Flag'] = (
        daily_stats['Irrad_OK'] & 
        (daily_stats['PR'] < pr_drop_factor * daily_stats['PR_Baseline'])
    )
    
    num_flagged = daily_stats['Failure_Flag'].sum()
    print(f"Analyzed {len(daily_stats)} days.")
    print(f"Flagged {num_flagged} potential inverter failures (PR < {pr_drop_factor:.0%} of Baseline).")
    
    if num_flagged > 0:
        print("Top 5 Flagged Days:")
        print(daily_stats[daily_stats['Failure_Flag']].head(5)[['E_AC', 'H_POA', 'PR', 'PR_Baseline']])

    # 6. Plot
    fig = go.Figure()
    
    # Scatter: Daily PR
    fig.add_trace(go.Scatter(
        x=daily_stats.index,
        y=daily_stats['PR'],
        mode='markers',
        name='Daily PR',
        marker=dict(size=4, color='blue', opacity=0.6)
    ))
    
    # Line: Baseline
    fig.add_trace(go.Scatter(
        x=daily_stats.index,
        y=daily_stats['PR_Baseline'],
        mode='lines',
        name='Baseline PR (30d Median)',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Markers: Flags
    failures = daily_stats[daily_stats['Failure_Flag']]
    if not failures.empty:
        fig.add_trace(go.Scatter(
            x=failures.index,
            y=failures['PR'],
            mode='markers',
            name='Potential Failures',
            marker=dict(size=8, color='red', symbol='x')
        ))

    fig.update_layout(
        title=f"Quality Check 6: Inverter Failure Detection via PR<br>(Threshold: <{pr_drop_factor:.0%} Baseline, Min Irrad: {min_daily_irrad} Wh/m²)",
        xaxis_title="Date",
        yaxis_title="Performance Ratio (PR)",
        template="plotly_white",
        height=500,
        yaxis=dict(range=[0, 1.1]) # PR usually 0.7-0.9
    )
    
    fig.show()
    
    return daily_stats

