
import sys
from unittest.mock import MagicMock

# Mock plotly to avoid import error if not installed
sys.modules['plotly'] = MagicMock()
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['plotly.subplots'] = MagicMock()

import pandas as pd
import numpy as np
from preprocessing_data import apply_horizon_mask

def test_shading_qc():
    print("Creating synthetic data...")
    # Create 1000 points
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
    df = pd.DataFrame({'date_time': dates})
    
    # Defaults: High Power, High POA, High Elevation
    df['power'] = 10.0 # Capacity is approx 10
    df['poa_global_calc'] = 800.0
    df['elevation'] = 60.0
    df['azimuth'] = 180.0
    
    # ----------------------------------------------------
    # Case 1: Normal Operation (High Sun)
    # Rows 0-100: Already set above.
    
    # ----------------------------------------------------
    # Case 2: Shading Candidate
    # Low Power, High POA, Low Elevation, "Bad" Azimuth
    # We need enough of these to trigger the biological threshold detection!
    # The function needs >10 anomalies to even run.
    # We'll create a cluster of 50 points.
    
    shading_indices = range(100, 200)
    df.loc[shading_indices, 'power'] = 0.0 # Mismatch!
    df.loc[shading_indices, 'poa_global_calc'] = 600.0 # High POA
    df.loc[shading_indices, 'elevation'] = 20.0 # Low El (should be < derived threshold)
    
    # We need these to cluster in specific azimuths to trigger the bad sector logic.
    # Let's put them all at Azimuth 100-110.
    df.loc[shading_indices, 'azimuth'] = np.random.uniform(100, 110, len(shading_indices))
    
    # ----------------------------------------------------
    # Case 3: Random Inverter Failure (Not Shading)
    # Low Power, High POA, High Elevation
    # This matches "Mismatch" but fails "Geometric Consistency"
    
    failure_indices = range(300, 320)
    df.loc[failure_indices, 'power'] = 0.0
    df.loc[failure_indices, 'poa_global_calc'] = 800.0
    df.loc[failure_indices, 'elevation'] = 70.0 # Too high for shading
    df.loc[failure_indices, 'azimuth'] = 180.0
    
    # ----------------------------------------------------
    # Run Function
    print("Running apply_horizon_mask...")
    df_result = apply_horizon_mask(df, lat=0, lon=0) # Lat/Lon ignored if columns exist
    
    # ----------------------------------------------------
    # Verify
    print("\nVerifying Results:")
    
    # Check 1: Normal Data -> False
    normal_falgs = df_result.loc[0:50, 'is_shading_candidate'].sum()
    print(f"Normal Data Flagged: {normal_falgs} (Expected 0)")
    
    # Check 2: Shading Candidates -> True
    # Note: Variable threshold means we might miss some if 20deg isn't low enough?
    # But 90th percentile of anomalies (20deg) will be 20deg. So 20 < 20 is False. 
    # Wait, percentile logic: min(percentile, 45).
    # If all anomalies are at 20, 90th percentile is 20. mask is < 20. 
    # So exact 20 might be excluded. Let's make anomalies vary 10-20, setup threshold, then check 15.
    # Actually, let's look at the result.
    
    shading_flags = df_result.loc[shading_indices, 'is_shading_candidate'].sum()
    print(f"Shading Data Flagged: {shading_flags}/{len(shading_indices)}")
    
    # Check 3: High Elevation Failures -> False
    failure_flags = df_result.loc[failure_indices, 'is_shading_candidate'].sum()
    print(f"Random Failure Flagged: {failure_flags} (Expected 0)")
    
    # Check Column Existence
    if 'is_shading_candidate' in df_result.columns:
        print("SUCCESS: Column 'is_shading_candidate' exists.")
    else:
        print("FAILURE: Column missing.")

    # Check that rows were NOT deleted
    if len(df_result) == 1000:
         print("SUCCESS: Row count preserved (1000).")
    else:
         print(f"FAILURE: Rows deleted. Count: {len(df_result)}")

if __name__ == "__main__":
    test_shading_qc()
