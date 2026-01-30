
"""
Example script to run V05 Benchmark (No Wavelet)
Assumes 'merged_df' is already prepared with Station 07 + Station 08 + Deltas + K_CS.
"""

import sys
import os
import pandas as pd

# 1. Import from the NEW V05 file
from physics_residual_mamba_spatial_v05 import run_comparative_benchmark, SpatialMambaConfigs

# 2. Setup Configuration
# Note: V05 configs are compatible with V02/V04 data structures
my_spatial_config = SpatialMambaConfigs(n_splits=4)
my_spatial_config.warmup_epochs = 15 
my_spatial_config.epochs = 30

# 3. Ensure 'merged_df' exists
# If you are running this in a script, you need to load your data here.
# If running in a notebook, just pass your existing merged_df.

if 'merged_df' in locals():
    # 4. Run Benchmark
    print("Starting V05 Benchmark...")
    results_spatial = run_comparative_benchmark(
        merged_df, 
        n_splits=4, 
        custom_config=my_spatial_config
    )
    
    # 5. Results are printed to stdout and returned in 'results_spatial'
    print("Done.")
else:
    print("Please load 'merged_df' before running this snippet.")
