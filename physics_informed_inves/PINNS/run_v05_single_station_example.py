
"""
Example script to run V05 Single Station Benchmark.
This script benchmarks models using ONLY Station 07 data (plus K_CS), 
ignoring all spatial features (Station 08, Deltas).
"""

import sys
import os
import pandas as pd

# 1. Import from the SINGLE STATION module
from physics_residual_mamba_SINGLE_v05 import run_single_station_benchmark, SingleMambaConfigs

# 2. Setup Configuration
my_config = SingleMambaConfigs(n_splits=4)
my_config.warmup_epochs = 15 
my_config.epochs = 30 
# Note: config.PAST_INPUT_COLS automatically excludes 's08_' and 'spatial_' in the module logic,
# but using SingleMambaConfigs ensures they aren't even initialized.

# 3. Execution
if 'merged_df' in locals():
    print("Starting V05 Single Station Benchmark...")
    results_single = run_single_station_benchmark(
        merged_df, 
        n_splits=4, 
        custom_config=my_config
    )
    print("Done. Interactive plots saved as HTML files.")
else:
    print("Please load 'merged_df' before running this snippet.")
