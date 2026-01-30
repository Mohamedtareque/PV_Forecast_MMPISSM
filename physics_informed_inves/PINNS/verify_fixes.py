
import sys
import os
import torch
import pandas as pd

# Add path to sys to import modules
sys.path.append('/home/muhammadhassan/App_v02/physics_informed_inves/PINNS')

try:
    from v05_experiment_runner import SpatialMambaConfigs as ConfigV05
    from physics_residual_mamba_spatial_v05 import run_comparative_benchmark as run_v05
    from physics_residual_mamba_spatial_v02 import SpatialMambaConfigs as ConfigV02
    from physics_residual_mamba_spatial_v02 import MMPISSM_Model_01
    
    print("Imports successful.")
    
    # Check V05 Config
    c5 = ConfigV05()
    print(f"V05 seq_len: {c5.seq_len}")
    if 'K_CS' in c5.FUTURE_INPUT_COLS:
        print("FAIL: K_CS found in V05 FUTURE_INPUT_COLS (Leakage!)")
    else:
        print("PASS: K_CS not in V05 FUTURE_INPUT_COLS")
        
    if 's08_K_PV' in c5.FUTURE_INPUT_COLS:
        print("FAIL: s08_K_PV found in V05 FUTURE_INPUT_COLS (Leakage!)")
    else:
        print("PASS: s08_K_PV not in V05 FUTURE_INPUT_COLS")

    # Check V02 Typo
    try:
        model = MMPISSM_Model_01(ConfigV02())
        if hasattr(model, 'decomposition'):
            print("PASS: MMPISSM_Model_01 has 'decomposition' attribute.")
        else:
            print("FAIL: MMPISSM_Model_01 missing 'decomposition'.")
            
        if hasattr(model, 'decompsition'):
             print("WARN: MMPISSM_Model_01 still has 'decompsition' (did you forget to remove old one? or just renamed?)")
    except Exception as e:
        print(f"Error checking V02 model: {e}")

except Exception as e:
    print(f"Import/Runtime Error: {e}")
