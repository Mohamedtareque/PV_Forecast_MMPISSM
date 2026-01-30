import torch
import sys
import os

# --- CONFIGURATION ---
# Note: This script analyzes .pth files directly without needing to import model classes.
# It works by examining the state_dict (saved weights) structure.

# List the paths to your saved .pth files
MODEL_PATHS = [
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Base_Power_Mamba_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Mamba_Physics_V05_(Hyper-Robust)_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Physics_V04_(Robust)_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Power_Mamba_Physics_V05_(Hyper-Robust)_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Power_Mamba_Physics_V05_+_Advection_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Pure_Mamba_Physics_V05_(Hyper-Robust)_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/single_Base_Power_Mamba_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/single_Pure_Mamba_Physics_V05_(Hyper-Robust)_fold4.pth",
    "/home/muhammadhassan/App_v02/physics_informed_inves/PINNS/saved_models/Vanilla_Mamba_fold4.pth"
]
# ---------------------

def identify_model_type(filename):
    """
    Identify which model class to use based on filename.
    Returns a tuple: (ModelClass, description)
    """
    basename = os.path.basename(filename).lower()
    
    if 'vanilla_mamba' in basename:
        return ('VanillaMamba', 'Vanilla Mamba (Baseline)')
    elif 'pure_mamba_physics' in basename:
        return ('PureMambaPhysics', 'Pure Mamba Physics')
    elif 'power_mamba_physics' in basename and 'advection' in basename:
        return ('PowerMambaPhysicsAdvection', 'Power Mamba Physics V05 + Advection')
    elif 'power_mamba_physics' in basename:
        return ('PowerMambaPhysics', 'Power Mamba Physics')
    elif 'base_power_mamba' in basename:
        return ('PowerMamba', 'Base Power Mamba')
    elif 'mamba_physics' in basename:
        return ('MambaPhysics', 'Mamba Physics')
    elif 'physics_v04' in basename:
        return ('PhysicsV04', 'Physics V04 (Robust)')
    else:
        return ('Unknown', 'Unknown Model Type')

def count_parameters(model_path):
    """
    Count total and physics parameters for a given model checkpoint.
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {os.path.basename(model_path)}")
    print(f"{'='*80}")

    # A. Load the state dictionary (weights)
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"ℹ️  Checkpoint contains additional info (epoch, optimizer, etc.)")
        elif isinstance(checkpoint, dict) and any(k.startswith('module.') for k in checkpoint.keys()):
            # Handle DataParallel saved models
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            print(f"ℹ️  Detected DataParallel model, removing 'module.' prefix")
        else:
            state_dict = checkpoint
            
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None

    # B. Identify model type
    model_type, model_desc = identify_model_type(model_path)
    print(f"📊 Model Type: {model_desc}")
    
    # C. Count parameters directly from state_dict
    total_params = sum(p.numel() for p in state_dict.values())
    
    print(f"\n{'─'*80}")
    print(f"📈 Total Parameters: {total_params:,}")
    
    # D. Identify Physics Parameters
    print(f"\n{'─'*80}")
    print(f"🔬 Physics Parameter Breakdown")
    print(f"{'─'*80}")
    print(f"{'Parameter Name':<60} {'Shape':<20} {'Count':>10}")
    print(f"{'─'*80}")
    
    physics_count = 0
    backbone_count = 0
    physics_params = []
    
    # Keywords that typically indicate physics parameters
    physics_keywords = [
        'physics', 'thermal', 'efficiency', 'inverter', 'advection',
        'u0', 'u1', 'gamma', 'eta', 'inv_k', 'coeff', 'alpha', 'beta',
        'fluid_gate', 'relational'
    ]
    
    for name, param in state_dict.items():
        param_count = param.numel()
        
        # Check if this is likely a physics parameter
        is_physics = False
        
        # Method 1: Check naming convention
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in physics_keywords):
            is_physics = True
        
        # Method 2: Check if it's a small tensor (scalars or small vectors)
        # Physics params are typically <= 10 elements
        if param_count <= 10 and not ('norm' in name_lower or 'bias' in name_lower):
            is_physics = True
            
        if is_physics:
            physics_count += param_count
            physics_params.append((name, param.shape, param_count))
            marker = "🔬"
        else:
            backbone_count += param_count
            marker = "  "
            
    # Print physics parameters first
    if physics_params:
        print(f"\n{' PHYSICS PARAMETERS ':-^80}")
        for name, shape, count in physics_params:
            print(f"🔬 {name:<57} {str(shape):<20} {count:>10,}")
    else:
        print(f"\n⚠️  No physics parameters detected (this might be a baseline model)")
    
    # Summary
    print(f"\n{'─'*80}")
    print(f"📊 SUMMARY FOR LATEX TABLE")
    print(f"{'─'*80}")
    print(f"Model Name: {model_desc}")
    print(f"Total Parameters: {total_params:,}")
    print(f"  └─ Backbone Parameters: {backbone_count:,}")
    print(f"  └─ Physics Parameters: {physics_count:,}")
    print(f"\n💡 For LaTeX table:")
    if physics_count > 0:
        print(f"   Total Params Column: {total_params/1e6:.2f}M")
        print(f"   Learnable Physics Column: {physics_count}")
    else:
        print(f"   Total Params Column: {total_params/1e6:.2f}M")
        print(f"   Learnable Physics Column: 0 (Baseline/No Physics)")
    
    return total_params, physics_count

def generate_summary_table(results):
    """
    Generate a summary table of all models.
    """
    print(f"\n\n{'='*80}")
    print(f"{'SUMMARY TABLE - ALL MODELS':^80}")
    print(f"{'='*80}")
    print(f"{'Model':<45} {'Total (M)':<15} {'Physics':<15}")
    print(f"{'─'*80}")
    
    for model_path, (total, physics) in results.items():
        if total is not None:
            model_name = os.path.basename(model_path).replace('_fold4.pth', '')
            print(f"{model_name:<45} {total/1e6:>8.2f}M      {physics:>10,}")
    
    print(f"{'='*80}\n")

# # Main execution
# if __name__ == "__main__":
#     results = {}
    
#     for path in MODEL_PATHS:
#         if os.path.exists(path):
#             total, physics = count_parameters(path)
#             results[path] = (total, physics)
#         else:
#             print(f"\n❌ File not found: {path}")
#             results[path] = (None, None)
    
#     # Generate summary table
#     generate_summary_table(results)
    
#     print("\n✅ Analysis complete!")
#     print("\n💡 Tips for your thesis table:")
#     print("   1. Use the 'Total (M)' values for the 'Total Params' column")
#     print("   2. Use the 'Physics' values for the 'Learnable Physics' column")
#     print("   3. Models with 0 physics params are baseline/vanilla models")
#     print("   4. Review the detailed breakdown above to verify physics parameter detection")
