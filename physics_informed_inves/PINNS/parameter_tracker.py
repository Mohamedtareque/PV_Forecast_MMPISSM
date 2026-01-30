"""
Parameter Tracking Module for Physics-Informed Mamba Models
============================================================

This module provides infrastructure for tracking learnable physics parameters
and residual network weights during training, with interactive Plotly visualization.

Author: Auto-generated
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Plotly imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Physical Reference Ranges
# =============================================================================

PHYSICAL_RANGES = {
    'U0': {'min': 20.0, 'max': 30.0, 'unit': 'W/m²K', 'desc': 'Heat transfer coeff (no wind)'},
    'U1': {'min': 1.0, 'max': 3.0, 'unit': 'W·s/m³K', 'desc': 'Wind heat transfer coeff'},
    'eta': {'min': 0.15, 'max': 0.20, 'unit': '', 'desc': 'Module efficiency'},
    'gamma': {'min': -0.005, 'max': -0.002, 'unit': '1/K', 'desc': 'Temperature coefficient'},
    'P_stc': {'min': 18.0, 'max': 20.0, 'unit': 'MW', 'desc': 'Power at STC'},
    'b0': {'min': 0.0, 'max': 0.1, 'unit': '', 'desc': 'IAM coefficient'},
    'inv_k': {'min': 5.0, 'max': 15.0, 'unit': '', 'desc': 'Inverter steepness'},
    'inv_thresh': {'min': 0.05, 'max': 0.15, 'unit': 'MW', 'desc': 'Inverter threshold'},
}

# Color palette for folds
FOLD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

# =============================================================================
# Activation Functions (matching model implementations)
# =============================================================================

def softplus(x):
    """Softplus activation: log(1 + exp(x))"""
    return np.log1p(np.exp(np.clip(x, -20, 20)))

def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

def apply_physics_activation(param_name: str, raw_value: float, P_stc_limit: float = 19.0) -> float:
    """Apply appropriate activation function based on parameter name."""
    if 'U0' in param_name or 'U1' in param_name:
        return softplus(raw_value)
    elif 'eta' in param_name:
        return sigmoid(raw_value)
    elif 'gamma' in param_name:
        return -softplus(raw_value)
    elif 'Pstc' in param_name:
        return P_stc_limit * sigmoid(raw_value)
    elif 'b0' in param_name:
        return sigmoid(raw_value) * 0.2
    elif 'inv_k' in param_name:
        return softplus(raw_value) * 10.0
    elif 'inv_thresh' in param_name:
        return softplus(raw_value)
    elif 'diffuse_frac' in param_name:
        return sigmoid(raw_value)
    elif 'spec_coeff' in param_name:
        return raw_value  # No activation
    else:
        return raw_value

# =============================================================================
# Parameter Tracker Class
# =============================================================================

class ParameterTracker:
    """
    Tracks learnable parameters across epochs and folds.
    
    Usage:
        tracker = ParameterTracker()
        for fold in folds:
            for epoch in epochs:
                tracker.record_epoch(model, fold, epoch)
        tracker.generate_plots(output_dir)
    """
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
        self.physics_params = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.residual_norms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.gradient_norms = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.P_stc_limit = 19.0
        
    def record_epoch(self, model: torch.nn.Module, fold: int, epoch: int):
        """Record all parameters after an epoch."""
        self._extract_physics_params(model, fold, epoch)
        self._extract_residual_norms(model, fold, epoch)
        self._extract_gradient_norms(model, fold, epoch)
        
    def _extract_physics_params(self, model: torch.nn.Module, fold: int, epoch: int):
        """Extract physics layer parameters."""
        if not hasattr(model, 'physics_layer'):
            return
            
        physics_layer = model.physics_layer
        
        # Get P_stc_limit if available
        if hasattr(physics_layer, 'P_stc_limit'):
            self.P_stc_limit = float(physics_layer.P_stc_limit)
        
        for name, param in physics_layer.named_parameters():
            if param.requires_grad:
                raw_value = param.detach().cpu().numpy().flatten()[0]
                activated_value = apply_physics_activation(name, raw_value, self.P_stc_limit)
                
                # Store both raw and activated
                self.physics_params[fold][f'{name}_raw'][epoch].append(raw_value)
                self.physics_params[fold][f'{name}_activated'][epoch].append(activated_value)
                
                # Map to physical parameter name
                phys_name = self._map_to_physical_name(name)
                if phys_name:
                    self.physics_params[fold][phys_name][epoch].append(activated_value)
    
    def _map_to_physical_name(self, param_name: str) -> Optional[str]:
        """Map raw parameter name to physical parameter name."""
        mappings = {
            'U0_raw': 'U0', 'U0_base_raw': 'U0',
            'U1_raw': 'U1', 'U_wind_raw': 'U1',
            'eta_raw': 'eta',
            'gamma_raw': 'gamma',
            'Pstc_raw': 'P_stc',
            'b0_raw': 'b0',
            'inv_k_raw': 'inv_k',
            'inv_thresh_raw': 'inv_thresh',
        }
        return mappings.get(param_name)
    
    def _extract_residual_norms(self, model: torch.nn.Module, fold: int, epoch: int):
        """Extract weight norms from residual network (Mamba backbone)."""
        if not hasattr(model, 'mamba_model'):
            return
            
        for name, param in model.mamba_model.named_parameters():
            if 'weight' in name and param.requires_grad:
                weight_norm = torch.norm(param.detach()).cpu().numpy()
                self.residual_norms[fold][name][epoch].append(float(weight_norm))
    
    def _extract_gradient_norms(self, model: torch.nn.Module, fold: int, epoch: int):
        """Extract gradient norms for all parameters."""
        # Physics layer gradients
        if hasattr(model, 'physics_layer'):
            for name, param in model.physics_layer.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.detach()).cpu().numpy()
                    self.gradient_norms[fold][f'physics_{name}'][epoch].append(float(grad_norm))
        
        # Mamba backbone gradients
        if hasattr(model, 'mamba_model'):
            for name, param in model.mamba_model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad.detach()).cpu().numpy()
                    self.gradient_norms[fold][f'mamba_{name}'][epoch].append(float(grad_norm))
    
    def get_physics_summary(self) -> Dict:
        """Get summary of physics parameters across folds."""
        summary = {}
        for fold, params in self.physics_params.items():
            for param_name, epochs in params.items():
                if param_name not in summary:
                    summary[param_name] = {}
                # Get final epoch value
                if epochs:
                    max_epoch = max(epochs.keys())
                    summary[param_name][fold] = epochs[max_epoch][-1] if epochs[max_epoch] else None
        return summary
    
    def generate_all_plots(self, output_dir: str):
        """Generate all visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate physics convergence plot
        self.plot_physics_convergence(os.path.join(output_dir, f'{self.model_name.replace(" ", "_")}_physics_convergence.html'))
        
        # Generate gradient norms plot
        self.plot_gradient_norms(os.path.join(output_dir, f'{self.model_name.replace(" ", "_")}_gradient_norms.html'))
        
        # Generate weight norms plot
        self.plot_weight_norms(os.path.join(output_dir, f'{self.model_name.replace(" ", "_")}_weight_norms.html'))
        
        # Generate physical plausibility heatmap
        self.plot_plausibility_heatmap(os.path.join(output_dir, f'{self.model_name.replace(" ", "_")}_plausibility.html'))
        
        print(f"[ParameterTracker] Generated all plots for {self.model_name} in {output_dir}")
    
    def plot_physics_convergence(self, save_path: str):
        """Generate interactive physics parameter convergence plot."""
        if not self.physics_params:
            print(f"[ParameterTracker] No physics params to plot for {self.model_name}")
            return
        
        # Get all physical parameter names (activated, not raw)
        phys_param_names = set()
        for fold, params in self.physics_params.items():
            for name in params.keys():
                if '_activated' in name or name in PHYSICAL_RANGES:
                    phys_param_names.add(name.replace('_activated', ''))
        
        phys_param_names = sorted(phys_param_names)
        n_params = len(phys_param_names)
        
        if n_params == 0:
            return
        
        # Create subplots
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=[f'{name}' for name in phys_param_names],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        for idx, param_name in enumerate(phys_param_names):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            # Add traces for each fold
            for fold in sorted(self.physics_params.keys()):
                # Try activated first, then raw
                key = f'{param_name}_activated' if f'{param_name}_activated' in self.physics_params[fold] else param_name
                
                if key not in self.physics_params[fold]:
                    continue
                    
                epochs_data = self.physics_params[fold][key]
                epochs = sorted(epochs_data.keys())
                values = [epochs_data[e][-1] if epochs_data[e] else np.nan for e in epochs]
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode='lines+markers',
                        name=f'Fold {fold+1}',
                        line=dict(color=FOLD_COLORS[fold % len(FOLD_COLORS)], width=2),
                        marker=dict(size=4),
                        legendgroup=f'fold{fold}',
                        showlegend=(idx == 0),
                        hovertemplate=f'Fold {fold+1}<br>Epoch: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>'
                    ),
                    row=row, col=col
                )
            
            # Add physical reference range as shaded region
            phys_key = param_name.replace('_raw', '')
            if phys_key in PHYSICAL_RANGES:
                ref = PHYSICAL_RANGES[phys_key]
                max_epoch = max(max(self.physics_params[f][key].keys()) for f in self.physics_params if key in self.physics_params[f])
                
                fig.add_hrect(
                    y0=ref['min'], y1=ref['max'],
                    fillcolor='green', opacity=0.15,
                    line_width=0,
                    row=row, col=col
                )
                
                # Add reference lines
                fig.add_hline(y=ref['min'], line_dash='dash', line_color='green', opacity=0.5, row=row, col=col)
                fig.add_hline(y=ref['max'], line_dash='dash', line_color='green', opacity=0.5, row=row, col=col)
        
        fig.update_layout(
            title=dict(text=f'{self.model_name} - Physics Parameter Convergence', font=dict(size=18)),
            height=300 * n_rows,
            width=1200,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        # Add range slider to first x-axis
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05), row=n_rows, col=1)
        
        fig.write_html(save_path)
        print(f"[ParameterTracker] Physics convergence plot saved to {save_path}")
    
    def plot_gradient_norms(self, save_path: str):
        """Generate gradient norm evolution plot."""
        if not self.gradient_norms:
            print(f"[ParameterTracker] No gradient data to plot for {self.model_name}")
            return
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Physics Layer Gradients', 'Mamba Backbone Gradients'],
            horizontal_spacing=0.1
        )
        
        # Physics gradients
        for fold in sorted(self.gradient_norms.keys()):
            physics_names = [n for n in self.gradient_norms[fold].keys() if n.startswith('physics_')]
            
            for name in physics_names:
                epochs_data = self.gradient_norms[fold][name]
                epochs = sorted(epochs_data.keys())
                values = [epochs_data[e][-1] if epochs_data[e] else np.nan for e in epochs]
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode='lines',
                        name=f'{name.replace("physics_", "")} (F{fold+1})',
                        line=dict(width=1.5),
                        opacity=0.8,
                        hovertemplate=f'{name}<br>Epoch: %{{x}}<br>Grad Norm: %{{y:.4e}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Mamba gradients (aggregate by layer type)
        for fold in sorted(self.gradient_norms.keys()):
            mamba_names = [n for n in self.gradient_norms[fold].keys() if n.startswith('mamba_')]
            
            # Group by layer type
            layer_groups = defaultdict(list)
            for name in mamba_names:
                layer_type = name.split('.')[-1] if '.' in name else name
                layer_groups[layer_type].append(name)
            
            for layer_type in ['lin1.weight', 'lin2.weight', 'lin3.weight']:
                matching = [n for n in mamba_names if layer_type in n]
                if matching:
                    name = matching[0]
                    epochs_data = self.gradient_norms[fold][name]
                    epochs = sorted(epochs_data.keys())
                    values = [epochs_data[e][-1] if epochs_data[e] else np.nan for e in epochs]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=epochs,
                            y=values,
                            mode='lines',
                            name=f'{layer_type} (F{fold+1})',
                            line=dict(width=1.5),
                            opacity=0.8
                        ),
                        row=1, col=2
                    )
        
        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_yaxes(type='log', row=1, col=2)
        
        fig.update_layout(
            title=dict(text=f'{self.model_name} - Gradient Norm Evolution', font=dict(size=18)),
            height=500,
            width=1200,
            hovermode='x unified'
        )
        
        fig.write_html(save_path)
        print(f"[ParameterTracker] Gradient norms plot saved to {save_path}")
    
    def plot_weight_norms(self, save_path: str):
        """Generate weight norm evolution plot for residual network."""
        if not self.residual_norms:
            print(f"[ParameterTracker] No weight norm data to plot for {self.model_name}")
            return
        
        fig = go.Figure()
        
        # Get unique layer names across all folds
        all_layers = set()
        for fold, layers in self.residual_norms.items():
            all_layers.update(layers.keys())
        
        # Filter to key layers
        key_layers = [l for l in all_layers if any(k in l for k in ['lin1', 'lin2', 'lin3', 'mamba'])]
        
        for fold in sorted(self.residual_norms.keys()):
            for layer_name in key_layers:
                if layer_name not in self.residual_norms[fold]:
                    continue
                    
                epochs_data = self.residual_norms[fold][layer_name]
                epochs = sorted(epochs_data.keys())
                values = [epochs_data[e][-1] if epochs_data[e] else np.nan for e in epochs]
                
                short_name = layer_name.split('.')[-2] + '.' + layer_name.split('.')[-1] if '.' in layer_name else layer_name
                
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode='lines',
                        name=f'{short_name} (F{fold+1})',
                        line=dict(width=1.5),
                        opacity=0.8,
                        hovertemplate=f'{short_name}<br>Epoch: %{{x}}<br>Weight Norm: %{{y:.2f}}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=dict(text=f'{self.model_name} - Weight Norm Evolution', font=dict(size=18)),
            xaxis_title='Epoch',
            yaxis_title='L2 Norm',
            height=500,
            width=1000,
            hovermode='x unified',
            legend=dict(orientation='v', yanchor='top', y=1, xanchor='left', x=1.02)
        )
        
        fig.write_html(save_path)
        print(f"[ParameterTracker] Weight norms plot saved to {save_path}")
    
    def plot_plausibility_heatmap(self, save_path: str):
        """Generate physical plausibility heatmap."""
        if not self.physics_params:
            return
        
        # Collect final values for each parameter across folds
        param_deviations = {}
        
        for phys_name, ref in PHYSICAL_RANGES.items():
            deviations = []
            for fold in sorted(self.physics_params.keys()):
                key = f'{phys_name}_activated' if f'{phys_name}_activated' in self.physics_params[fold] else phys_name
                
                if key not in self.physics_params[fold]:
                    # Try mapping from raw names
                    for raw_name, mapped in [('U0_raw', 'U0'), ('eta_raw', 'eta'), ('gamma_raw', 'gamma')]:
                        if mapped == phys_name and f'{raw_name}_activated' in self.physics_params[fold]:
                            key = f'{raw_name}_activated'
                            break
                
                if key in self.physics_params[fold]:
                    epochs_data = self.physics_params[fold][key]
                    if epochs_data:
                        max_epoch = max(epochs_data.keys())
                        final_val = epochs_data[max_epoch][-1] if epochs_data[max_epoch] else np.nan
                        
                        # Calculate deviation from range (0 = in range, positive = outside)
                        if final_val < ref['min']:
                            dev = (ref['min'] - final_val) / (ref['max'] - ref['min'])
                        elif final_val > ref['max']:
                            dev = (final_val - ref['max']) / (ref['max'] - ref['min'])
                        else:
                            dev = 0
                        deviations.append(dev)
                    else:
                        deviations.append(np.nan)
                else:
                    deviations.append(np.nan)
            
            if any(not np.isnan(d) for d in deviations):
                param_deviations[phys_name] = deviations
        
        if not param_deviations:
            return
        
        params = list(param_deviations.keys())
        folds = [f'Fold {i+1}' for i in range(len(list(param_deviations.values())[0]))]
        z_data = [param_deviations[p] for p in params]
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=folds,
            y=params,
            colorscale=[
                [0, 'green'],
                [0.3, 'yellow'],
                [0.7, 'orange'],
                [1, 'red']
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title='Deviation', tickvals=[0, 0.5, 1], ticktext=['In Range', '50%', '100%+']),
            hovertemplate='%{y}<br>%{x}<br>Deviation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'{self.model_name} - Physical Plausibility', font=dict(size=18)),
            xaxis_title='Fold',
            yaxis_title='Parameter',
            height=400,
            width=600
        )
        
        fig.write_html(save_path)
        print(f"[ParameterTracker] Plausibility heatmap saved to {save_path}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracker(model_name: str) -> ParameterTracker:
    """Create a new parameter tracker instance."""
    return ParameterTracker(model_name)

def generate_combined_comparison(trackers: Dict[str, ParameterTracker], output_dir: str):
    """Generate combined comparison plots across multiple models."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combined physics parameter comparison
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=['U0', 'eta', 'gamma', 'P_stc', 'b0', 'inv_k'],
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )
    
    param_positions = {'U0': (1, 1), 'eta': (1, 2), 'gamma': (1, 3), 
                       'P_stc': (2, 1), 'b0': (2, 2), 'inv_k': (2, 3)}
    
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for model_idx, (model_name, tracker) in enumerate(trackers.items()):
        color = model_colors[model_idx % len(model_colors)]
        
        for param_name, (row, col) in param_positions.items():
            # Aggregate across folds
            all_epochs = set()
            fold_values = {}
            
            for fold, params in tracker.physics_params.items():
                key = f'{param_name}_activated' if f'{param_name}_activated' in params else param_name
                if key in params:
                    fold_values[fold] = params[key]
                    all_epochs.update(params[key].keys())
            
            if not fold_values:
                continue
            
            # Average across folds
            epochs = sorted(all_epochs)
            avg_values = []
            for e in epochs:
                vals = [fold_values[f][e][-1] for f in fold_values if e in fold_values[f] and fold_values[f][e]]
                avg_values.append(np.mean(vals) if vals else np.nan)
            
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=avg_values,
                    mode='lines',
                    name=model_name,
                    line=dict(color=color, width=2),
                    legendgroup=model_name,
                    showlegend=(row == 1 and col == 1),
                    hovertemplate=f'{model_name}<br>Epoch: %{{x}}<br>{param_name}: %{{y:.4f}}<extra></extra>'
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title=dict(text='Model Comparison - Physics Parameters (Fold Average)', font=dict(size=18)),
        height=600,
        width=1200,
        hovermode='x unified'
    )
    
    fig.write_html(os.path.join(output_dir, 'model_comparison_physics.html'))
    print(f"[ParameterTracker] Combined comparison saved to {output_dir}")


if __name__ == "__main__":
    print("Parameter Tracker Module - Ready for import")
