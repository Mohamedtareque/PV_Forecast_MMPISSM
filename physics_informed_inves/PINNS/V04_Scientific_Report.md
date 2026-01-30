# Scientific Analysis: Physics-Residual Mamba V04 (Adaptive Spatial Wavelet)

**Date**: 2026-01-15
**Subject**: Theoretical Deconstruction of `physics_residual_mamba_spatial_wavelet_v04.py`
**Architecture Integration**: Physics-Informed Neural Networks (PINNs), State Space Models (Mamba), Wavelet Theory, and Bayesian Deep Learning.

---

## 1. Introduction & Scientific Rationale

The `physics_residual_mamba_spatial_wavelet_v04.py` implements a **Physics-Residual Hybrid System** designed for Photovoltaic (PV) power forecasting. Unlike "black-box" deep learning models, this architecture explicitly decouples the predictable physical behavior of the PV plant from the stochastic residual errors caused by atmospheric dynamics.

The core innovation in V04 is the **Adaptive Loss Weighting** mechanisms based on Homoscedastic Uncertainty, addressing the "Gradient Pathology" often observed in multi-objective optimization where physics gradients are overwhelmed by data gradients (or vice versa).

---

## 2. Theoretical Core I: Adaptive Physics Loss
**Problem Addressed**: Vanishing Gradients in PINNs.
When training a model with two competing objectives—minimizing data error ($\mathcal{L}_{data}$) and enforcing physical smoothness/consistency ($\mathcal{L}_{phys}$)—a fixed weight hyperparameter ($\lambda$) is typically used:
$$ \mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{phys} $$
If $\nabla \mathcal{L}_{phys}$ differs significantly in magnitude from $\nabla \mathcal{L}_{data}$, one task dominates, and the other gradient effectively "vanishes," leading to suboptimal convergence.

**Solution**: Homoscedastic Uncertainty Weighting.
Derived from Bayesian Deep Learning (Kendall et al., 2018), we interpret the multi-task loss as maximizing the Gaussian log-likelihood with task-dependent uncertainty parameters $\sigma$.

**Mathematical Formulation**:
The probabilistic model assumes the noise for each task $\tau$ is Gaussian with variance $\sigma_{\tau}^2$. The negative log-likelihood minimization results in:

$$ \mathcal{L}(\theta, \sigma_1, \sigma_2) = \frac{1}{2\sigma_{data}^2} \mathcal{L}_{data}(\theta) + \frac{1}{2\sigma_{phys}^2} \mathcal{L}_{phys}(\theta) + \log \sigma_{data} + \log \sigma_{phys} $$

In the code (`AdaptivePhysicsLoss`), this is implemented using `log_sigma` parameters for numerical stability:
$$ s := \log(\sigma^2) \rightarrow \frac{1}{2}e^{-s}\mathcal{L} + \frac{1}{2}s $$

**Scientific Logic**:
- **Automatic Balancing**: As $\mathcal{L}_{data}$ decreases, the model decreases $\sigma_{data}$ (increasing weight) to refine the fit. If physics constraints are violated (high $\mathcal{L}_{phys}$), $\sigma_{phys}$ increases to penalize uncertainty, effectively re-weighting the gradients dynamically.
- **Heteroscedasticity vs. Homoscedasticity**: The code uses homoscedasticity (task-dependent, not input-dependent uncertainty), meaning $\sigma$ is a learnable scalar for the entire dataset, not a function of $x$.

---

## 3. Theoretical Core II: Hyper-Robust Physics Layer (V05)
The code retains the "V05" physics layer, which replaces simplified PV approximations with high-fidelity physical equations essential for spatial generalization.

### A. Irradiance Decomposition
Standard models assume Global Horizontal Irradiance ($GHI$) correlates linearly with power. V05 decomposes $GHI$ into Beam ($G_b$) and Diffuse ($G_d$) components using a learnable sigmoid parameter $\delta$ (diffuse fraction):
$$ G_{diff} = G \cdot \sigma(\delta_{raw}) $$
$$ G_{beam} = G \cdot (1 - \sigma(\delta_{raw})) $$

**Why?**: Geometric adjustments (Tilt, Incidence Angle Modifier) strictly apply to the **Beam** component. Diffuse light is isotropic. Applying geometry to the full GHI introduces physical error during low-sun angles.

### B. Dynamic Logarithmic Cooling
The standard linear Faiman model ($T_c = T_a + \frac{G}{U_0 + U_1 W_s}$) fails at high wind speeds. V05 introduces a Logarithmic boundary layer model:
$$ U_{dynamic} = U_{base} + U_{wind} \cdot \ln(1 + W_s + \epsilon) $$

**Scientific Logic**: Turbulent heat transfer coefficient scales logarithmically with Reynolds number (wind speed) over flat plates (PV modules), not linearly. This prevents the model from predicting unrealistic cooling during storms.

### C. Spectral Shift Efficiency
PV efficiency drops when the solar spectrum redshifts (at sunset/sunrise, creating high Air Mass). The model explicitly calculates Air Mass ($AM$) using the Kasten-Young formulation from Solar Zenith Angle ($Z$):
$$ AM \approx \frac{1}{\cos(Z) + 0.50572(96.07995 - Z)^{-1.6364}} $$
$$ \eta_{spectral} = 1 + \omega \cdot (1 - AM) $$
Where $\omega$ (`spec_coeff`) is the learnable spectral coefficients.

---

## 4. Theoretical Core III: Wavelet & Mamba Backbone
The underlying neural architecture processes the residuals ($y - y_{physics}$).

### A. Wavelet Decomposition (db4)
The Input $X$ passes through a Discrete Wavelet Transform (Daubechies 4):
$$ X \rightarrow [T(X), S(X)] $$
- **Approximation Coeffs ($T$)**: Represents the low-frequency **Trend**.
- **Detail Coeffs ($S$)**: Represents high-frequency **Seasonality** and noise.

**Scientific Logic**: The Mamba backbone processes these streams separately (or stacked). PV power has strong diurnal seasonality ($24h$ cycle) but stochastic high-frequency cloud events. Decomposing them allows the SSM to learn long-range trend dependencies without being confused by high-frequency noise.

### B. Mamba (State Space Model)
Instead of Transformers ($O(L^2)$), Mamba uses Structured State Space Models ($O(L)$):
$$ h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t) $$
$$ y(t) = \mathbf{C}h(t) $$
Discretized via Zero-Order Hold (ZOH).
**Why?**: Weather patterns (advection) imply long-range dependencies (e.g., a cloud front 100km away affecting the site hours later). Mamba captures this efficient continuous-time context better than RNNs and faster than Transformers.

---

## 5. Spatial Logic: The Physics Isolation Principle
The code implements a strict information routing strategy:

- **Physics Layer Input**: $X_{St07}$ (Target Station) **ONLY**.
- **Mamba Layer Input**: $X_{St07} \oplus X_{St08}$ (Neighbor Station).

**Hypothesis**: The Physics Layer represents the *Device Under Test* (the PV panel). A PV panel at Station 07 cannot physically respond to Irradiance at Station 08. Feeding Station 08 data to the Physics Layer would force it to learn non-physical correlations (leakage).
**Residual Learning**: The Mamba layer sees *both* stations. It uses Station 08 as a spatial covariate to predict the *error* in the physics model (e.g., "Physics predicts Clear Sky, but Station 08 saw a cloud 10 mins ago, so I predict a drop").

---

## Summary of Input/Output Transformation

1.  **Input**: Raw Numerical Weather Prediction (NWP) + Historical Lagged Data + Spatial Neighbor Data ($X_{St08}$).
2.  **Physics Branch**: Extracts $G, T, W$ for St07 $\rightarrow$ Applies $Eq_{V05}$ $\rightarrow$ Outputs $P_{phys}$ (The "Base Load").
3.  **Neural Branch**:
    *   Takes Full Spatial Input.
    *   **RevIN**: Normalizes distribution shifts (non-stationarity).
    *   **Wavelet**: Splits into Trend/Seasonality tokens.
    *   **Mamba**: Contextualizes tokens over time.
    *   Outputs $P_{resid}$ (The "Correction").
4.  **Fusion**: $P_{final} = \text{ReLU}(P_{phys} + P_{resid}) \times \text{ClearSkyMask}$
5.  **Optimization**: Balanced via $\frac{1}{2\sigma^2}$ adaptive weighting.
