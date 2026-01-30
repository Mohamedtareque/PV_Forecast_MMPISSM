# Methodology Notes: Clear Sky Indices Interpretation

## 1. Overview of Indices
This study employs two distinct normalization indices to isolate atmospheric effects from system performance:

- **$K_{CS}$ (Clear Sky Index for Irradiance):**
  Defined as the ratio of measured Global Horizontal Irradiance (GHI) to the modeled clear-sky GHI ($GHI_{clr}$) at the same solar geometry.
  $$ K_{CS} = \frac{GHI_{meas}}{GHI_{clr}} $$
  *Interpretation:* $K_{CS}$ is strictly an **atmospheric metric**. Values near $1.0$ indicate clear skies. Values significantly below $1.0$ quantify cloud attenuation. Values slightly above $1.0$ (up to $\sim1.25$) represent "cloud-edge enhancement" where scattering from cumulus clouds momentarily intensifies solar radiation. $K_{CS}$ does *not* reflect the state of the PV array (e.g., it is insensitive to soiling or inverter faults).

- **$K_{PV}$ (Clear Sky Performance Index):**
  Defined as the ratio of measured AC power output ($P_{meas}$) to the modeled clear-sky AC power ($P_{clr}$), which accounts for array geometry, temperature, and inverter efficiency.
  $$ K_{PV} = \frac{P_{meas}}{P_{clr}} $$
  *Interpretation:* $K_{PV}$ is a **system performance metric**. It normalizes power output against the theoretical maximum for the given ambient conditions.
    - **$K_{PV} \approx 1.0$ (0.95–1.05):** Indicates the system is operating as expected under clear skies.
    - **$K_{PV} \ll 1.0$:** Indicates performance loss due to clouds, shading, soiling, degradation, or faults.
    - **$K_{PV} > 1.05$:** Serves as a **Warning Flag**. Sustained values above 1.05 typically indicate measurement errors (e.g., sensor calibration drift), incorrect model parameters (e.g., under-estimated capacity), or valid but rare cloud-edge effects. It should not be interpreted as the plant "outperforming" physics.

## 2. Handling of Zeros and Missing Data (NaNs)
In the raw time-series data, the indices are mathematically undefined during nighttime or when the denominator is negligible.
- **For continuous plotting:** We assign a value of **0.0** to these periods to maintain time-series continuity (masking).
- **For statistical analysis (Means, Distributions, Quantiles):** These 0.0 values must be treated as **Invalid Data**. They are excluded from all aggregations to prevent biasing the results downward. A night-time $K_{PV}$ of 0.0 does not imply "zero performance"; it implies "undefined operating condition."

## 3. Physical Bounds and Quality Control
To mitigate the impact of measurement outliers:
- A strict **Quality Control (QC) Cap** is applied at **1.25** for both indices. Values exceeding this threshold are clipped, as they are physically implausible for sustained operation and likely result from data synchronization errors or sensor spikes.
- In reporting, $K_{PV}$ values in the range **1.05–1.25** are flagged as "Anomalous High Performance" requiring investigation, rather than being accepted as valid high-yield operation.
