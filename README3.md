# TrafficLens — CTM with Lane-Drop Bottleneck (Assignment 3)

## Overview

This Streamlit application implements a **Cell Transmission Model (CTM)** with a triangular fundamental diagram to simulate traffic flow through a 5-mile highway segment featuring a **2→1 lane-drop bottleneck**. The model demonstrates critical traffic flow phenomena including queue formation, shockwave propagation, and capacity reduction effects.

## Mathematical Foundation

### Triangular Fundamental Diagram

The model uses a triangular fundamental diagram with three key parameters:

- **Free-flow speed** (v_f): Maximum speed vehicles can achieve in uncongested conditions
- **Congestion wave speed** (w): Speed at which congestion propagates upstream
- **Jam density per lane** (k_j): Maximum vehicle density per lane

#### Derived Parameters

```
Critical density: k_c = (w × k_j) / (v_f + w)
Maximum flow per lane: q_max = v_f × k_c = (v_f × w × k_j) / (v_f + w)
```

### Cell Transmission Model Equations

The CTM discretizes the highway into cells and tracks traffic state evolution:

#### Demand and Supply Functions
```
Demand: S_i^t = min(v_f × k_i^t, Q_i)
Supply: R_i^t = min(w × (k_j - k_i^t), Q_i)
```

#### Interface Flow Computation
```
f_{i+1/2}^t = min(S_i^t, R_{i+1}^t, C_{i+1/2})
```

#### Density Update (Conservation Equation)
```
k_i^{t+1} = k_i^t + (Δt/Δx) × (f_{i-1/2}^t - f_{i+1/2}^t)
```

#### Stability Condition (CFL)
```
Δt ≤ Δx / max(v_f, w)
```

## Implementation Details

### 1. Model Parameters & Justification

#### Spatial Discretization
- **Cell length (dx)**: 50-500m (default: 100m)
  - *Justification*: Balances computational efficiency with spatial resolution. 100m provides adequate detail for 5-mile segment while maintaining stability.

#### Temporal Discretization
- **Time step (dt)**: 0.5-5.0s (auto-limited by CFL condition)
  - *Justification*: Automatic enforcement of CFL stability ensures numerical accuracy. Typical values 1-2s provide good temporal resolution.

#### Traffic Flow Parameters
- **Free-flow speed**: 20-90 mph (default: 70 mph)
  - *Justification*: Typical highway free-flow speeds. 70 mph represents standard interstate conditions.

- **Wave speed**: 5-30 mph (default: 18 mph)
  - *Justification*: Empirically observed congestion wave speeds. 18 mph is well-established in literature.

- **Jam density**: 60-240 veh/km/lane (default: 140 veh/km/lane)
  - *Justification*: Representative of highway conditions. 140 veh/km/lane corresponds to ~7m vehicle spacing at jam.

#### Bottleneck Configuration
- **Position**: Variable along 5-mile segment (default: 2.5 miles)
  - *Justification*: Mid-segment placement allows observation of both upstream queue formation and downstream recovery.

- **Capacity drop factor**: 0.40-0.60 (default: 0.50)
  - *Justification*: Represents 50% capacity reduction typical in lane-drop scenarios. Based on empirical observations of merge capacity.

### 2. Lane Configuration

The model implements a **2→1 lane-drop**:
- **Upstream of bottleneck**: 2 lanes
- **Downstream of bottleneck**: 1 lane
- **Capacity scaling**: All parameters scale linearly with lane count

```python
lanes = np.where(x_cell_mid < x_interfaces[bn_idx], 2, 1)
kj_cell_vpkm = kj_lane_vpkm * lanes
qcap_cell_vps = qmax_lane_vps * lanes
```

### 3. Demand Profile Options

#### Piecewise Constant (3-segment)
- **Default**: 15min @ 1800 veh/h → 30min @ 3600 veh/h → remainder @ 2400 veh/h
- *Justification*: Simulates typical rush-hour pattern with peak demand exceeding bottleneck capacity

#### CSV Upload
- **Format**: Columns `time_s`, `demand_vph`
- **Interpolation**: Linear between specified points
- *Justification*: Allows realistic demand profiles from field data

### 4. Initial Conditions

Four initialization modes:
- **Free**: 0.05 × k_j (light traffic)
- **Capacity**: k_c (at capacity)
- **Near jam**: 0.90 × k_j (heavily congested)
- **Custom**: User-specified density

*Justification*: Covers full range of initial traffic states to explore different scenarios.

### 5. Output Options

#### Per-lane vs Total Outputs
- **Per-lane**: Density and flow normalized by lane count
- **Total**: Aggregate across all lanes
- *Justification*: Per-lane outputs enable direct comparison with field measurements and traffic engineering standards.

## Key Features & Visualizations

### 1. Real-time Probe Analysis
- **Time series plots** at user-selected location
- **Metrics**: Speed, density, flow evolution
- *Purpose*: Detailed examination of traffic conditions at specific points

### 2. Space-Time Heatmaps
- **Jet colormap** with percentile-based scaling (1st-99th percentile)
- **Grid overlays** with adaptive tick spacing
- **Bottleneck annotation** with dashed line
- *Purpose*: Visualize shockwave propagation and queue evolution

### 3. N-curve Analysis
- **Cumulative vehicle counts** at upstream/downstream detectors
- **Queue visualization** via vertical gap between curves
- **Peak queue annotation** with maximum stored vehicles
- *Purpose*: Quantify queue formation and dissipation dynamics

### 4. Automatic Diagnostics
- **Queue detection**: k > k_c criterion upstream of bottleneck
- **Throughput analysis**: Peak and 95th percentile flows
- **Performance metrics**: Total queued time, detector spacing
- *Purpose*: Automated assessment of bottleneck performance

## Technical Implementation

### 1. Performance Optimization

#### Caching Strategy
```python
@st.cache_data(show_spinner=True)
def run_ctm(...):
```
- **Purpose**: Prevents re-computation when only visualization parameters change
- **Signature tracking**: Ensures cache invalidation when model parameters change

#### Memory-Efficient Arrays
```python
dtype="float32"  # Reduces memory usage vs float64
```
- **Justification**: Sufficient precision for traffic modeling while reducing memory footprint

### 2. Numerical Stability

#### CFL Condition Enforcement
```python
dt_max = 0.9 * dx_m / max(vf_mps, w_mps)
dt_s = min(dt_user, dt_max)
```
- **Safety factor**: 0.9 × theoretical limit ensures stability
- **Automatic adjustment**: User warned when dt reduced for stability

#### Density Clipping
```python
k_vpm[t + 1] = np.clip(k_new, 0.0, kj_cell_vpm)
```
- **Physical bounds**: Prevents negative densities and overcrowding

### 3. Unit Conversion System

Comprehensive unit handling for international compatibility:
```python
MPH_TO_MPS = 0.44704    # Speed conversion
MILE_TO_M = 1609.344    # Distance conversion
```

Functions for all conversions:
- Speed: mph ↔ m/s
- Flow: veh/h ↔ veh/s
- Density: veh/km ↔ veh/m

## Quick Calibration Feature

### TGSIM Integration
- **Purpose**: Suggest model parameters from real trajectory data
- **Method**: Extract flow-density relationships from TGSIM data
- **Output**: Recommended q_max, k_c, k_j values

```python
def quick_calibrate_from_tgsim(df, vf_mph, w_mph):
    # Aggregate by time bins and lanes
    # Extract maximum observed flows
    # Derive triangular FD parameters
```

## Execution Instructions

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Required Dependencies

```
streamlit>=1.38
pandas>=2.2
numpy>=1.26
plotly>=5.22
```

### 3. Running the Application

```powershell
# Navigate to project directory
cd "c:\Users\johan\Box\05 Repositories\TrafficLens"

# Launch Streamlit app
streamlit run app3.py
```

### 4. Configuration

Ensure `.streamlit/config.toml` is properly configured for local development:

```toml
[server]
maxUploadSize = 2048
enableCORS = false
address = "localhost"
port = 8501
headless = false

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501
```

## Usage Workflow

### 1. Parameter Configuration
1. Set **model parameters** (v_f, w, k_j) based on highway characteristics
2. Choose **discretization** (dx, dt) balancing accuracy and speed
3. Position **bottleneck** and set capacity drop factor
4. Define **demand profile** (piecewise or CSV)
5. Set **initial conditions** appropriate for scenario

### 2. Simulation Execution
1. Click **"Run / Re-run CTM"** to execute simulation
2. Monitor **automatic stability warnings** if dt adjusted
3. Review **derived parameters** (k_c, q_max) for reasonableness

### 3. Results Analysis
1. Examine **probe time series** for detailed temporal behavior
2. Study **space-time heatmaps** for shockwave patterns
3. Analyze **N-curves** for queue quantification
4. Review **automatic diagnostics** for performance summary

### 4. Parameter Sensitivity
1. Systematically vary **single parameters** to study sensitivity
2. Use **cached results** for efficient visualization parameter tuning
3. Compare **scenarios** by documenting key metrics

## Validation & Verification

### 1. Conservation Check
Total vehicles entering = Total vehicles in system + Total vehicles exiting

### 2. Capacity Verification
Bottleneck throughput should not exceed configured capacity limit

### 3. Physical Bounds
- Densities: 0 ≤ k ≤ k_j
- Speeds: 0 ≤ v ≤ v_f
- Flows: 0 ≤ q ≤ q_max

### 4. Stability Monitoring
CFL condition automatically enforced with user feedback

## Common Use Cases

### 1. Incident Analysis
- Model lane closures with appropriate capacity drops
- Study queue formation and dissipation times
- Evaluate upstream detector placement strategies

### 2. Demand Management
- Test ramp metering effectiveness
- Analyze dynamic pricing scenarios
- Optimize signal timing coordination

### 3. Infrastructure Planning
- Evaluate bottleneck elimination alternatives
- Study capacity enhancement benefits
- Assess detector system requirements

### 4. Educational Demonstrations
- Illustrate fundamental traffic flow principles
- Demonstrate shockwave mechanics
- Show queue dynamics and N-curve interpretation

## Troubleshooting

### Common Issues

1. **Stability Warnings**: Reduce dt or increase dx
2. **Memory Errors**: Reduce simulation horizon or increase dt
3. **Unrealistic Results**: Check parameter values against typical ranges
4. **CSV Upload Errors**: Verify column names and format
5. **Performance Issues**: Enable caching and reduce visualization frequency

### Parameter Guidelines

- **v_f**: 50-80 mph for highways
- **w**: 15-25 mph empirically observed
- **k_j**: 120-160 veh/km/lane typical
- **dt**: Start with 1.0s, adjust based on CFL warnings
- **dx**: 100-200m for highway segments

## Further Development

### Potential Extensions

1. **Multi-class vehicles** (cars, trucks with different parameters)
2. **Variable speed limits** (time-dependent v_f)
3. **Ramp integration** (on/off-ramps with merging models)
4. **Weather effects** (capacity and speed reductions)
5. **Connected vehicle integration** (CV penetration effects)
6. **Adaptive signal control** (responsive bottleneck management)

### Model Enhancements

1. **Higher-order accuracy** (Godunov scheme, MUSCL)
2. **Stochastic elements** (demand variability, capacity fluctuations)
3. **Behavioral models** (lane-changing, gap acceptance)
4. **Network extension** (multiple bottlenecks, route choice)

---

**Note**: This implementation prioritizes clarity, educational value, and computational efficiency while maintaining sufficient accuracy for traffic engineering applications. The modular design facilitates both classroom instruction and research applications.