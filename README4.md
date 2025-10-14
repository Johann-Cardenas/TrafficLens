# TrafficLens: Two-Lane Ring Road Simulation
## Assignment 4 - Microscopic Traffic Flow with Lane-Changing

---

## 📋 Overview

This module implements a **microscopic traffic simulation** on a two-lane ring road, modeling individual vehicle dynamics through three different car-following models and lane-changing behavior via the MOBIL rule. The application provides interactive visualization of both microscopic (individual vehicle) and macroscopic (aggregate traffic) characteristics.

### Key Features
- **Three car-following models:** IDM, Gipps, and GM
- **MOBIL lane-changing rule:** Balances individual incentive with collective safety
- **Ring road topology:** Periodic boundary conditions for steady-state analysis
- **Interactive Streamlit interface:** Real-time parameter adjustment and visualization
- **Comprehensive analysis:** Microscopic trajectories and macroscopic traffic measures

---

## 🚀 Quick Start

### Installation

Ensure you have Python 3.8+ installed, then install dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `streamlit` - Web application framework
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `plotly` - Interactive visualizations

### Running the Application

```bash
streamlit run app4.py
```

The application will open in your default web browser at `http://localhost:8501`.

---

## 🔧 Implementation Details

### 1. Network Setup

#### Ring Road Configuration
- **Topology:** Two-lane circular road with periodic boundaries
- **Lane numbering:** Lane 0 (outer), Lane 1 (inner)
- **Vehicle length:** Zero (point-mass assumption)
- **Initial conditions:** Uniform spacing with small velocity perturbations

#### Initialization
Vehicles are distributed evenly between lanes with staggered positions to avoid initial conflicts:
- Lane 0: Vehicles at positions `i × (L / n₀)` for `i = 0, 1, ..., n₀-1`
- Lane 1: Vehicles at positions `(i × (L / n₁) + L / (2n₁)) mod L`
- Initial velocities: `v₀ = 20 m/s ± 2 m/s` (Gaussian perturbation)

### 2. Car-Following Models

#### 2.1 IDM (Intelligent Driver Model)

**Governing Equation:**
```
a = a_max [1 - (v/v₀)^δ - (s*/s)²]

where s* = s₀ + vT + (v·Δv)/(2√(a·b))
```

**Parameters:**
| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Desired velocity | v₀ | 30 | m/s | Target free-flow speed |
| Time headway | T | 1.5 | s | Safe following time |
| Min spacing | s₀ | 2.0 | m | Jam spacing |
| Max acceleration | a | 1.0 | m/s² | Acceleration capability |
| Comfortable decel | b | 1.5 | m/s² | Normal braking |
| Accel exponent | δ | 4.0 | - | Acceleration profile shape |

**Characteristics:**
- **Stability:** String-stable for typical parameters
- **Behavior:** Smooth transitions, realistic acceleration/deceleration
- **Applications:** Highway traffic, freeway simulation
- **Advantages:** Continuous acceleration, well-calibrated, physically plausible

#### 2.2 Gipps Model

**Governing Equations:**
```
v_free = v + 2.5aτ(1 - v/v₀)√(0.025 + v/v₀)

v_safe = bτ + √[b²τ² - b(2(s-s₀) - vτ - v_lead²/b)]

v_new = min(v_free, v_safe, v₀)

a = (v_new - v) / Δt
```

**Parameters:**
| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Desired velocity | v₀ | 30 | m/s | Target speed |
| Max acceleration | a | 1.0 | m/s² | Acceleration limit |
| Max braking | b | -1.5 | m/s² | Emergency braking |
| Reaction time | τ | 1.0 | s | Driver response delay |
| Vehicle length | s₀ | 2.0 | m | Effective length |

**Characteristics:**
- **Stability:** Highly stable, collision-free by design
- **Behavior:** Conservative, safety-first approach
- **Applications:** Safety-critical scenarios, urban traffic
- **Advantages:** Explicit collision avoidance, bounded speeds

#### 2.3 GM (Gazis-Herman-Rothery) Model

**Governing Equation:**
```
a = α × (v^l / s^m) × (-Δv)

where Δv = v - v_lead
```

**Parameters:**
| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Sensitivity | α | 0.5 | - | Response strength |
| Velocity exponent | l | 1.0 | - | Speed dependency |
| Spacing exponent | m | 0.0 | - | Gap dependency |

**Characteristics:**
- **Stability:** Conditionally stable (depends on α, l, m)
- **Behavior:** Simplified stimulus-response
- **Applications:** Theoretical analysis, string stability studies
- **Limitations:** Can be unstable, less realistic than IDM/Gipps

**Stability Criterion:**
For string stability, approximately: `α < 1 / (2T)` where T is effective time headway.

### 3. MOBIL Lane-Changing Rule

**MOBIL:** Minimizing Overall Braking Induced by Lane changes

#### Decision Criteria

**1. Safety Criterion:**
```
a_new_follower ≥ -b_safe
```
The new follower in the target lane must not brake harder than `b_safe` (typically 4 m/s²).

**2. Incentive Criterion:**
```
(a_target - a_current) + p(a_new_follower_target - a_new_follower_current) > a_th
```

Where:
- `a_target`: Ego vehicle's acceleration in target lane
- `a_current`: Ego vehicle's current acceleration
- `p`: Politeness factor (0 = egoistic, 1 = altruistic)
- `a_new_follower_*`: New follower's acceleration before/after lane change
- `a_th`: Acceleration threshold (typically 0.2 m/s²)

**Parameters:**
| Parameter | Symbol | Default | Units | Description |
|-----------|--------|---------|-------|-------------|
| Politeness | p | 0.3 | - | Weight of others' benefit |
| Safe decel | b_safe | 4.0 | m/s² | Safety threshold |
| Accel threshold | a_th | 0.2 | m/s² | Minimum incentive |

**Behavior:**
- **p = 0:** Purely selfish; only considers own benefit
- **p = 0.5:** Balanced; weighs self and others equally
- **p = 1:** Fully altruistic; prioritizes collective benefit
- **Higher b_safe:** More conservative lane changes
- **Higher a_th:** Fewer lane changes (requires stronger incentive)

### 4. Simulation Engine

#### Time-Stepping Algorithm

The simulation uses explicit Euler integration with three phases per time step:

**Phase 1: Lane-Changing Decision**
```python
for each vehicle:
    evaluate MOBIL criteria for target lane
    if both safety and incentive satisfied:
        schedule lane change
```

**Phase 2: Acceleration Calculation**
```python
for each vehicle:
    identify leader in current lane
    calculate gap and velocity difference
    compute acceleration using car-following model
```

**Phase 3: State Update**
```python
for each vehicle:
    v(t+Δt) = max(0, v(t) + a(t)·Δt)
    x(t+Δt) = (x(t) + v(t)·Δt) mod L
```

**Time Step Selection:**
- Default: `Δt = 0.1 s`
- Smaller values (0.05 s) for higher accuracy
- Larger values (0.2 s) for faster computation
- **Stability condition:** `Δt < s₀ / v_max` to prevent vehicle overlap

#### Leader-Follower Identification

On a ring road with periodic boundaries:
```
gap = (x_leader - x_ego) mod L
```
Leader is the vehicle with minimum positive gap in the target lane.

### 5. Visualization Components

#### 5.1 Microscopic Visualizations

**Vehicle Trajectories (Space-Time Diagram)**
- **X-axis:** Time (s)
- **Y-axis:** Position along ring (m)
- **Color:** Lane 0 (blue), Lane 1 (red)
- **Interpretation:** 
  - Slope = velocity (steeper = faster)
  - Horizontal lines = stopped vehicles
  - Parallel lines = platoon formation
  - Curved lines = acceleration/deceleration

**Velocity Profiles**
- **X-axis:** Time (s)
- **Y-axis:** Velocity (m/s)
- **Features:** Individual vehicle speed evolution
- **Interpretation:**
  - Oscillations indicate instability or stop-and-go
  - Smooth curves suggest stable following
  - Variance shows heterogeneity

**Lane Distribution**
- **X-axis:** Time (s)
- **Y-axis:** Number of vehicles per lane
- **Type:** Stacked area chart
- **Interpretation:**
  - Balance indicates effective lane usage
  - Imbalance suggests lane preference or asymmetry

#### 5.2 Macroscopic Visualizations

**Time-Series Plots**

1. **Average Speed vs Time**
   - Aggregate and per-lane mean velocities
   - Units: m/s
   
2. **Density vs Time**
   - Calculated as: `ρ = N / L` (vehicles/km)
   - Per-lane and total density
   
3. **Flow vs Time**
   - Calculated as: `q = ρ × v̄` (vehicles/hour)
   - Throughput measure

**Fundamental Diagram (Flow-Density)**
- **X-axis:** Density ρ (veh/km)
- **Y-axis:** Flow q (veh/h)
- **Expected shape:**
  - Rising branch (free-flow): `q = ρ × v_free`
  - Peak at critical density: `q_max` (capacity)
  - Falling branch (congested): lower speeds
- **Scatter:** Indicates transient dynamics and instabilities

---

## 📊 Parameter Guidelines

### Typical Configurations

#### Stable, Free-Flowing Scenario
```
Model: IDM
Road Length: 1000 m
Vehicles: 30
v₀: 30 m/s
T: 1.5 s
a: 1.0 m/s²
b: 1.5 m/s²
MOBIL p: 0.3
Duration: 120 s
```

#### Congested, Stop-and-Go Scenario
```
Model: IDM
Road Length: 1000 m
Vehicles: 60
v₀: 25 m/s
T: 1.2 s
a: 0.8 m/s²
b: 2.0 m/s²
MOBIL p: 0.5
Duration: 180 s
```

#### Unstable GM Scenario (for analysis)
```
Model: GM
Road Length: 1000 m
Vehicles: 40
α: 0.8
l: 1.5
m: 0.0
MOBIL p: 0.2
Duration: 150 s
```

### Parameter Sensitivity

| Parameter | Effect of Increase | Traffic Impact |
|-----------|-------------------|----------------|
| v₀ | Higher desired speeds | Increased flow, reduced density |
| T (IDM) | Larger headways | Lower capacity, more stable |
| a | Faster acceleration | Quicker recovery from perturbations |
| b | Harder braking | More stable, but harsher dynamics |
| α (GM) | Stronger response | Can cause instability if too high |
| MOBIL p | More altruistic | Fewer but safer lane changes |
| b_safe | More cautious | Reduced lane-change frequency |
| N vehicles | Higher density | Lower speeds, potential congestion |

---

## 🔬 Analysis and Observations

### Model Comparison

#### Stability Characteristics

**IDM:**
- **String-stable** for default parameters
- Exhibits stop-and-go waves only at very high densities (ρ > 60 veh/km)
- Lane-changing can help dissipate congestion by redistributing vehicles
- **Critical density:** ~50-60 veh/km (depends on v₀, T)

**Gipps:**
- **Highly stable** due to explicit collision avoidance
- Rarely produces stop-and-go waves
- May underutilize capacity due to conservative nature
- **Critical density:** ~40-50 veh/km (more conservative)

**GM:**
- **Conditionally stable** depending on (α, l, m)
- Can exhibit string instability with growing oscillations
- Useful for studying instability mechanisms
- **Critical density:** Highly parameter-dependent

#### Lane-Changing Impact

**Positive Effects:**
- Relieves local congestion by moving vehicles from dense to sparse lanes
- Increases overall road capacity utilization
- Dampens some instabilities by providing "escape route"

**Negative Effects:**
- Disrupts following vehicles in both origin and destination lanes
- Can amplify oscillations if politeness is too low
- May cause "zipper" effects at lane merge points

**MOBIL Parameter Effects:**
- **High p (0.7-1.0):** Fewer lane changes, smoother traffic, lower throughput
- **Low p (0.0-0.3):** More lane changes, potential disruption, higher throughput
- **Optimal p:** Typically 0.3-0.5 for balanced performance

### Fundamental Diagram Insights

**Shape Analysis:**

1. **Free-Flow Branch (low ρ):**
   - Linear relationship: q ≈ ρ × v₀
   - Vehicles travel at desired speed
   - Little interaction between vehicles

2. **Capacity Peak:**
   - Occurs at critical density ρ_c
   - Maximum flow q_max (road capacity)
   - Transition from free-flow to congested
   - **IDM:** q_max ≈ 1800-2200 veh/h/lane
   - **Gipps:** q_max ≈ 1600-1900 veh/h/lane
   - **GM:** Highly variable, can be lower

3. **Congested Branch (high ρ):**
   - Negative slope: higher density → lower flow
   - Stop-and-go waves common
   - Vehicles constrained by leaders

**Scatter Characteristics:**
- Wide scatter indicates transient dynamics
- Tight clustering suggests steady-state
- Hysteresis loops show capacity drop phenomena

### Stop-and-Go Wave Formation

**Mechanism:**
1. Small perturbation (e.g., slight braking)
2. Amplified by following vehicles (string instability)
3. Backward-propagating wave of stop-and-go
4. Wave speed: typically -15 to -20 km/h (against traffic flow)

**Factors Promoting Waves:**
- High density (ρ > ρ_critical)
- Large sensitivity (α in GM)
- Short time headways (T in IDM)
- Aggressive acceleration/braking

**Factors Suppressing Waves:**
- Longer time headways (T > 2.0 s)
- Lower density
- Stable car-following models (IDM, Gipps)
- Effective lane-changing (redistributes density)

### Capacity Analysis

**Two-Lane vs Single-Lane:**
- Ideal two-lane capacity: 2 × single-lane capacity
- Actual capacity: 1.7-1.9 × single-lane (due to lane-changing friction)
- **Lane-changing friction:** ~5-15% capacity loss

**Model-Specific Capacity:**
- **IDM:** ~2000 veh/h/lane (well-calibrated to real data)
- **Gipps:** ~1700 veh/h/lane (conservative)
- **GM:** Variable (800-1800 veh/h/lane depending on parameters)

---

## 🎯 Simulation Workflow

### Typical Analysis Procedure

1. **Baseline Simulation**
   - Run with default IDM parameters
   - Observe stable, free-flowing traffic
   - Note fundamental diagram shape

2. **Density Variation**
   - Increase vehicles from 20 to 80
   - Identify critical density for congestion onset
   - Observe transition in FD from free-flow to congested

3. **Model Comparison**
   - Run identical scenarios with IDM, Gipps, GM
   - Compare stability, capacity, oscillations
   - Note differences in velocity profiles

4. **MOBIL Parameter Study**
   - Vary politeness p from 0 to 1
   - Observe lane-change frequency
   - Assess impact on throughput and stability

5. **Perturbation Response**
   - Introduce localized slowdown (adjust initial conditions)
   - Track wave propagation in trajectories
   - Measure damping or amplification

### Interpreting Results

**Signs of Stability:**
- ✅ Smooth velocity profiles
- ✅ Parallel trajectory lines
- ✅ Constant macroscopic measures
- ✅ Tight FD clustering

**Signs of Instability:**
- ⚠️ Oscillating velocities
- ⚠️ Saw-tooth trajectory patterns
- ⚠️ Fluctuating flow/density
- ⚠️ Wide FD scatter

**Effective Lane-Changing:**
- Balanced lane distribution (~50/50)
- Smooth transitions (not excessive switching)
- Improved flow over single-lane equivalent

---

## 📈 Expected Outcomes

### Learning Objectives

After completing this assignment, you should understand:

1. **Car-Following Dynamics**
   - How different models represent driver behavior
   - Stability criteria and string instability
   - Calibration and parameter sensitivity

2. **Lane-Changing Behavior**
   - MOBIL's incentive and safety criteria
   - Impact of politeness on collective performance
   - Trade-offs between individual and system optimality

3. **Microscopic-Macroscopic Link**
   - How individual behavior emerges as aggregate patterns
   - Connection between following models and fundamental diagram
   - Capacity, density, and flow relationships

4. **Simulation Techniques**
   - Time-stepping algorithms
   - Periodic boundary conditions
   - Data collection and visualization

### Practical Applications

This simulation framework can be extended to:
- **Highway design:** Lane addition analysis
- **Traffic management:** Variable speed limits, ramp metering
- **Autonomous vehicles:** Testing AV car-following algorithms
- **Safety analysis:** Collision risk assessment
- **Emissions modeling:** Stop-and-go increases fuel consumption

---

## 🐛 Troubleshooting

### Common Issues

**1. Simulation Too Slow**
- Increase time step Δt (but ensure stability)
- Reduce number of vehicles
- Decrease simulation duration
- Use simpler model (GM instead of Gipps)

**2. Unrealistic Behavior (Crashes, Negative Speeds)**
- Decrease time step Δt
- Increase minimum spacing s₀
- Reduce maximum acceleration
- Check for NaN values in acceleration

**3. No Lane Changes Occurring**
- Lower acceleration threshold a_th
- Reduce politeness p (more selfish)
- Increase vehicle density (creates more incentive)
- Check if b_safe is too high

**4. Excessive Oscillations**
- Reduce sensitivity α (GM model)
- Increase time headway T (IDM)
- Use more stable model (Gipps)
- Lower density

### Validation Checks

**Physical Plausibility:**
- Velocities: 0 ≤ v ≤ v₀
- Accelerations: -b_max ≤ a ≤ a_max
- Spacings: s ≥ 0 (no overlap)
- Positions: 0 ≤ x < L

**Conservation:**
- Number of vehicles constant
- Total road length conserved
- No vehicles lost at boundaries

---

## 📚 References

### Key Papers

1. **IDM:** Treiber, M., Hennecke, A., & Helbing, D. (2000). "Congested traffic states in empirical observations and microscopic simulations." *Physical Review E*, 62(2), 1805.

2. **Gipps:** Gipps, P. G. (1981). "A behavioural car-following model for computer simulation." *Transportation Research Part B*, 15(2), 105-111.

3. **GM:** Gazis, D. C., Herman, R., & Rothery, R. W. (1961). "Nonlinear follow-the-leader models of traffic flow." *Operations Research*, 9(4), 545-567.

4. **MOBIL:** Kesting, A., Treiber, M., & Helbing, D. (2007). "General lane-changing model MOBIL for car-following models." *Transportation Research Record*, 1999(1), 86-94.

5. **Traffic Flow Theory:** Treiber, M., & Kesting, A. (2013). *Traffic Flow Dynamics: Data, Models and Simulation*. Springer.

### Online Resources

- [Traffic Simulation Website](https://traffic-simulation.de/) - Interactive demos
- [IDM Documentation](https://en.wikipedia.org/wiki/Intelligent_driver_model)
- [MOBIL Model](https://traffic-simulation.de/MOBIL.html)

---

## 🔧 Customization and Extension

### Adding New Car-Following Models

To implement a new model (e.g., Optimal Velocity Model):

```python
class OVM:
    def __init__(self, alpha: float = 0.5, v_max: float = 30.0):
        self.alpha = alpha
        self.v_max = v_max
    
    def optimal_velocity(self, s: float) -> float:
        # Sigmoid function for optimal velocity
        return self.v_max * (np.tanh(s - 5) + np.tanh(5)) / (1 + np.tanh(5))
    
    def acceleration(self, v: float, s: float, dv: float) -> float:
        v_opt = self.optimal_velocity(s)
        return self.alpha * (v_opt - v)
```

Then add to simulation initialization logic.

### Multi-Lane Extension

To extend beyond two lanes:
1. Modify `Vehicle` class to support `lane ∈ {0, 1, ..., n-1}`
2. Update `_check_lane_change` to consider left and right lanes
3. Add lane-specific visualization layers

### Stochastic Elements

Add noise for realism:
```python
def _calculate_acceleration(self, ...):
    a_deterministic = self.cf_model.acceleration(...)
    noise = np.random.normal(0, 0.1)  # m/s² noise
    return a_deterministic + noise
```

---

## 💻 Technical Specifications

**Code Structure:**
- `app4.py`: Main simulation and Streamlit interface (~1000 lines)
- Object-oriented design with clear separation:
  - `Vehicle`: Data class for vehicle state
  - `IDM`, `Gipps`, `GM`: Car-following model classes
  - `MOBIL`: Lane-changing logic
  - `RingRoadSimulation`: Simulation engine
  - Plotting functions: Modular visualization
  - `main()`: Streamlit app logic

**Performance:**
- Typical simulation (40 vehicles, 120 s, Δt=0.1): ~2-5 seconds
- Memory usage: <100 MB for trajectories
- Scales approximately O(N²) due to leader-follower search

**Dependencies:**
```
streamlit >= 1.24.0
numpy >= 1.24.0
pandas >= 2.0.0
plotly >= 5.14.0
```

---

## 🎓 Pedagogical Notes

### For Students

This assignment bridges **microscopic** (individual vehicle) and **macroscopic** (aggregate flow) perspectives in traffic engineering. Key learning points:

1. **Emergence:** Complex traffic patterns arise from simple rules
2. **Nonlinearity:** Small parameter changes can cause qualitative shifts
3. **Feedback:** Lane-changing creates coupled dynamics between lanes
4. **Validation:** Compare simulation to theory (e.g., fundamental diagram)

### For Instructors

**Assessment Criteria:**
- Correct implementation of all three models
- Proper MOBIL integration
- Accurate visualization of micro/macro measures
- Insightful analysis of stability and capacity
- Clear documentation and code quality

**Extension Ideas:**
- Add on-ramps/off-ramps
- Heterogeneous vehicle mix (cars, trucks)
- Adaptive cruise control (ACC) vehicles
- Driver heterogeneity (aggressive vs. cautious)

---

## 📝 Conclusion

This simulation provides a comprehensive framework for studying microscopic traffic dynamics on multi-lane roads. By implementing and comparing three established car-following models alongside realistic lane-changing behavior, we gain insight into:

- **Stability mechanisms:** Why some models produce smooth flow while others oscillate
- **Capacity determinants:** How model parameters and lane-changing affect throughput
- **Emergent phenomena:** Stop-and-go waves, lane imbalances, hysteresis
- **Design implications:** Optimal parameter choices for different scenarios

The interactive Streamlit interface makes exploration intuitive, while the modular code structure facilitates extensions and customization.

**Next Steps:** Consider implementing heterogeneous fleets, on/off-ramps, or integrating real traffic data for calibration.

---

## 📞 Contact and Support

For questions, bug reports, or contributions related to this assignment module, please refer to the main TrafficLens repository.

**Author:** TrafficLens Development Team  
**Version:** 1.0  
**Last Updated:** October 2025

---

*Happy Simulating! 🚗💨*

