# TrafficLens
Class project of CEE 416 Traffic Capacity Analysis

## TGSIM Explorer (Assignment 1)
Interactive Streamlit app to visualize and analyze the Third-Generation Simulation (TGSIM) I-90/I-94 Run 1 trajectory data.

### 1) How to run

```powershell
# From existing environment
python -m pip install --upgrade pip   # Upgrade pip
pip install -r requirements.txt       # Set up requirements
streamlit run app.py                  # Run app

# From a new environment
python -m venv .venv && . .venv/Scripts/activate  
pip install -r requirements.txt
streamlit run app.py
```

> [!NOTE]
> Place your dataset in ./data/ (CSV) or use the Upload option in the sidebar. 
> Notice there is a 200MB limit for uploads..

### 2) Dataset Access 
The following are required columns: `id`,  `time`, `xloc_kf`, `yloc_kf`, `lane_kf` , `speed_kf` , `acceleration_kf`, `length_smoothed`, `width_smoothed`, `type_most_common`, `av`,  `run_index`.

The source of the sample data is the **U.S. DOT data portal — TGSIM I-90/I-94, Run 1 trajectory data** .

See below the dataset column dictionary:

- `id`: Unique numerical identifier for each detected vehicle trajectory. Used to group observations, compute headways, and plot per-vehicle trajectories.
- `time`: Time stamp of the measuremente, relative to the start of the run (in seconds). Used for all time-based analysis (Trajectories, speeds, headways, flow).
- `xloc_kf`: Vehicle x-coordinate in global coordinates of the reference image, smoothed with a Kalman Filter. One spatial axis of the trajectory, also part of ROI selection.
- `yloc_kf`: Vehicle y-coordinate in global coordinates of the reference image, smoothed with a Kalman Filter. Often treated as the longitudinal axis for headway and density calculations.
- `lane_kf`: Lane assignment after Kalma Filtering. Lane IDs correspond to positions on the reference image.
  - For Runs 1, 3, and 5: Lane 1 = innermost, Lane 6 = rightmost.
  - For Runs 2, 4, and 6: Lane 10 = innermost, Lane 15 = rightmost. 
  - Express lanes are excluded. This field enables lane-based filtering and lane occupancy plots.
- `speed_kf`: Vehicle speed after Kalman Filtering. Basis for speed distributions, time-series plots, and space-mean speed.
- `acceleration_kf`: Vehicle acceleration after Kalman Filtering. Used in acceleration distribution plots and for examining vehicle dynamics.
- `length_smoothed`: Smoothed bounding-box length of the detected vehicle (proxy for physical vehicle length). Useful if refining space headways into bumper-to-bumper gaps.
- `width_smoothed`: Smoothed bounding-box width of the detected vehicle (proxy for physical vehicle width). Relevant for cross-lane spaceing checks.
- `type_most_common`: Vehicle classification (e.g., small vehicle, large vehicle). Provides filtering and stratification of analyses.
- `av`: Indicator of automation status. Labeled `Yes` for automated vehicles and `No` otherwise. Supports comparative studies (AV vs human-driven).
- `run_index`: Index of the data collection process (e.g., Run 1). Ensures subsets of the dataset are analyzed in context of the correct experimental run.

### 3) Preprocessing

- Timestamps normalized to numeric seconds if provided as datetime strings.
- Memory optimization: numeric downcasting; type_most_common, av as categoricals.
- Optional conversion to Parquet is recommended for faster reloads.

### 4) App Layout

- **Filters**: Run, lane, vehicle type, AV flag, and (optional) specific vehicle IDs.

- **ROI (Region of Interest)**: Choose X-range and Y-range sliders to define a roadway segment used by the analytics.

- **Zoom & selection**: Use Plotly’s toolbar (zoom/box select/lasso) for visual exploration. The ROI sliders control what data feeds the distributions.

- **Longitudinal axis**: Select whether the roadway direction aligns with xloc_kf or yloc_kf—affects headway/density definitions.

### 5) Visualization

- **Speeds**:  Histogram of individual speeds within the ROI.

- **Time headway**: Differences between consecutive vehicle entry times into the ROI (per run & lane).

- **Space headway**: Within each time bin, sort vehicles along the longitudinal axis and take neighbor spacing (meters).

- **Space-mean speed**: Harmonic mean speed across vehicles present in the ROI per time bin (units: m/s).

- **Flow**: Unique vehicles traversing the ROI per selected time window (veh / window).

- **Density**: Vehicles per kilometer within the ROI per time bin.

### 6) Interpretation

- **Flow vs. Density**: Expect increasing flow with density up to capacity, then a drop as congestion sets in.

- **Space-mean vs. time-mean speed**: Space-mean (harmonic) downweights high speeds; appropriate for roadway segments.

- **Headways**: Smaller headways reflect higher flow and potentially reduced safety margins.
