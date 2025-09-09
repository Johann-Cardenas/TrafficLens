import io
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TrafficLens — TGSIM Explorer", layout="wide")

# ============ Required columns & fast-read config ============
REQ_COLS = {
    "id", "time", "xloc_kf", "yloc_kf", "lane_kf",
    "speed_kf", "acceleration_kf", "length_smoothed",
    "width_smoothed", "type_most_common", "av", "run_index"
}
USECOLS = sorted(list(REQ_COLS))

# Nullable / memory-friendly dtypes for CSV read (coerced post-read if needed)
DTYPES = {
    "id": "Int32",
    "time": "float32",  # if time is ISO, we'll convert later
    "xloc_kf": "float32",
    "yloc_kf": "float32",
    "lane_kf": "Int16",
    "speed_kf": "float32",
    "acceleration_kf": "float32",
    "length_smoothed": "float32",
    "width_smoothed": "float32",
    "type_most_common": "string",  # will cast to category later
    "av": "string",                # will cast to category later
    "run_index": "Int16",
}

def validate_columns(df: pd.DataFrame) -> list[str]:
    return sorted(list(REQ_COLS.difference(df.columns)))

def _norm_path(p: str) -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.expanduser(p.strip())
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(base, p))
    return p

# ============ Smart, fast readers ============

def _csv_to_parquet_sidecar(csv_path: str, df: pd.DataFrame) -> str | None:
    """
    Write a compressed Parquet sidecar next to the CSV for future fast loads.
    Returns sidecar path if successful, else None.
    """
    try:
        sidecar = os.path.splitext(csv_path)[0] + ".sidecar.parquet"
        # Categories now, so Parquet stores dictionaries (smaller)
        for c in ("type_most_common", "av"):
            if c in df.columns:
                df[c] = df[c].astype("category")
        df.to_parquet(sidecar, compression="zstd", index=False)
        return sidecar
    except Exception:
        return None

def _try_read_csv_fast(path_or_buf, name: str) -> pd.DataFrame:
    """
    Fast CSV read using PyArrow engine + usecols + dtypes.
    Falls back to default engine if PyArrow unavailable.
    """
    kwargs = dict(usecols=USECOLS, dtype=DTYPES, low_memory=False)
    try:
        return pd.read_csv(path_or_buf, engine="pyarrow", **kwargs)
    except Exception:
        return pd.read_csv(path_or_buf, **kwargs)

@st.cache_data(show_spinner=False)
def _read_any(file_like_or_path, name: str, mtime_key: float) -> pd.DataFrame:
    """
    Load CSV/Parquet from path or file-like.
    - If CSV path: prefer an existing sidecar Parquet if fresher than CSV.
      Otherwise read CSV fast, then write sidecar for future runs.
    - If upload: read CSV fast or Parquet directly (no sidecar).
    """
    name_l = name.lower()

    # --- Filesystem path branch ---
    if isinstance(file_like_or_path, (str, os.PathLike)):
        path = str(file_like_or_path)
        if name_l.endswith((".parquet", ".pq")):
            return pd.read_parquet(path, columns=USECOLS)

        if name_l.endswith(".csv"):
            csv_mtime = os.path.getmtime(path) if os.path.exists(path) else 0
            sidecar = os.path.splitext(path)[0] + ".sidecar.parquet"
            if os.path.exists(sidecar) and os.path.getmtime(sidecar) >= csv_mtime:
                # Fast path: use the sidecar
                return pd.read_parquet(sidecar, columns=USECOLS)
            # Else read CSV fast, then create sidecar
            df = _try_read_csv_fast(path, name)
            _csv_to_parquet_sidecar(path, df)
            return df

        # Unknown extension: try CSV fast, else default
        try:
            df = _try_read_csv_fast(path, name)
        except Exception:
            df = pd.read_csv(path)
        return df

    # --- Upload (file-like) branch ---
    if name_l.endswith((".parquet", ".pq")):
        return pd.read_parquet(file_like_or_path, columns=USECOLS)
    if name_l.endswith(".csv"):
        return _try_read_csv_fast(file_like_or_path, name)
    # Fallback: assume CSV
    return _try_read_csv_fast(file_like_or_path, name)

@st.cache_data(show_spinner=False)
def list_dataset_files(folder_abs: str) -> list[tuple[str, str]]:
    if not os.path.isdir(folder_abs):
        return []
    out = []
    for fn in sorted(os.listdir(folder_abs)):
        if fn.lower().endswith((".csv", ".parquet", ".pq")):
            out.append((fn, os.path.join(folder_abs, fn)))
    return out

@st.cache_data(show_spinner=True)
def load_from_path_cached(path_abs: str, mtime: float) -> pd.DataFrame:
    return _read_any(path_abs, path_abs, mtime)

# ============ Sidebar: data ============
with st.sidebar:
    st.header("Data")
    source = st.radio("Choose data source", ["Upload", "From folder"], horizontal=True)
    df = None

    if source == "Upload":
        up = st.file_uploader("Upload TGSIM trajectory file", type=["csv", "parquet", "pq"])
        if up is not None:
            bio = io.BytesIO(up.getvalue())
            with st.spinner(f"Loading {up.name}…"):
                # BytesIO has no mtime; pass time() to bust/refresh cache on each upload
                df = _read_any(bio, up.name, time.time())
    else:
        folder_input = st.text_input("Folder path", value="./data/")
        folder_abs = _norm_path(folder_input)
        files = list_dataset_files(folder_abs)
        if files:
            choice = st.selectbox("Pick a file", [f[0] for f in files], index=0)
            chosen_path = dict(files)[choice]
            try:
                mtime = os.path.getmtime(chosen_path)
            except FileNotFoundError:
                mtime = time.time()
            with st.spinner(f"Loading {os.path.basename(chosen_path)}…"):
                df = load_from_path_cached(chosen_path, mtime)
        else:
            st.info(f"No files in {folder_abs}")

if df is None:
    st.stop()

# ============ Post-load sanitation & optimization ============

# If 'time' is not numeric (e.g., ISO strings), normalize to seconds since min
if "time" in df.columns and not np.issubdtype(df["time"].dtype, np.number):
    # First, try cheap numeric cast (many CSVs already numeric as strings)
    try:
        df["time"] = pd.to_numeric(df["time"], errors="raise")
    except Exception:
        ts = pd.to_datetime(df["time"], errors="coerce")
        t0 = ts.min()
        df["time"] = (ts - t0).dt.total_seconds().astype("float32")

# Downcast numerics (if not already) + categories for low-cardinality strings
for c in ("xloc_kf", "yloc_kf", "speed_kf", "acceleration_kf", "length_smoothed", "width_smoothed", "time"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
for c in ("id", "lane_kf", "run_index"):
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
for c in ("type_most_common", "av"):
    if c in df.columns:
        # to avoid huge category cardinality if dirty data, limit switch
        if df[c].dtype.name != "category":
            df[c] = df[c].astype("category")

missing = validate_columns(df)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# One-time presort used by multiple plots (dramatically reduces repeat sorts)
@st.cache_data(show_spinner=False)
def _presort_views(df_in: pd.DataFrame) -> dict:
    # Return multiple common sort orders (avoid re-sorting in each figure)
    return {
        "by_lane_id_time": df_in.sort_values(["lane_kf", "id", "time"], kind="mergesort"),
        "by_id_time": df_in.sort_values(["id", "time"], kind="mergesort"),
    }

views = _presort_views(df)

# ============ Sidebar filters & sampling ============
with st.sidebar:
    st.header("Filters")
    runs_all = sorted(df["run_index"].dropna().unique().tolist())
    lanes_all = sorted(df["lane_kf"].dropna().unique().tolist())
    types_all = sorted(df["type_most_common"].astype(str).unique().tolist())
    av_all = sorted(df["av"].astype(str).unique().tolist())
    
    # Initialize once: show only Run 1 on first render (if present)
    if "run_sel" not in st.session_state:
        st.session_state.run_sel = [1] if 1 in runs_all else runs_all
    else:
        # Keep session value valid if dataet changes (e.g., re-upload)
        st.session_state.run_sel = [
            r for r in st.session_state.run_sel if r in runs_all] or ([1] if 1 in runs_all else runs_all)
        
    # No "default=" when using a session_state-backed key
    run_sel = st.multiselect("Run index", runs_all, key="run_sel")
        
    lane_sel = st.multiselect("Lanes", lanes_all, default=lanes_all)
    type_sel = st.multiselect("Vehicle type", types_all, default=types_all)
    av_sel = st.multiselect("AV", av_all, default=av_all)

    st.header("Performance")
    max_plot_vehicles = st.slider(
        "Max vehicles to render (plots only)", 200, 5000, 1000, 100,
        help="Caps number of vehicles drawn for heavy plots. Analytics still use full filtered data.")
    
    flow_window = st.slider("Flow window (s)", 5, 120, 30, 5)
    time_bin = st.slider("Bin (s) for density & space-mean speed", 1, 120, 10, 1)
    
    st.header("Macroscopic Study Controls")

# Apply filters (vectorized) — we compute df_f before using bounds for macro controls
mask_global = (
    df["run_index"].isin(st.session_state.run_sel) &
    df["lane_kf"].isin(lanes_all) # temp placeholder; lane filter applied below with widget value
)
# Full filter with all widget selections happens after widgets; we need bounds first, so do a quick pass:
df_tmp = df.copy()

# Now finalize filtered df
mask = (
    df["run_index"].isin(run_sel) &
    df["lane_kf"].isin(lane_sel) &
    df["type_most_common"].astype(str).isin(type_sel) &
    df["av"].astype(str).isin(av_sel)
)
df_f = df.loc[mask]

if df_f.empty:
    st.warning("No data after filtering")
    st.stop()
    
# Determine spatial bounds for macro controls
x_min_all = float(np.nanmin(df_f["xloc_kf"]))
x_max_all = float(np.nanmax(df_f["xloc_kf"]))    

with st.sidebar:
    with st.expander("Select duration & location to study relationships", expanded=True):
        st.caption("Define two scenarios (A/B) to see how measurement duration and spatial location change the curves.")
        # Scenario A
        st.subheader("Scenario A")
        macro_bin_A = st.slider("Measurement duration A (s)", 5, 600, 60, 5, key="macro_bin_A")
        x_range_A = st.slider(
            "Location A: x-range (m)",
             min_value=float(x_min_all), max_value=float(x_max_all),
             value=(float(x_min_all), float(x_max_all)), step=1.0, key="x_range_A"
         )
        # Scenario B
        st.subheader("Scenario B")
        macro_bin_B = st.slider("Measurement duration B (s)", 5, 600, 120, 5, key="macro_bin_B")
        mid = (x_min_all + x_max_all) / 2.0
        default_B = (float(mid), float(x_max_all)) if (x_max_all - x_min_all) > 1 else (float(x_min_all), float(x_max_all))
        x_range_B = st.slider(
            "Location B: x-range (m)",
            min_value=float(x_min_all), max_value=float(x_max_all),
            value=default_B, step=1.0, key="x_range_B"
         )

# Sample vehicle IDs for plotting (NOT for analytics)
@st.cache_data(show_spinner=False)
def _sample_vehicle_ids(ids: np.ndarray, max_n: int, seed: int = 1337) -> np.ndarray:
    if len(ids) <= max_n:
        return ids
    rng = np.random.default_rng(seed)
    return rng.choice(ids, size=max_n, replace=False)

veh_ids = df_f["id"].dropna().unique().astype(np.int64, copy=False)
veh_ids_plot = _sample_vehicle_ids(veh_ids, max_plot_vehicles)
df_plot_subset = df_f[df_f["id"].isin(veh_ids_plot)]

# ============ Metric helpers (optimized) ============
@st.cache_data(show_spinner=False)
def compute_time_headways(df_in: pd.DataFrame) -> pd.Series:
    if df_in.empty:
        return pd.Series(dtype="float32")
    # First timestamp per vehicle, then diff
    first_t = df_in.groupby("id", sort=True, observed=True)["time"].min().sort_values(kind="mergesort").to_numpy()
    if first_t.size < 2:
        return pd.Series(dtype="float32")
    diffs = np.diff(first_t)
    return pd.Series(diffs[diffs > 0], dtype="float32")

@st.cache_data(show_spinner=False)
def compute_space_headways(df_in: pd.DataFrame, axis: str = "xloc_kf", time_bin: int = 10) -> pd.Series:
    if df_in.empty:
        return pd.Series(dtype="float32")
    tb = (np.floor(df_in["time"].to_numpy(dtype="float32") / time_bin) * time_bin).astype("int32")
    tmp = df_in.assign(_tb=tb)
    gaps_arr = []
    # groupby single key (fast); lane grouping optional but increases groups
    for _, g in tmp.groupby("_tb", sort=False, observed=True):
        vals = np.sort(g[axis].to_numpy(dtype="float32"))
        if vals.size >= 2:
            d = np.diff(vals)
            pos = d[d > 0]
            if pos.size:
                gaps_arr.append(pos)
    if not gaps_arr:
        return pd.Series(dtype="float32")
    return pd.Series(np.concatenate(gaps_arr), dtype="float32")

@st.cache_data(show_spinner=False)
def compute_flow(df_in: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame({"t": [], "flow": []})
    tb = (np.floor(df_in["time"].to_numpy(dtype="float32") / window) * window).astype("int32")
    tmp = df_in.assign(_tb=tb)
    res = tmp.groupby("_tb", observed=True)["id"].nunique().reset_index()
    res.columns = ["t", "flow"]
    return res

@st.cache_data(show_spinner=False)
def compute_density(df_in: pd.DataFrame, seg_length_m: float = 500.0, time_bin: int = 10) -> pd.DataFrame:
    if df_in.empty or seg_length_m <= 0:
        return pd.DataFrame({"t": [], "density": []})
    tb = (np.floor(df_in["time"].to_numpy(dtype="float32") / time_bin) * time_bin).astype("int32")
    tmp = df_in.assign(_tb=tb)
    cnt = tmp.groupby("_tb", observed=True)["id"].nunique().reset_index()
    cnt.columns = ["t", "veh"]
    cnt["density"] = cnt["veh"] / max(seg_length_m / 1000.0, 1e-6)
    return cnt[["t", "density"]]

@st.cache_data(show_spinner=False)
def compute_space_mean_speed(df_in: pd.DataFrame, time_bin: int = 10) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame({"t": [], "v": []})
    tb = (np.floor(df_in["time"].to_numpy(dtype="float32") / time_bin) * time_bin).astype("int32")
    tmp = df_in.assign(_tb=tb)
    def harmonic_mean(arr: np.ndarray) -> float:
        arr = np.clip(arr.astype("float32"), 1e-6, None)
        return float(len(arr) / np.sum(1.0 / arr))
    # Use numpy inside groupby-apply via to_numpy for speed
    v = tmp.groupby("_tb", observed=True)["speed_kf"].apply(lambda s: harmonic_mean(s.to_numpy()))
    res = v.reset_index()
    res.columns = ["t", "v"]
    return res


# === Macroscopic Relationships (speed-flow-density) ===
@st.cache_data(show_spinner=False)
def compute_macros(df_in: pd.DataFrame, x_min: float, x_max: float, bin_s: int) -> pd.DataFrame:
    """Aggregate within a spatial segment and duration to get (flow, density, sms)."""
    if df_in.empty or x_max <= x_min:
        return pd.DataFrame({"t": [], "flow_vph": [], "density_vpkm": [], "sms_mps": []})
    seg = df_in[(df_in["xloc_kf"] >= x_min) & (df_in["xloc_kf"] <= x_max)]
    if seg.empty:
        return pd.DataFrame({"t": [], "flow_vph": [], "density_vpkm": [], "sms_mps": []})

    tb = (np.floor(seg["time"].to_numpy(dtype="float32") / bin_s) * bin_s).astype("int32")
    tmp = seg.assign(_tb=tb)
    g = tmp.groupby("_tb", observed=True)

    # Flow: vehicles per hour (approximate using unique vehicles seen in the window)
    n = g["id"].nunique()
    flow_vph = n * (3600.0 / max(bin_s, 1e-6))

    # Density: vehicles per km within the chosen segment (approximate)
    length_km = max((x_max - x_min) / 1000.0, 1e-6)
    density_vpkm = n / length_km

    # Space-mean speed: harmonic mean of speeds
    def harmonic_mean_series(s: pd.Series) -> float:
        arr = np.clip(s.to_numpy(dtype="float32"), 1e-6, None)
        return float(len(arr) / np.sum(1.0 / arr))
    v_sms = g["speed_kf"].apply(harmonic_mean_series)

    out = pd.DataFrame({
        "t": n.index.values.astype("int32"),
        "flow_vph": flow_vph.astype("float32"),
        "density_vpkm": density_vpkm.astype("float32"),
        "sms_mps": v_sms.astype("float32"),
    })
    return out

# Compute scenarios A and B
macro_A = compute_macros(df_f, x_range_A[0], x_range_A[1], int(macro_bin_A))
macro_B = compute_macros(df_f, x_range_B[0], x_range_B[1], int(macro_bin_B))

# ============ Main layout ============
# Main title and caption
st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:3.0em; font-weight:bold;">🚦 Traffic Lens</h1>
<p style="font-size:0.9em; color:gray;">An interactive tool to explore large-scale TGSIM trajectory data, visualize traffic dynamics, and analyze flow–density–speed relationships.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("Main Trajectory Plot — Space–Time Diagram")
fig_main = px.line(
    views["by_lane_id_time"][views["by_lane_id_time"]["id"].isin(veh_ids_plot)],
    x="time", y="xloc_kf",
    color="lane_kf", line_group="id",
    hover_data=["id", "speed_kf", "lane_kf"],
    title="X vs Time"
)
fig_main.update_xaxes(title="Time [s]")
fig_main.update_yaxes(title="xloc_kf [m]")
st.plotly_chart(fig_main, use_container_width=True)

# Lane Occupancy vs Speed (full width)
st.subheader("Lane Occupancy vs Speed")
fig_occ = px.scatter(
    df_plot_subset, x="speed_kf", y="lane_kf",
    color="lane_kf", hover_data=["id", "time"],
    title="Lane Occupancy vs Speed"
)
fig_occ.update_xaxes(title="Speed [m/s]")
fig_occ.update_yaxes(title="Lane ID")
st.plotly_chart(fig_occ, use_container_width=True)

# Row: Speed vs Time + Speed Histogram
c1, c2 = st.columns(2)
with c1:
    fig_st = px.line(
        views["by_id_time"][views["by_id_time"]["id"].isin(veh_ids_plot)],
        x="time", y="speed_kf",
        color="id", line_group="id",
        hover_data=["lane_kf"],
        title="Speed vs Time"
    )
    fig_st.update_xaxes(title="Time [s]")
    fig_st.update_yaxes(title="Speed [m/s]")
    st.plotly_chart(fig_st, use_container_width=True)
with c2:
    # Histogram can use all filtered data (cheap) or sampled; keep all for fidelity
    fig_hist = px.histogram(df_f, x="speed_kf", nbins=50, title="Speed Histogram")
    fig_hist.update_xaxes(title="Speed [m/s]")
    st.plotly_chart(fig_hist, use_container_width=True)

# Row: Speed Distribution by Lane + Time Headways
c3, c4 = st.columns(2)
with c3:
    # Box on all filtered data (uses quantiles; still cheap)
    fig_sd = px.box(df_f, x="lane_kf", y="speed_kf", points=False, title="Speed Distribution by Lane")
    fig_sd.update_yaxes(title="Speed [m/s]")
    st.plotly_chart(fig_sd, use_container_width=True)
with c4:
    hw = compute_time_headways(df_f)
    if not hw.empty:
        fig_hw = px.histogram(hw, x=hw, nbins=60, title="Time Headway Distribution")
        fig_hw.update_xaxes(title="Headway [s]")
        st.plotly_chart(fig_hw, use_container_width=True)
    else:
        st.info("Not enough vehicles for headway calc")

# Row: Flow + Density + Space-mean speed (full filtered data, cached)
st.subheader("Flow, Density, and Space-Mean Speed")
c5, c6, c7 = st.columns(3)
with c5:
    flow = compute_flow(df_f, window=flow_window)
    fig_flow = px.line(flow, x="t", y="flow", title=f"Flow (veh / {flow_window}s window)")
    fig_flow.update_xaxes(title="Time [s]")
    fig_flow.update_yaxes(title=f"Flow [veh / {flow_window}s]")
    st.plotly_chart(fig_flow, use_container_width=True)
with c6:
    dens = compute_density(df_f, seg_length_m=500.0, time_bin=time_bin)
    fig_dens = px.line(dens, x="t", y="density", title=f"Density (veh/km, bin={time_bin}s)")
    fig_dens.update_xaxes(title="Time [s]")
    fig_dens.update_yaxes(title="Density [veh/km]")
    st.plotly_chart(fig_dens, use_container_width=True)
with c7:
    sms = compute_space_mean_speed(df_f, time_bin=time_bin)
    fig_sms = px.line(sms, x="t", y="v", title=f"Space-Mean Speed (bin={time_bin}s)")
    fig_sms.update_xaxes(title="Time [s]")
    fig_sms.update_yaxes(title="Speed [m/s]")
    st.plotly_chart(fig_sms, use_container_width=True)


# == Assignment 2 ==
st.subheader("Macroscopic Relationships - Impact of Duration & Location")

# Flow-Density
fig_fd = go.Figure()
if not macro_A.empty:
    fig_fd.add_trace(go.Scatter(
        x=macro_A["density_vpkm"], y=macro_A["flow_vph"],
        mode="markers", name=f"A: {int(macro_bin_A)}s, x∈[{x_range_A[0]:.0f},{x_range_A[1]:.0f}] m",
        hovertemplate="density=%{x:.2f} veh/km<br>flow=%{y:.0f} veh/h<extra></extra>"
    ))

if not macro_B.empty:
    fig_fd.add_trace(go.Scatter(
        x=macro_B["density_vpkm"], y=macro_B["flow_vph"], mode="markers",
        name=f"B: {int(macro_bin_B)}s, x∈[{x_range_B[0]:.0f},{x_range_B[1]:.0f}] m",
        hovertemplate="density=%{x:.2f} veh/km<br>flow=%{y:.0f} veh/h<extra></extra>"
    ))
fig_fd.update_layout(title="Flow–Density", xaxis_title="Density [veh/km]", yaxis_title="Flow [veh/h]")

# Speed–Density
fig_sd_rel = go.Figure()
if not macro_A.empty:
    fig_sd_rel.add_trace(go.Scatter(
        x=macro_A["density_vpkm"], y=macro_A["sms_mps"], mode="markers",
        name=f"A: {int(macro_bin_A)}s, x∈[{x_range_A[0]:.0f},{x_range_A[1]:.0f}] m",
        hovertemplate="density=%{x:.2f} veh/km<br>v_sms=%{y:.2f} m/s<extra></extra>"    
    ))
if not macro_B.empty:
    fig_sd_rel.add_trace(go.Scatter(
        x=macro_B["density_vpkm"], y=macro_B["sms_mps"], mode="markers",
        name=f"B: {int(macro_bin_B)}s, x∈[{x_range_B[0]:.0f},{x_range_B[1]:.0f}] m",
        hovertemplate="density=%{x:.2f} veh/km<br>v_sms=%{y:.2f} m/s<extra></extra>"
    ))
fig_sd_rel.update_layout(title="Speed–Density", xaxis_title="Density [veh/km]", yaxis_title="Space-Mean Speed [m/s]")

# Speed–Flow
fig_sf = go.Figure()
if not macro_A.empty:
    fig_sf.add_trace(go.Scatter(
        x=macro_A["flow_vph"], y=macro_A["sms_mps"], mode="markers",
        name=f"A: {int(macro_bin_A)}s, x∈[{x_range_A[0]:.0f},{x_range_A[1]:.0f}] m",
        hovertemplate="flow=%{x:.0f} veh/h<br>v_sms=%{y:.2f} m/s<extra></extra>"
    ))
if not macro_B.empty:
    fig_sf.add_trace(go.Scatter(
        x=macro_B["flow_vph"], y=macro_B["sms_mps"], mode="markers",
        name=f"B: {int(macro_bin_B)}s, x∈[{x_range_B[0]:.0f},{x_range_B[1]:.0f}] m",
        hovertemplate="flow=%{x:.0f} veh/h<br>v_sms=%{y:.2f} m/s<extra></extra>"
    ))
fig_sf.update_layout(title="Speed–Flow", xaxis_title="Flow [veh/h]", yaxis_title="Space-Mean Speed [m/s]")


cA, cB, cC = st.columns(3)
with cA:
    st.plotly_chart(fig_fd, use_container_width=True)
with cB:
    st.plotly_chart(fig_sd_rel, use_container_width=True)
with cC:
    st.plotly_chart(fig_sf, use_container_width=True)

# Preview
with st.expander("Preview filtered data"):
    st.dataframe(df_f.head(50), use_container_width=True)