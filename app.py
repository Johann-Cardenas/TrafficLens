import io
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TrafficLens — TGSIM Explorer", layout="wide")

# ---------- Required columns ----------
REQ_COLS = {
    "id", "time", "xloc_kf", "yloc_kf", "lane_kf",
    "speed_kf", "acceleration_kf", "length_smoothed",
    "width_smoothed", "type_most_common", "av", "run_index"
}

def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return missing required columns."""
    return sorted(list(REQ_COLS.difference(df.columns)))


# ---------- Data loading (Cached) ----------
@st.cache_data(show_spinner=False)
def _read_file(file_like_or_path, name: str) -> pd.DataFrame:
    """Load CSV/Parquet from a BytesIO, file handle, or filesystem path."""
    name_l = name.lower()
    if isinstance(file_like_or_path, (str, os.PathLike)):
        if name_l.endswith(".csv"):
            return pd.read_csv(file_like_or_path)
        elif name_l.endswith((".parquet", ".pq")):
            return pd.read_parquet(file_like_or_path)
        else:
            return pd.read_csv(file_like_or_path)  # fallback
    if name_l.endswith(".csv"):
        return pd.read_csv(file_like_or_path)
    if name_l.endswith((".parquet", ".pq")):
        return pd.read_parquet(file_like_or_path)
    return pd.read_csv(file_like_or_path)  # fallback

def _norm_path(p: str) -> str:
    """Expand ~ and make absolute, allow relative to app.py for ./data."""
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.expanduser(p.strip())
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(base, p))
    return p


@st.cache_data(show_spinner=False)
def list_dataset_files(folder_abs: str) -> list[tuple[str, str]]:
    """Return [(display, fullpath), ...] for csv/parquet files in folder."""
    if not os.path.isdir(folder_abs):
        return []
    out = []
    for fn in sorted(os.listdir(folder_abs)):
        if fn.lower().endswith((".csv", ".parquet", ".pq")):
            out.append((fn, os.path.join(folder_abs, fn)))
    return out

@st.cache_data(show_spinner=True)
def load_from_path_cached(path_abs: str, _mtime_key: float) -> pd.DataFrame:
    """Cached by file content mtime; change file -> cache invalidates."""
    return _read_file(path_abs, path_abs)

def _safe_to_numeric(s: pd.Series, downcast: str | None = None) -> pd.Series:
    try:
        return pd.to_numeric(s, downcast=downcast)  # default errors='raise'
    except Exception:
        return s  # mimic old errors='ignore' behavior

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Memory-friendly dtypes for big TGSIM files."""
    num_float_cols = [
        "xloc_kf", "yloc_kf", "speed_kf", "acceleration_kf",
        "length_smoothed", "width_smoothed", "time"
    ]
    for c in num_float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")

    int_like = ["lane_kf", "run_index", "id"]
    for c in int_like:
        if c in df.columns:
            df[c] = _safe_to_numeric(df[c], downcast="integer")

    for c in ["type_most_common", "av"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype("category")

    return df


# ---------- Sidebar: choose source ----------
with st.sidebar:
    st.header("Data")
    source = st.radio("Choose data source", ["Upload", "From folder"], horizontal=True)
    df = None

    if source == "Upload":
        up = st.file_uploader("Upload TGSIM trajectory file", type=["csv", "parquet", "pq"])
        if up is not None:
            bio = io.BytesIO(up.getvalue())
            with st.spinner(f"Loading {up.name}…"):
                df = _read_file(bio, up.name)

    else:  # From folder
        folder_input = st.text_input("Folder path", value="./data/", help="Relative or absolute path")
        folder_abs = _norm_path(folder_input)
        files = list_dataset_files(folder_abs)
        if not files:
            st.info(f"No CSV/Parquet files found in: `{folder_abs}`")
        else:
            choice = st.selectbox("Pick a file", [f[0] for f in files], index=0)
            chosen_path = dict(files)[choice]
            try:
                mtime = os.path.getmtime(chosen_path)
            except FileNotFoundError:
                mtime = time.time()
            with st.spinner(f"Loading {os.path.basename(chosen_path)}…"):
                df = load_from_path_cached(chosen_path, mtime)


# ---------- Guard ----------
if df is None:
    st.info("Provide data via **Upload** or **From folder** to continue.")
    st.stop()

# ---- Minimal sanitation ----
if "time" in df.columns and not np.issubdtype(df["time"].dtype, np.number):
    try:
        ts = pd.to_datetime(df["time"])
        df["time"] = (ts - ts.min()).dt.total_seconds()
    except Exception:
        st.error("Column 'time' must be numeric seconds or parseable to timestamps.")
        st.stop()

missing = validate_columns(df)
if missing:
    st.error(f"Missing required columns: {missing}")
    st.dataframe(df.head(20))
    st.stop()

# Optimize dtypes (memory win for big files)
df = optimize_dtypes(df)
mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
st.caption(f"Loaded rows: {len(df):,} | Memory usage: {mem_mb:.1f} MB")


# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    runs = sorted(pd.Series(df["run_index"]).dropna().unique().tolist())
    run_sel = st.multiselect("Run index", runs, default=runs)

    lanes = sorted(pd.Series(df["lane_kf"]).dropna().unique().tolist())
    lane_sel = st.multiselect("Lanes", lanes, default=lanes)

    types = sorted(df["type_most_common"].astype(str).unique().tolist())
    type_sel = st.multiselect("Vehicle type", types, default=types)

    av_vals = sorted(df["av"].astype(str).unique().tolist())
    av_sel = st.multiselect("AV", av_vals, default=av_vals)

    some_ids = df["id"].dropna().unique()
    show_id_picker = st.checkbox("Filter by specific vehicle IDs", value=False)
    id_sel = []
    if show_id_picker:
        id_choices = sorted(pd.unique(some_ids[:2000]).tolist())
        id_sel = st.multiselect("Vehicle IDs (first 2000 shown)", id_choices)

    st.header("Analysis windows")
    flow_window = st.slider("Flow window (s)", min_value=5, max_value=120, value=30, step=5)
    time_bin = st.slider("Bin (s) for density & space-mean speed", 1, 120, 10, 1)

    st.header("Region of Interest (ROI)")
    # Determine bounds once (from the unfiltered df for a stable UI range)
    x_min, x_max = float(df["xloc_kf"].min()), float(df["xloc_kf"].max())
    y_min, y_max = float(df["yloc_kf"].min()), float(df["yloc_kf"].max())
    roi_x = st.slider("X-range (m)", min_value=x_min, max_value=x_max,
                      value=(x_min, x_max),
                      step=float(max(1.0, (x_max - x_min) / 500)))
    roi_y = st.slider("Y-range (m)", min_value=y_min, max_value=y_max,
                      value=(y_min, y_max),
                      step=float(max(1.0, (y_max - y_min) / 500)))
    longitudinal_axis = st.radio("Longitudinal axis for headway/density",
                                 ["yloc_kf", "xloc_kf"], horizontal=True)


# ---------- Apply filters ----------
mask = (
    df["run_index"].isin(run_sel) &
    df["lane_kf"].isin(lane_sel) &
    df["type_most_common"].astype(str).isin(type_sel) &
    df["av"].astype(str).isin(av_sel)
)
if show_id_picker and id_sel:
    mask &= df["id"].isin(id_sel)

df_f = df.loc[mask].copy()
st.caption(f"Filtered rows: {len(df_f):,}")

# Constrain to ROI (used by many metrics)
in_roi = (
    df_f["xloc_kf"].between(roi_x[0], roi_x[1]) &
    df_f["yloc_kf"].between(roi_y[0], roi_y[1])
)
df_roi = df_f.loc[in_roi].copy()

# ROI length along chosen longitudinal axis
seg_len_m = max(1e-6, (roi_x[1] - roi_x[0])) if longitudinal_axis == "xloc_kf" else max(1e-6, (roi_y[1] - roi_y[0]))
st.caption(f"ROI rows: {len(df_roi):,} | ROI length along {longitudinal_axis} ≈ {seg_len_m:.1f} m")

# ---------- Helpers for metrics ----------
def compute_flow(df_in: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Approximate flow as unique vehicles per non-overlapping time window."""
    if df_in.empty:
        return pd.DataFrame({"window_start": [], "flow": []})
    bucket = (np.floor(df_in["time"].to_numpy() / window_s) * window_s).astype(int)
    tmp = df_in.assign(_bucket=bucket)
    flow = (
        tmp.groupby("_bucket")["id"]
           .nunique()
           .rename("flow")
           .reset_index()
           .rename(columns={"_bucket": "window_start"})
           .sort_values("window_start")
    )
    return flow

def compute_time_headways(df_in: pd.DataFrame) -> pd.Series:
    """
    Time headway (s): consecutive entry times of unique vehicles into ROI.
    We define 'entry time' as the first time a vehicle is observed within ROI
    per (run, lane).
    """
    if df_in.empty:
        return pd.Series(dtype=float)

    g = (
        df_in.sort_values(["run_index", "lane_kf", "id", "time"])
             .groupby(["run_index", "lane_kf", "id"], as_index=False)
             .agg(entry_time=("time", "first"))
    )
    g = g.sort_values(["run_index", "lane_kf", "entry_time"])
    hw = g.groupby(["run_index", "lane_kf"])["entry_time"].diff().dropna()
    return hw[hw > 0]


def compute_space_headways(df_in: pd.DataFrame, time_bin: int, axis: str) -> pd.Series:
    """
    Space headway (m): within each time bin, sort vehicles along longitudinal
    axis (x or y) and take neighbor gaps.
    """
    if df_in.empty:
        return pd.Series(dtype=float)

    tbucket = (np.floor(df_in["time"].to_numpy() / time_bin) * time_bin).astype(int)
    tmp = df_in.assign(_tb=tbucket)

    gaps = []
    coord = tmp[axis].astype(float).to_numpy()
    for (_, _), grp in tmp.groupby(["run_index", "lane_kf", "_tb"]):
        vals = np.sort(grp[axis].astype(float).to_numpy())
        if vals.size >= 2:
            g = np.diff(vals)
            if g.size:
                gaps.append(g[g > 0])
    if not gaps:
        return pd.Series(dtype=float)
    return pd.Series(np.concatenate(gaps), dtype=float)


def compute_speed_histogram_data(df_in: pd.DataFrame) -> pd.Series:
    """Raw speeds within ROI for histogram."""
    if df_in.empty:
        return pd.Series(dtype=float)
    return df_in["speed_kf"].dropna().astype(float)


def compute_space_mean_speed(df_in: pd.DataFrame, time_bin: int) -> pd.DataFrame:
    """
    Space-mean speed (m/s): harmonic mean across vehicles present in ROI per time bin.
    """
    if df_in.empty:
        return pd.DataFrame({"t": [], "v_space_mean": []})

    tbucket = (np.floor(df_in["time"].to_numpy() / time_bin) * time_bin).astype(int)
    tmp = df_in.assign(_tb=tbucket)

    def harmonic_mean(s):
        s = s.clip(lower=1e-6)
        return len(s) / np.sum(1.0 / s)

    sm = (
        tmp.groupby("_tb")["speed_kf"]
           .apply(lambda s: harmonic_mean(s.astype(float)))
           .rename("v_space_mean")
           .reset_index()
           .rename(columns={"_tb": "t"})
           .sort_values("t")
    )
    return sm


def compute_density(df_in: pd.DataFrame, time_bin: int, seg_length_m: float) -> pd.DataFrame:
    """
    Density (veh/km): count unique vehicles in ROI per time bin divided by segment length (km).
    """
    if df_in.empty or seg_length_m <= 0:
        return pd.DataFrame({"t": [], "density_vpk": []})

    tbucket = (np.floor(df_in["time"].to_numpy() / time_bin) * time_bin).astype(int)
    tmp = df_in.assign(_tb=tbucket)
    cnt = (
        tmp.groupby("_tb")["id"]
           .nunique()
           .rename("veh")
           .reset_index()
           .rename(columns={"_tb": "t"})
           .sort_values("t")
    )
    seg_km = seg_length_m / 1000.0
    cnt["density_vpk"] = cnt["veh"] / max(seg_km, 1e-6)
    return cnt[["t", "density_vpk"]]


# ---------- Views ----------
plot_choice = st.selectbox(
    "Choose a view",
    [
        "Trajectory (x vs y)",
        "Speed vs Time",
        "Speed Histogram",
        "Lane Occupancy vs Time",
        "Acceleration Histogram",
        "Flow (vehicles / window)",
        "Time Headway Histogram (ROI)",
        "Space Headway Histogram (ROI)",
        "Space-Mean Speed (ROI, time series + dist.)",
        "Density (ROI, time series)"
    ],
    index=0
)

if df_f.empty:
    st.warning("No data after filters.")
    st.stop()

# ---------- Plots ----------
if plot_choice == "Trajectory (x vs y)":
    title = "Vehicle Trajectories (xloc_kf vs yloc_kf)"
    df_plot = df_f.sort_values(["id", "time"])
    fig = px.line(
        df_plot, x="xloc_kf", y="yloc_kf",
        color="id", line_group="id",
        hover_data=["time", "speed_kf", "lane_kf", "type_most_common", "av"],
        title=title
    )
    # Overlay ROI rectangle for visual guidance
    fig.add_shape(
        type="rect",
        x0=roi_x[0], x1=roi_x[1],
        y0=roi_y[0], y1=roi_y[1],
        line=dict(width=2, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
        layer="above"
    )
    fig.update_layout(legend_title_text="Vehicle ID")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Use Plotly's zoom/box-select tools in the toolbar to explore regions. ROI sliders drive the analytics below.")

elif plot_choice == "Speed vs Time":
    fig = px.line(
        df_f.sort_values(["id", "time"]),
        x="time", y="speed_kf",
        color="id",
        line_group="id",
        hover_data=["lane_kf", "type_most_common", "av"],
        title="Speed Over Time"
    )
    fig.update_xaxes(title="Time [s]")
    fig.update_yaxes(title="Speed [m/s]")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Speed Histogram":
    sp = compute_speed_histogram_data(df_roi)
    if sp.empty:
        st.info("No speeds in ROI with current filters.")
    else:
        fig = px.histogram(sp, x=sp, nbins=50, title="Speed Distribution (ROI)")
        fig.update_xaxes(title="Speed [m/s]")
        st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Lane Occupancy vs Time":
    jitter = (np.random.rand(len(df_f)) - 0.5) * 0.1
    tmp = df_f.assign(_lane_jitter=df_f["lane_kf"].astype(float).to_numpy() + jitter)
    fig = px.scatter(
        tmp, x="time", y="_lane_jitter",
        color="id",
        hover_data=["lane_kf", "speed_kf", "type_most_common", "av"],
        size_max=6, opacity=0.8,
        title="Lane Occupancy Over Time (jittered per vehicle)"
    )
    fig.update_yaxes(title="Lane (jittered)")
    fig.update_xaxes(title="Time [s]")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Acceleration Histogram":
    fig = px.histogram(
        df_f, x="acceleration_kf", nbins=50,
        title="Distribution of Accelerations",
        marginal="rug"
    )
    fig.update_xaxes(title="Acceleration [m/s²]")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Flow (vehicles / window)":
    flow = compute_flow(df_roi[["time", "id"]].drop_duplicates(), flow_window)
    if flow.empty:
        st.info("No flow to display with current filters/ROI.")
    else:
        fig = px.bar(
            flow, x="window_start", y="flow",
            title=f"Traffic Flow in ROI (unique vehicles / {flow_window}s)"
        )
        fig.update_xaxes(title="Window start [s]")
        fig.update_yaxes(title="Vehicles per window")
        st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Time Headway Histogram (ROI)":
    cols = ["run_index", "lane_kf", "id", "time"]
    hw = compute_time_headways(df_roi[cols].dropna())
    if hw.empty:
        st.info("No headways computed — widen ROI or adjust filters.")
    else:
        fig = px.histogram(hw, x=hw, nbins=60, title="Time Headway Distribution (ROI)")
        fig.update_xaxes(title="Headway [s]")
        st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Space Headway Histogram (ROI)":
    cols = ["run_index", "lane_kf", "time", longitudinal_axis]
    sh = compute_space_headways(df_roi[cols].dropna(), time_bin, longitudinal_axis)
    if sh.empty:
        st.info("No space headways computed — widen ROI or adjust filters.")
    else:
        fig = px.histogram(sh, x=sh, nbins=60, title=f"Space Headway Distribution (ROI along {longitudinal_axis})")
        fig.update_xaxes(title="Space headway [m]")
        st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Space-Mean Speed (ROI, time series + dist.)":
    sm = compute_space_mean_speed(df_roi[["time", "speed_kf"]].dropna(), time_bin)
    if sm.empty:
        st.info("No data for space-mean speed.")
    else:
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            fig_ts = px.line(sm, x="t", y="v_space_mean",
                             title=f"Space-Mean Speed in ROI (bin={time_bin}s)")
            fig_ts.update_xaxes(title="Time [s]")
            fig_ts.update_yaxes(title="Space-mean speed [m/s]")
            st.plotly_chart(fig_ts, use_container_width=True)
        with c2:
            fig_hist = px.histogram(sm["v_space_mean"], x=sm["v_space_mean"], nbins=40,
                                    title="Distribution")
            fig_hist.update_xaxes(title="Space-mean speed [m/s]")
            st.plotly_chart(fig_hist, use_container_width=True)

elif plot_choice == "Density (ROI, time series)":
    den = compute_density(df_roi[["time", "id"]].drop_duplicates(), time_bin, seg_len_m)
    if den.empty:
        st.info("No density data.")
    else:
        fig = px.line(den, x="t", y="density_vpk",
                      title=f"Density in ROI (veh/km, bin={time_bin}s)")
        fig.update_xaxes(title="Time [s]")
        fig.update_yaxes(title="Density [veh/km]")
        st.plotly_chart(fig, use_container_width=True)


# ---------- Data preview ----------
with st.expander("Preview filtered data"):
    st.dataframe(df_f.head(50), use_container_width=True)