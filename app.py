import io
import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
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


# ---------- Data loading ----------
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


# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    runs = sorted(df["run_index"].unique().tolist())
    run_sel = st.multiselect("Run index", runs, default=runs)

    lanes = sorted(df["lane_kf"].unique().tolist())
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

    st.header("Flow window")
    flow_window = st.slider("Window (seconds)", min_value=5, max_value=120, value=30, step=5)


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


# ---------- Helper: compute flow ----------
def compute_flow(df_in: pd.DataFrame, window_s: int) -> pd.DataFrame:
    """Approximate flow as unique vehicles per non-overlapping window."""
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


# ---------- UI: Plot picker ----------
plot_choice = st.selectbox(
    "Choose a view",
    [
        "Trajectory (x vs y)",
        "Speed vs Time",
        "Lane Occupancy vs Time",
        "Acceleration Histogram",
        "Flow (vehicles / window)"
    ],
    index=0
)


# ---------- Plots ----------
if df_f.empty:
    st.warning("No data after filters.")
    st.stop()

if plot_choice == "Trajectory (x vs y)":
    title = "Vehicle Trajectories (xloc_kf vs yloc_kf)"
    fig = px.line(
        df_f.sort_values(["id", "time"]),
        x="xloc_kf", y="yloc_kf",
        color="id",
        line_group="id",
        hover_data=["time", "speed_kf", "lane_kf", "type_most_common", "av"],
        title=title
    )
    fig.update_layout(legend_title_text="Vehicle ID")
    st.plotly_chart(fig, use_container_width=True)

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
    fig.update_yaxes(title="Speed")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Lane Occupancy vs Time":
    jitter = (np.random.rand(len(df_f)) - 0.5) * 0.1
    tmp = df_f.assign(_lane_jitter=df_f["lane_kf"].astype(float).to_numpy() + jitter)
    fig = px.scatter(
        tmp, x="time", y="_lane_jitter",
        color="id",
        hover_data=["lane_kf", "speed_kf", "type_most_common", "av"],
        size_max=6, opacity=0.8,
        title="Lane Occupancy Over Time"
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
    fig.update_xaxes(title="Acceleration")
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Flow (vehicles / window)":
    flow = compute_flow(df_f[["time", "id"]].drop_duplicates(), flow_window)
    if flow.empty:
        st.info("No flow to display with current filters.")
    else:
        fig = px.bar(
            flow, x="window_start", y="flow",
            title=f"Traffic Flow (unique vehicles / {flow_window}s)"
        )
        fig.update_xaxes(title="Window start [s]")
        fig.update_yaxes(title="Vehicles per window")
        st.plotly_chart(fig, use_container_width=True)


# ---------- Data preview ----------
with st.expander("Preview filtered data"):
    st.dataframe(df_f.head(50), use_container_width=True)