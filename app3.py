import io
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================
# ---- App configuration ----
# ==========================
st.set_page_config(page_title="TrafficLens — CTM (Assignment 3)", layout="wide")

# Persist last simulation & signature so UI controls (like N-curves) don't force re-runs
if "sim" not in st.session_state:
    st.session_state.sim = None
if "sim_sig" not in st.session_state:
    st.session_state.sim_sig = None
if "up_off_m" not in st.session_state:
    st.session_state.up_off_m = 800
if "dn_off_m" not in st.session_state:
    st.session_state.dn_off_m = 800

# ==========================
# ---- Unit helpers ---------
# ==========================
MPH_TO_MPS = 0.44704
MILE_TO_M = 1609.344

def mph_to_mps(v_mph: float) -> float:
    return v_mph * MPH_TO_MPS

def mps_to_mph(v_mps: float) -> float:
    return v_mps / MPH_TO_MPS

def vehph_to_vehps(q_vph: float | np.ndarray) -> np.ndarray:
    return np.asarray(q_vph, dtype="float64") / 3600.0

def vehps_to_vehph(q_vps: np.ndarray) -> np.ndarray:
    return np.asarray(q_vps, dtype="float64") * 3600.0

def vehpkm_to_vehpm(k_vpkm: float | np.ndarray) -> np.ndarray:
    return np.asarray(k_vpkm, dtype="float64") / 1000.0

def vehpm_to_vehpkm(k_vpm: np.ndarray) -> np.ndarray:
    return np.asarray(k_vpm, dtype="float64") * 1000.0

# ==========================
# ---- Triangular FD math ---
# ==========================
def fd_kcrit(vf_mps: float, w_mps: float, kj_per_lane_vpkm: float) -> float:
    # k_c = (w * k_j) / (v_f + w)
    return (w_mps * kj_per_lane_vpkm) / (vf_mps + w_mps)

def fd_qmax_per_lane(vf_mps: float, w_mps: float, kj_per_lane_vpkm: float) -> float:
    # q_max [veh/h/ln] = v_f[km/h] * k_c[veh/km/ln]
    vf_kmh = vf_mps * 3.6
    k_c = fd_kcrit(vf_mps, w_mps, kj_per_lane_vpkm)
    return vf_kmh * k_c

def fd_demand(vf_mps: float, qcap_cell_vps: np.ndarray, k_cell_vpm: np.ndarray) -> np.ndarray:
    # S = min(v_f * k, Q_cap)
    return np.minimum(vf_mps * k_cell_vpm, qcap_cell_vps)

def fd_supply(w_mps: float, qcap_cell_vps: np.ndarray, kj_cell_vpm: np.ndarray, k_cell_vpm: np.ndarray) -> np.ndarray:
    # R = min(w * (k_j - k), Q_cap)
    supply = w_mps * np.maximum(kj_cell_vpm - k_cell_vpm, 0.0)
    return np.minimum(supply, qcap_cell_vps)

# ==========================
# ---- Demand profiles -------
# ==========================
def build_piecewise_demand(steps: int, dt: float, segments_min: list[float], demands_vph: list[float]) -> np.ndarray:
    """Piecewise-constant total upstream demand [veh/s] over T=steps*dt."""
    assert len(segments_min) == len(demands_vph)
    T = steps * dt
    t_edges = np.cumsum([m * 60.0 for m in segments_min])
    profile = np.zeros(steps, dtype="float32")
    t0 = 0.0
    for edge, q_vph in zip(t_edges, demands_vph):
        i0 = int(np.floor(t0 / dt))
        i1 = int(min(steps, np.floor(edge / dt)))
        profile[i0:i1] = vehph_to_vehps(q_vph)
        t0 = edge
    if t0 < T:
        profile[int(np.floor(t0 / dt)):] = vehph_to_vehps(demands_vph[-1])
    return profile

def resample_csv_demand_to_steps(df: pd.DataFrame, steps: int, dt: float) -> np.ndarray:
    """Expect columns: time_s, demand_vph (total). Step-hold to next change."""
    d = df.copy()
    if not {"time_s", "demand_vph"}.issubset(d.columns):
        raise ValueError("CSV must have columns: time_s, demand_vph")
    d = d.sort_values("time_s")
    t_grid = np.arange(steps) * dt
    demand_vph = np.interp(t_grid, d["time_s"].to_numpy(), d["demand_vph"].to_numpy(),
                           left=d["demand_vph"].iloc[0], right=d["demand_vph"].iloc[-1])
    return vehph_to_vehps(demand_vph).astype("float32")

# ==========================
# ---- Heatmap prettifiers ---
# ==========================
def _nice_dtick_minutes(horizon_min: float) -> float:
    if horizon_min <= 30: return 5
    if horizon_min <= 60: return 10
    if horizon_min <= 120: return 15
    return 20

def _space_dtick_miles(L_miles: float) -> float:
    if L_miles <= 2.5: return 0.1
    if L_miles <= 6:   return 0.25
    return 0.5

def _heatmap(z, x_min, y_mi, title, cbar_title, unit, bn_y_mi=None,
             dtick_min=10, dtick_space_mi=0.25, z_q=(1, 99)):
    zmin, zmax = (float(np.nanpercentile(z, z_q[0])), float(np.nanpercentile(z, z_q[1])))
    fig = px.imshow(
        np.asarray(z).T,
        origin="lower",
        aspect="auto",
        x=x_min,
        y=y_mi,
        zmin=zmin, zmax=zmax,
        color_continuous_scale="Jet",
        labels=dict(x="Time [min]", y="Distance [mi]", color=cbar_title),
    )
    fig.update_traces(hovertemplate=f"t=%{{x:.1f}} min<br>x=%{{y:.2f}} mi<br>{unit}=%{{z:.2f}}<extra></extra>")
    fig.update_layout(
        title=title,
        margin=dict(l=70, r=10, t=50, b=50),
        coloraxis_colorbar=dict(title=cbar_title, titleside="right", ticks="outside", ticklen=4),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.25)", zeroline=False,
                     ticks="outside", ticklen=5, dtick=dtick_min)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.25)", zeroline=False,
                     ticks="outside", ticklen=5, dtick=dtick_space_mi)
    if bn_y_mi is not None:
        fig.add_hline(y=bn_y_mi, line_dash="dash", line_color="black", line_width=1)
        fig.add_annotation(x=x_min[len(x_min)//2], y=bn_y_mi, text="Bottleneck", showarrow=False,
                           bgcolor="rgba(255,255,255,0.65)", font=dict(size=12))
    return fig

# ==========================
# ---- Small utilities -------
# ==========================
def _runs_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous True-runs as (start_idx, end_idx)."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)
    return [(int(g[0]), int(g[-1])) for g in groups]

def _sim_signature(**params) -> tuple:
    """Hashable signature of CTM inputs that change model state."""
    return (
        params["L_m"], params["dx_m"], params["dt_s"], params["steps"],
        params["vf_mps"], params["w_mps"],
        params["kj_lane_vpkm"], params["qmax_lane_vps"],
        params["bn_pos_m"], params["cap_drop_factor"],
        tuple(np.asarray(params["demand_total_vps"]).round(6)),
        params["init_mode"], round(float(params["init_value_vpkm_per_lane"]), 6),
        params["per_lane_outputs"],
    )

# ==========================
# ---- CTM Simulation --------
# ==========================
@st.cache_data(show_spinner=True)
def run_ctm(
    L_m: float,
    dx_m: float,
    dt_s: float,
    steps: int,
    vf_mps: float,
    w_mps: float,
    kj_lane_vpkm: float,
    qmax_lane_vps: float,
    bn_pos_m: float,
    cap_drop_factor: float,
    demand_total_vps: np.ndarray,
    init_mode: str,
    init_value_vpkm_per_lane: float,
    per_lane_outputs: bool = True,
):
    # Spatial grid
    N = max(2, int(L_m // dx_m))
    dx_eff = L_m / N
    x_cell_mid = (np.arange(N) + 0.5) * dx_eff
    x_interfaces = np.arange(N + 1) * dx_eff

    # Lane drop at interface bn_idx (2 lanes upstream, 1 downstream)
    bn_idx = int(np.clip(np.floor(bn_pos_m / dx_eff), 1, N - 1))
    lanes = np.where(x_cell_mid < x_interfaces[bn_idx], 2, 1).astype(int)

    # Per-cell jam density & capacity (totals)
    kj_cell_vpkm = kj_lane_vpkm * lanes.astype("float32")
    kj_cell_vpm = vehpkm_to_vehpm(kj_cell_vpkm)
    qcap_cell_vps = qmax_lane_vps * lanes.astype("float32")

    # Interface capacity (∞ except at bottleneck interface)
    cap_interface_vps = np.full(N + 1, np.inf, dtype="float32")
    cap_interface_vps[bn_idx] = float(cap_drop_factor * 2.0 * qmax_lane_vps)

    # Initial densities (veh/m total)
    k0_vpkm_per_lane = {
        "free": 0.05 * kj_lane_vpkm,
        "capacity": fd_kcrit(vf_mps, w_mps, kj_lane_vpkm),
        "near_jam": 0.90 * kj_lane_vpkm,
        "custom": max(1e-6, init_value_vpkm_per_lane),
    }[init_mode]
    k0_total_vpkm = k0_vpkm_per_lane * lanes.astype("float32")
    k_vpm = np.repeat(vehpkm_to_vehpm(k0_total_vpkm)[None, :], steps, axis=0)  # (T,N)

    f_vps = np.zeros((steps, N + 1), dtype="float32")  # interface flows
    q_cell_vps = np.zeros((steps, N), dtype="float32")
    v_cell_mps = np.zeros((steps, N), dtype="float32")
    S = np.zeros(N, dtype="float32")
    R = np.zeros(N, dtype="float32")

    lam = dt_s / dx_eff
    for t in range(steps):
        S[:] = fd_demand(vf_mps, qcap_cell_vps, k_vpm[t])
        R[:] = fd_supply(w_mps, qcap_cell_vps, kj_cell_vpm, k_vpm[t])

        f_vps[t, 0] = min(demand_total_vps[t], R[0], cap_interface_vps[0])
        min_SR = np.minimum(S[:-1], R[1:])
        f_vps[t, 1:N] = np.minimum(min_SR, cap_interface_vps[1:N])
        f_vps[t, N] = min(S[-1], cap_interface_vps[-1])

        if t + 1 < steps:
            inflow = f_vps[t, 0:N]
            outflow = f_vps[t, 1:N + 1]
            k_new = k_vpm[t] + lam * (inflow - outflow)
            k_vpm[t + 1] = np.clip(k_new, 0.0, kj_cell_vpm)

        q_cell_vps[t] = 0.5 * (f_vps[t, 0:N] + f_vps[t, 1:N + 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            v_cell = np.where(k_vpm[t] > 1e-9, q_cell_vps[t] / k_vpm[t], vf_mps)
        v_cell_mps[t] = np.clip(v_cell, 0.0, vf_mps)

    if per_lane_outputs:
        dens_vpkm = vehpm_to_vehpkm(k_vpm) / lanes
        flow_vph = vehps_to_vehph(q_cell_vps) / lanes
    else:
        dens_vpkm = vehpm_to_vehpkm(k_vpm)
        flow_vph = vehps_to_vehph(q_cell_vps)

    speed_mph = mps_to_mph(v_cell_mps)
    N_cum = np.cumsum(f_vps * dt_s, axis=0, dtype="float64")

    return {
        "x_cell_m": x_cell_mid,
        "x_if_m": x_interfaces,
        "lanes": lanes,
        "dens_vpkm": dens_vpkm.astype("float32"),
        "flow_vph": flow_vph.astype("float32"),
        "speed_mph": speed_mph.astype("float32"),
        "f_vps": f_vps,
        "N_cum": N_cum,
        "dx_eff": dx_eff,
        "bn_idx": bn_idx,
    }

# ==========================
# ---- Optional calibration --
# ==========================
@st.cache_data(show_spinner=False)
def quick_calibrate_from_tgsim(df: pd.DataFrame, vf_mph: float, w_mph: float) -> dict:
    tb = (np.floor(pd.to_numeric(df["time"], errors="coerce") / 60.0) * 60).astype("int64")
    df2 = df.assign(_tb=tb)
    per = df2.groupby(["_tb", "lane_kf"], observed=True)["id"].nunique().reset_index()
    per["flow_vph"] = per["id"] * (3600.0 / 60.0)
    qmax_lane_obs = per.groupby("lane_kf")["flow_vph"].max().median()
    vf_kmh = vf_mph * 1.609344
    w_kmh = max(1e-3, w_mph * 1.609344)
    kcrit_vpkm_lane = qmax_lane_obs / vf_kmh
    kj_vpkm_lane = qmax_lane_obs * (1.0 / vf_kmh + 1.0 / w_kmh)
    return {
        "qmax_lane_vph": float(qmax_lane_obs),
        "kcrit_lane_vpkm": float(kcrit_vpkm_lane),
        "kj_lane_vpkm": float(kj_vpkm_lane),
    }

# ==========================
# ---- Sidebar: controls ----
# ==========================
with st.sidebar:
    st.header("Model & Discretization")

    L_m = 5.0 * MILE_TO_M
    dx_m = st.slider("Cell length dx [m]", min_value=50, max_value=500, value=100, step=10)

    vf_mph = st.number_input("Free-flow speed v_f [mph]", min_value=20.0, max_value=90.0,
                             value=70.0, step=1.0, format="%.1f")
    w_mph = st.number_input("Congestion wave speed w [mph]", min_value=5.0, max_value=30.0,
                            value=18.0, step=0.5, format="%.1f")
    kj_lane_vpkm = st.number_input("Jam density per lane k_j [veh/km/ln]", min_value=60.0, max_value=240.0,
                                   value=140.0, step=5.0, format="%.1f")

    vf_mps = mph_to_mps(vf_mph)
    w_mps = mph_to_mps(w_mph)

    kcrit_lane = fd_kcrit(vf_mps, w_mps, kj_lane_vpkm)
    qmax_lane_vph = fd_qmax_per_lane(vf_mps, w_mps, kj_lane_vpkm)
    qmax_lane_vps = vehph_to_vehps(qmax_lane_vph)

    st.caption(f"Derived per-lane: kₖ = {kcrit_lane:.1f} veh/km/ln • qₘₐₓ = {qmax_lane_vph:.0f} veh/h/ln")

    dt_max = 0.9 * dx_m / max(vf_mps, w_mps)
    dt_user = st.slider("Time step dt [s]", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    dt_s = float(min(dt_user, dt_max))
    if dt_s < dt_user - 1e-9:
        st.warning(f"dt reduced to {dt_s:.2f}s to satisfy stability (≤ {dt_max:.2f}s).")
    else:
        st.caption(f"Max stable dt ≈ {dt_max:.2f}s")

    horizon_min = st.slider("Simulation horizon [minutes]", min_value=10, max_value=120, value=60, step=5)
    steps = int(np.ceil((horizon_min * 60.0) / dt_s))

    st.header("Lane-drop Bottleneck")
    bn_pos_m = st.slider("Bottleneck position along 5 mi [m]",
                         min_value=200, max_value=int(L_m - 200), value=int(L_m * 0.5), step=50)
    cap_drop_factor = st.slider("Capacity drop factor (× 2-lane cap at interface)",
                                min_value=0.40, max_value=0.60, value=0.50, step=0.01)

    st.header("Upstream Demand")
    demand_mode = st.radio("Demand input", ["Piecewise (3-step)", "Upload CSV"], horizontal=True)

    if demand_mode == "Piecewise (3-step)":
        st.caption("Durations in minutes and total demand in veh/h (both lanes combined).")
        colA, colB = st.columns(2)
        with colA:
            d1 = st.number_input("Seg 1 duration [min]", min_value=1.0, value=15.0, step=1.0)
            d2 = st.number_input("Seg 2 duration [min]", min_value=1.0, value=30.0, step=1.0)
            d3 = st.number_input("Seg 3 duration [min]", min_value=1.0, value=float(max(1.0, horizon_min - d1 - d2)), step=1.0)
        with colB:
            q1 = st.number_input("Seg 1 demand [veh/h total]", min_value=0.0, value=1800.0, step=100.0)
            q2 = st.number_input("Seg 2 demand [veh/h total]", min_value=0.0, value=3600.0, step=100.0)
            q3 = st.number_input("Seg 3 demand [veh/h total]", min_value=0.0, value=2400.0, step=100.0)
        demand_total_vps = build_piecewise_demand(steps, dt_s, [d1, d2, d3], [q1, q2, q3])
    else:
        up = st.file_uploader("Upload CSV with columns: time_s, demand_vph", type=["csv"])
        if up is not None:
            try:
                df_dem = pd.read_csv(io.BytesIO(up.getvalue()))
                demand_total_vps = resample_csv_demand_to_steps(df_dem, steps, dt_s)
                st.success("Demand profile loaded.")
            except Exception as e:
                st.error(f"Failed to parse demand CSV: {e}")
                demand_total_vps = np.zeros(steps, dtype="float32")
        else:
            st.info("Awaiting CSV… Using zero demand placeholder.")
            demand_total_vps = np.zeros(steps, dtype="float32")

    st.header("Initial Conditions")
    init_mode = st.selectbox("Initial density", ["free", "capacity", "near_jam", "custom"])
    init_custom = 0.2 * kj_lane_vpkm
    if init_mode == "custom":
        init_custom = st.number_input("Custom k₀ per lane [veh/km/ln]",
                                      min_value=0.0, max_value=kj_lane_vpkm, value=init_custom, step=5.0)

    st.header("Outputs & Probes")
    per_lane_outputs = st.checkbox("Report density/flow per lane (recommended)", value=True)
    probe_x_m = st.slider("Probe location x [m]", min_value=50, max_value=int(L_m - 50),
                          value=int(0.75 * L_m), step=10)

    st.divider()
    st.caption("Optional: Quick calibration from TGSIM (single-lane focus)")
    with st.expander("Upload TGSIM to suggest qmax and k_j per lane"):
        up2 = st.file_uploader("TGSIM CSV/Parquet", type=["csv", "parquet", "pq"], key="tgsim_calib")
        if up2 is not None:
            try:
                if up2.name.lower().endswith((".parquet", ".pq")):
                    df_up = pd.read_parquet(io.BytesIO(up2.getvalue()))
                else:
                    df_up = pd.read_csv(io.BytesIO(up2.getvalue()))
                rec = quick_calibrate_from_tgsim(df_up, vf_mph=vf_mph, w_mph=w_mph)
                st.write(f"Suggested qₘₐₓ per lane ≈ **{rec['qmax_lane_vph']:.0f} veh/h/ln**")
                st.write(f"Suggested k_c per lane ≈ **{rec['kcrit_lane_vpkm']:.1f} veh/km/ln**")
                st.write(f"Suggested k_j per lane ≈ **{rec['kj_lane_vpkm']:.1f} veh/km/ln**")
                st.caption("Use these to adjust the inputs above if desired.")
            except Exception as e:
                st.error(f"Calibration failed: {e}")

# ==========================
# ---- Run / reuse CTM -------
# ==========================
def _run_and_store_sim(kwargs):
    st.session_state.sim = run_ctm(**kwargs)
    st.session_state.sim_sig = _sim_signature(**kwargs)

_sim_kwargs = dict(
    L_m=L_m, dx_m=dx_m, dt_s=dt_s, steps=steps,
    vf_mps=vf_mps, w_mps=w_mps,
    kj_lane_vpkm=kj_lane_vpkm, qmax_lane_vps=qmax_lane_vps,
    bn_pos_m=float(bn_pos_m), cap_drop_factor=float(cap_drop_factor),
    demand_total_vps=demand_total_vps,
    init_mode=init_mode, init_value_vpkm_per_lane=float(init_custom),
    per_lane_outputs=per_lane_outputs,
)

st.button("Run / Re-run CTM", type="primary", use_container_width=True,
          on_click=_run_and_store_sim, kwargs={"kwargs": _sim_kwargs})

if st.session_state.sim is None:
    st.info("Set parameters and click **Run / Re-run CTM**.")
    st.stop()

if st.session_state.sim_sig != _sim_signature(**_sim_kwargs):
    st.warning("Model parameters changed since the last run. Click **Run / Re-run CTM** to update results.")

sim = st.session_state.sim

# Unpack simulation
x_cell_m = sim["x_cell_m"]
x_if_m = sim["x_if_m"]
lanes = sim["lanes"]
dens = sim["dens_vpkm"]       # (T,N)
flow = sim["flow_vph"]        # (T,N)
speed = sim["speed_mph"]      # (T,N)
f_if = sim["f_vps"]           # (T,N+1)
N_cum = sim["N_cum"]          # (T,N+1)
dx_eff = sim["dx_eff"]
bn_idx = sim["bn_idx"]

T = dens.shape[0]
t_s = np.arange(T) * dt_s
t_min = t_s / 60.0

# Map probe to nearest cell
probe_cell = int(np.clip(np.floor((probe_x_m - 0.5 * dx_eff) / dx_eff), 0, len(x_cell_m) - 1))

# ==========================
# ---- Layout & Plots -------
# ==========================
st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:2.4em; font-weight:800;">🚧 TrafficLens — CTM with Lane-Drop</h1>
<p style="font-size:0.95em; color:gray;">
Triangular FD + Cell Transmission Model over a 5-mile segment with a 2→1 bottleneck. Explore queueing, shockwaves, and throughput.
</p>
</div>
""", unsafe_allow_html=True)

# Info bar
c0, c1, c2, c3 = st.columns(4)
with c0:
    st.metric("Cells", f"{len(x_cell_m)}")
with c1:
    st.metric("dt used", f"{dt_s:.2f} s")
with c2:
    st.metric("Probe @ x", f"{probe_x_m/ MILE_TO_M:.2f} mi")
with c3:
    st.metric("Bottleneck @ x", f"{x_if_m[bn_idx]/ MILE_TO_M:.2f} mi")

# --- Row 1: Probe time series
st.subheader("Probe Time Series at Selected Location")
cA, cB, cC = st.columns(3)
with cA:
    fig_v = px.line(x=t_min, y=speed[:, probe_cell], labels={"x": "Time [min]", "y": "Speed [mph]"})
    fig_v.update_layout(title=f"Speed vs Time (cell {probe_cell})")
    st.plotly_chart(fig_v, use_container_width=True)
with cB:
    dens_label = "Density [veh/km/ln]" if per_lane_outputs else "Density [veh/km]"
    fig_k = px.line(x=t_min, y=dens[:, probe_cell], labels={"x": "Time [min]", "y": dens_label})
    fig_k.update_layout(title="Density vs Time")
    st.plotly_chart(fig_k, use_container_width=True)
with cC:
    flow_label = "Flow [veh/h/ln]" if per_lane_outputs else "Flow [veh/h]"
    fig_q = px.line(x=t_min, y=flow[:, probe_cell], labels={"x": "Time [min]", "y": flow_label})
    fig_q.update_layout(title="Flow vs Time")
    st.plotly_chart(fig_q, use_container_width=True)

# --- Row 2: Space–time heatmaps (Jet + grids + bottleneck line)
st.subheader("Space–Time Fields")
y_space_mi = x_cell_m / MILE_TO_M
L_miles = L_m / MILE_TO_M
dtick_min = _nice_dtick_minutes(horizon_min)
dtick_space_mi = _space_dtick_miles(L_miles)
bn_y_mi = x_if_m[bn_idx] / MILE_TO_M

fig_hv = _heatmap(
    z=speed, x_min=t_min, y_mi=y_space_mi,
    title="Speed (mph) — space–time", cbar_title="mph", unit="Speed",
    bn_y_mi=bn_y_mi, dtick_min=dtick_min, dtick_space_mi=dtick_space_mi, z_q=(1, 99),
)
st.plotly_chart(fig_hv, use_container_width=True)

dens_color = "veh/km/ln" if per_lane_outputs else "veh/km"
fig_hk = _heatmap(
    z=dens, x_min=t_min, y_mi=y_space_mi,
    title=f"Density ({dens_color}) — space–time", cbar_title=dens_color, unit="Density",
    bn_y_mi=bn_y_mi, dtick_min=dtick_min, dtick_space_mi=dtick_space_mi, z_q=(1, 99),
)
st.plotly_chart(fig_hk, use_container_width=True)

# --- Row 3: Cumulative counts (N-curves) with sticky controls + shaded gap
st.subheader("Cumulative Counts (N-curves) to Visualize Queueing")
st.caption("Comparing upstream vs downstream interfaces reveals queue formation (fan-out) and dissipation (re-convergence).")
colU, colD, colPlot = st.columns([1, 1, 2])
with colU:
    st.number_input("Upstream detector offset from bottleneck [m]",
                    min_value=0, max_value=int(L_m), step=50, key="up_off_m")
with colD:
    st.number_input("Downstream detector offset from bottleneck [m]",
                    min_value=0, max_value=int(L_m), step=50, key="dn_off_m")

bn_x = x_if_m[bn_idx]
up_x = max(0.0, bn_x - float(st.session_state.up_off_m))
dn_x = min(x_if_m[-1], bn_x + float(st.session_state.dn_off_m))

up_if = int(np.clip(np.floor(up_x / dx_eff), 0, len(x_if_m) - 1))
dn_if = int(np.clip(np.floor(dn_x / dx_eff), 0, len(x_if_m) - 1))

if up_if == dn_if:
    st.info("Upstream and downstream detectors coincide. Increase offsets to see queue separation.")

Nu = N_cum[:, up_if]
Nd = N_cum[:, dn_if]
gap = np.maximum(Nu - Nd, 0)

with colPlot:
    figN = go.Figure()
    figN.add_trace(go.Scatter(x=t_min, y=Nu, mode="lines",
                              name=f"Upstream @ {up_x/ MILE_TO_M:.2f} mi"))
    figN.add_trace(go.Scatter(x=t_min, y=Nd, mode="lines",
                              name=f"Downstream @ {dn_x/ MILE_TO_M:.2f} mi",
                              fill="tonexty", fillcolor="rgba(0,120,255,0.10)"))
    if gap.size:
        imax = int(np.argmax(gap))
        figN.add_annotation(x=t_min[imax], y=Nu[imax],
                            text=f"Max gap ≈ {gap[imax]:.0f} veh",
                            showarrow=True, arrowhead=2, ax=40, ay=-40,
                            bgcolor="rgba(255,255,255,0.7)")
    figN.update_layout(
        title="Cumulative Vehicle Counts",
        xaxis_title="Time [min]", yaxis_title="Vehicles",
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98),
    )
    st.plotly_chart(figN, use_container_width=True)

st.caption(
    f"Peak stored vehicles between detectors ≈ **{gap.max():.0f}**. "
    f"Detector spacing: **{(dn_x - up_x)/MILE_TO_M:.2f} mi**."
)

# ==========================
# ---- Diagnostics & math ----
# ==========================
with st.expander("Automatic Diagnostics & Notes"):
    idx_up = max(0, bn_idx - 2)
    dens_pl_up = dens[:, idx_up] if per_lane_outputs else dens[:, idx_up] / max(1, int(lanes[idx_up]))

    st.latex(r"\text{Queue criterion:}\quad k(x_{\text{up}},t) > k_c")
    over = dens_pl_up > kcrit_lane
    runs = _runs_from_mask(over)

    if runs:
        i0, i1 = runs[0]
        t_on, t_off = t_min[i0], t_min[i1]
        total_queued_min = sum((t_min[j1] - t_min[j0]) for j0, j1 in runs)
        st.markdown(
            f"**Queue detected upstream** near the bottleneck from **{t_on:.1f} to {t_off:.1f} min** "
            f"(first episode, $k>k_c$). Total queued time ≈ **{total_queued_min:.1f} min**."
        )
    else:
        st.markdown("No sustained queue detected upstream of the bottleneck ($k\\le k_c$).")

    q_bn_vph = vehps_to_vehph(np.nan_to_num(f_if[:, bn_idx], nan=0.0))
    q_peak = float(np.nanmax(q_bn_vph)) if q_bn_vph.size else 0.0
    q_p95 = float(np.nanpercentile(q_bn_vph, 95)) if q_bn_vph.size else 0.0

    st.latex(
        r"\text{Interface capacity setting:}\quad "
        r"C_{i+\tfrac{1}{2}} = \alpha \, (2\, q_{\max}^{\text{lane}}),\quad \alpha = "
        + f"{cap_drop_factor:.2f}"
    )
    st.markdown(
        f"**Bottleneck throughput** — peak: **{q_peak:.0f} veh/h**, robust (95th pct): **{q_p95:.0f} veh/h**.  \n"
        f"Configured capacity: **{vehps_to_vehph(cap_drop_factor*2*qmax_lane_vps):.0f} veh/h**."
    )

    st.latex(r"k_c = \frac{w\,k_j}{v_f + w},\qquad q_{\max}=v_f\,k_c=\frac{v_f\,w\,k_j}{v_f+w},\qquad "
             r"\Delta t \le \frac{\Delta x}{\max(v_f,w)}")

with st.expander("Model equations (CTM + triangular fundamental diagram)", expanded=False):
    st.latex(r"""
                \begin{aligned}
                k_c &= \frac{w\,k_j}{v_f + w}, & \qquad q_{\max} &= v_f\,k_c = \frac{v_f\,w\,k_j}{v_f + w} \\
                S_i^t &= \min\!\big(v_f\,k_i^t,\; Q_i\big), & \qquad R_{i+1}^t &= \min\!\big(w\,(k_{j,i+1}-k_{i+1}^t),\; Q_{i+1}\big) \\
                f_{i+\frac{1}{2}}^t &= \min\!\big(S_i^t,\; R_{i+1}^t,\; C_{i+\frac{1}{2}}\big) \\
                k_i^{t+1} &= k_i^t + \frac{\Delta t}{\Delta x}\,\big(f_{i-\frac{1}{2}}^t - f_{i+\frac{1}{2}}^t\big) \\
                \text{CFL stability:}\quad & \Delta t \le \frac{\Delta x}{\max(v_f,\,w)}
                \end{aligned}
                """)

# ==========================
# ---- Preview raw arrays ----
# ==========================
with st.expander("Preview arrays (first 5 rows)"):
    st.write("Speed [mph]:")
    st.dataframe(pd.DataFrame(speed[:5, :]), use_container_width=True)
    st.write("Density:")
    st.dataframe(pd.DataFrame(dens[:5, :]), use_container_width=True)
    st.write("Flow:")
    st.dataframe(pd.DataFrame(flow[:5, :]), use_container_width=True)
