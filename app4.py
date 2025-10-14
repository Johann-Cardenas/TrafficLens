"""
Assignment 4: Microscopic Traffic Flow with Lane-Changing

Simulates individual vehicle dynamics on a two-lane ring road
using car-following models (IDM, Gipps, GM) and MOBIL lane-changing logic.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass
from typing import List, Tuple, Callable
import time

# ============================================================================
# VEHICLE CLASS
# ============================================================================

@dataclass
class Vehicle:
    """
    Represents a single vehicle in the simulation.
    
    Attributes:
        id: Unique vehicle identifier
        x: Position along ring (m)
        v: Velocity (m/s)
        lane: Lane number (0 or 1)
        a: Current acceleration (m/s²)
    """
    id: int
    x: float
    v: float
    lane: int
    a: float = 0.0


# ============================================================================
# CAR-FOLLOWING MODELS
# ============================================================================

class IDM:
    """
    Intelligent Driver Model (IDM)
    
    Acceleration model balancing free-flow acceleration and interaction term.
    Reference: Treiber, Hennecke, Helbing (2000)
    """
    
    def __init__(self, v0: float = 30.0, T: float = 1.5, s0: float = 2.0, 
                 a: float = 1.0, b: float = 1.5, delta: float = 4.0):
        """
        Parameters:
            v0: Desired velocity (m/s)
            T: Safe time headway (s)
            s0: Minimum spacing (m)
            a: Maximum acceleration (m/s²)
            b: Comfortable deceleration (m/s²)
            delta: Acceleration exponent
        """
        self.v0 = v0
        self.T = T
        self.s0 = s0
        self.a = a
        self.b = b
        self.delta = delta
    
    def acceleration(self, v: float, s: float, dv: float) -> float:
        """
        Calculate acceleration based on IDM formula.
        
        Args:
            v: Current velocity (m/s)
            s: Gap to leader (m)
            dv: Velocity difference (v - v_leader) (m/s)
        
        Returns:
            Acceleration (m/s²)
        """
        if s <= 0:
            return -self.b * 2  # Emergency braking
        
        # Desired spacing
        s_star = self.s0 + max(0, v * self.T + (v * dv) / (2 * np.sqrt(self.a * self.b)))
        
        # IDM acceleration
        acc = self.a * (1 - (v / self.v0) ** self.delta - (s_star / s) ** 2)
        
        return acc


class Gipps:
    """
    Gipps Car-Following Model
    
    Safe speed model considering free-flow and collision avoidance.
    Reference: Gipps (1981)
    """
    
    def __init__(self, v0: float = 30.0, a: float = 1.0, b: float = -1.5, 
                 tau: float = 1.0, s0: float = 2.0):
        """
        Parameters:
            v0: Desired velocity (m/s)
            a: Maximum acceleration (m/s²)
            b: Most severe braking (m/s², negative)
            tau: Reaction time (s)
            s0: Effective vehicle length (m)
        """
        self.v0 = v0
        self.a = a
        self.b = abs(b)  # Store as positive
        self.tau = tau
        self.s0 = s0
    
    def acceleration(self, v: float, s: float, v_lead: float, dt: float = 0.1) -> float:
        """
        Calculate acceleration based on Gipps model.
        
        Args:
            v: Current velocity (m/s)
            s: Gap to leader (m)
            v_lead: Leader velocity (m/s)
            dt: Time step (s)
        
        Returns:
            Acceleration (m/s²)
        """
        if s <= 0:
            return -self.b  # Emergency braking
        
        # Free-flow acceleration term
        v_free = v + 2.5 * self.a * self.tau * (1 - v / self.v0) * (0.025 + v / self.v0) ** 0.5
        
        # Safe speed considering leader
        discriminant = self.b ** 2 * self.tau ** 2 - self.b * (2 * (s - self.s0) - v * self.tau - v_lead ** 2 / self.b)
        
        if discriminant < 0:
            v_safe = 0
        else:
            v_safe = self.b * self.tau + np.sqrt(discriminant)
        
        # Target velocity is minimum of free and safe
        v_target = min(v_free, v_safe, self.v0)
        
        # Convert to acceleration
        acc = (v_target - v) / dt
        
        return np.clip(acc, -self.b, self.a)


class GM:
    """
    Simplified Gazis-Herman-Rothery (GM) Model
    
    Classical stimulus-response model.
    Reference: Gazis, Herman, Rothery (1961)
    """
    
    def __init__(self, alpha: float = 0.5, l: float = 1.0, m: float = 0.0):
        """
        Parameters:
            alpha: Sensitivity coefficient
            l: Velocity exponent
            m: Spacing exponent
        """
        self.alpha = alpha
        self.l = l
        self.m = m
    
    def acceleration(self, v: float, s: float, dv: float) -> float:
        """
        Calculate acceleration based on GM model.
        
        Args:
            v: Current velocity (m/s)
            s: Gap to leader (m)
            dv: Velocity difference (v - v_leader) (m/s)
        
        Returns:
            Acceleration (m/s²)
        """
        if s <= 0.1:
            return -5.0  # Emergency braking
        
        # GM acceleration
        acc = self.alpha * (v ** self.l) * (-dv) / (s ** self.m)
        
        return np.clip(acc, -5.0, 3.0)


# ============================================================================
# MOBIL LANE-CHANGING MODEL
# ============================================================================

class MOBIL:
    """
    MOBIL: Minimizing Overall Braking Induced by Lane changes
    
    Lane-changing decision model based on incentive and safety criteria.
    Reference: Kesting, Treiber, Helbing (2007)
    """
    
    def __init__(self, p: float = 0.3, b_safe: float = 4.0, a_th: float = 0.2):
        """
        Parameters:
            p: Politeness factor (0=egoistic, 1=altruistic)
            b_safe: Safe deceleration threshold (m/s²)
            a_th: Acceleration advantage threshold (m/s²)
        """
        self.p = p
        self.b_safe = b_safe
        self.a_th = a_th
    
    def should_change_lane(self, a_c: float, a_o: float, a_n: float, 
                          a_no: float, a_nn: float) -> bool:
        """
        Determine if lane change should occur.
        
        Args:
            a_c: Current acceleration of ego vehicle
            a_o: Prospective acceleration after lane change
            a_n: Current acceleration of new follower
            a_no: Prospective acceleration of new follower after change
            a_nn: Prospective acceleration of new leader after change
        
        Returns:
            True if lane change should occur
        """
        # Safety criterion
        if a_no < -self.b_safe:
            return False
        
        # Incentive criterion
        incentive = (a_o - a_c) + self.p * (a_no - a_n)
        
        return incentive > self.a_th


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class RingRoadSimulation:
    """
    Two-lane ring road simulation with microscopic vehicle dynamics.
    """
    
    def __init__(self, 
                 road_length: float = 1000.0,
                 n_vehicles: int = 40,
                 model_type: str = "IDM",
                 model_params: dict = None,
                 mobil_params: dict = None,
                 dt: float = 0.1,
                 seed: int = 42):
        """
        Initialize simulation.
        
        Args:
            road_length: Length of ring road (m)
            n_vehicles: Total number of vehicles
            model_type: Car-following model ("IDM", "Gipps", or "GM")
            model_params: Parameters for car-following model
            mobil_params: Parameters for MOBIL model
            dt: Time step (s)
            seed: Random seed for initialization
        """
        self.L = road_length
        self.n_vehicles = n_vehicles
        self.dt = dt
        
        np.random.seed(seed)
        
        # Initialize car-following model
        if model_params is None:
            model_params = {}
        
        if model_type == "IDM":
            self.cf_model = IDM(**model_params)
        elif model_type == "Gipps":
            self.cf_model = Gipps(**model_params)
        elif model_type == "GM":
            self.cf_model = GM(**model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        
        # Initialize MOBIL
        if mobil_params is None:
            mobil_params = {}
        self.mobil = MOBIL(**mobil_params)
        
        # Initialize vehicles
        self.vehicles = self._initialize_vehicles()
        
        # Storage for trajectory data
        self.history = {
            'time': [],
            'positions': [],
            'velocities': [],
            'lanes': [],
            'accelerations': []
        }
    
    def _initialize_vehicles(self) -> List[Vehicle]:
        """
        Initialize vehicles with uniform spacing on two lanes.
        
        Returns:
            List of Vehicle objects
        """
        vehicles = []
        
        # Distribute vehicles evenly between lanes
        n_per_lane = self.n_vehicles // 2
        
        # Lane 0 vehicles
        spacing_0 = self.L / n_per_lane
        for i in range(n_per_lane):
            x = i * spacing_0
            v = 20.0 + np.random.normal(0, 2.0)  # Small velocity perturbation
            vehicles.append(Vehicle(id=i, x=x, v=max(v, 0), lane=0))
        
        # Lane 1 vehicles (offset by half spacing)
        spacing_1 = self.L / (self.n_vehicles - n_per_lane)
        for i in range(self.n_vehicles - n_per_lane):
            x = (i * spacing_1 + spacing_1 / 2) % self.L
            v = 20.0 + np.random.normal(0, 2.0)
            vehicles.append(Vehicle(id=n_per_lane + i, x=x, v=max(v, 0), lane=1))
        
        return vehicles
    
    def _get_leader(self, vehicle: Vehicle, lane: int = None) -> Tuple[Vehicle, float]:
        """
        Find leader vehicle and gap in specified lane.
        
        Args:
            vehicle: Subject vehicle
            lane: Lane to search (if None, use vehicle's current lane)
        
        Returns:
            (leader_vehicle, gap)
        """
        if lane is None:
            lane = vehicle.lane
        
        # Get all vehicles in target lane
        lane_vehicles = [v for v in self.vehicles if v.lane == lane and v.id != vehicle.id]
        
        if not lane_vehicles:
            # No other vehicles in lane
            return None, self.L
        
        # Find closest vehicle ahead (considering periodic boundary)
        min_gap = float('inf')
        leader = None
        
        for v in lane_vehicles:
            gap = (v.x - vehicle.x) % self.L
            if gap < min_gap and gap > 0:
                min_gap = gap
                leader = v
        
        return leader, min_gap
    
    def _get_follower(self, vehicle: Vehicle, lane: int = None) -> Tuple[Vehicle, float]:
        """
        Find follower vehicle and gap in specified lane.
        
        Args:
            vehicle: Subject vehicle
            lane: Lane to search (if None, use vehicle's current lane)
        
        Returns:
            (follower_vehicle, gap)
        """
        if lane is None:
            lane = vehicle.lane
        
        # Get all vehicles in target lane
        lane_vehicles = [v for v in self.vehicles if v.lane == lane and v.id != vehicle.id]
        
        if not lane_vehicles:
            return None, self.L
        
        # Find closest vehicle behind
        min_gap = float('inf')
        follower = None
        
        for v in lane_vehicles:
            gap = (vehicle.x - v.x) % self.L
            if gap < min_gap and gap > 0:
                min_gap = gap
                follower = v
        
        return follower, min_gap
    
    def _calculate_acceleration(self, vehicle: Vehicle, leader: Vehicle, gap: float) -> float:
        """
        Calculate acceleration for vehicle given leader.
        
        Args:
            vehicle: Subject vehicle
            leader: Leader vehicle (or None)
            gap: Gap to leader (m)
        
        Returns:
            Acceleration (m/s²)
        """
        if leader is None:
            # No leader - free flow
            if self.model_type == "IDM":
                return self.cf_model.acceleration(vehicle.v, gap, 0)
            elif self.model_type == "Gipps":
                return self.cf_model.acceleration(vehicle.v, gap, vehicle.v, self.dt)
            elif self.model_type == "GM":
                return 0.0  # No acceleration change
        
        # With leader
        dv = vehicle.v - leader.v
        
        if self.model_type == "IDM":
            return self.cf_model.acceleration(vehicle.v, gap, dv)
        elif self.model_type == "Gipps":
            return self.cf_model.acceleration(vehicle.v, gap, leader.v, self.dt)
        elif self.model_type == "GM":
            return self.cf_model.acceleration(vehicle.v, gap, dv)
    
    def _check_lane_change(self, vehicle: Vehicle) -> int:
        """
        Check if vehicle should change lanes using MOBIL.
        
        Args:
            vehicle: Subject vehicle
        
        Returns:
            New lane (or current lane if no change)
        """
        current_lane = vehicle.lane
        target_lane = 1 - current_lane  # Other lane
        
        # Get current situation
        leader_c, gap_c = self._get_leader(vehicle, current_lane)
        follower_c, _ = self._get_follower(vehicle, current_lane)
        
        # Get target lane situation
        leader_t, gap_t = self._get_leader(vehicle, target_lane)
        follower_t, gap_ft = self._get_follower(vehicle, target_lane)
        
        # Calculate accelerations
        a_c = self._calculate_acceleration(vehicle, leader_c, gap_c)
        a_o = self._calculate_acceleration(vehicle, leader_t, gap_t)
        
        # New follower's current and prospective accelerations
        if follower_t:
            follower_t_leader, follower_t_gap = self._get_leader(follower_t, target_lane)
            a_n = self._calculate_acceleration(follower_t, follower_t_leader, follower_t_gap)
            a_no = self._calculate_acceleration(follower_t, vehicle, gap_ft)
        else:
            a_n = 0.0
            a_no = 0.0
        
        a_nn = 0.0  # Not used in basic MOBIL
        
        # Check MOBIL criteria
        if self.mobil.should_change_lane(a_c, a_o, a_n, a_no, a_nn):
            return target_lane
        
        return current_lane
    
    def step(self):
        """
        Perform one simulation time step.
        """
        # Phase 1: Check lane changes
        lane_changes = {}
        for vehicle in self.vehicles:
            new_lane = self._check_lane_change(vehicle)
            if new_lane != vehicle.lane:
                lane_changes[vehicle.id] = new_lane
        
        # Apply lane changes
        for vid, new_lane in lane_changes.items():
            vehicle = next(v for v in self.vehicles if v.id == vid)
            vehicle.lane = new_lane
        
        # Phase 2: Calculate accelerations
        for vehicle in self.vehicles:
            leader, gap = self._get_leader(vehicle)
            vehicle.a = self._calculate_acceleration(vehicle, leader, gap)
        
        # Phase 3: Update velocities and positions
        for vehicle in self.vehicles:
            # Update velocity
            vehicle.v = max(0, vehicle.v + vehicle.a * self.dt)
            
            # Update position with periodic boundary
            vehicle.x = (vehicle.x + vehicle.v * self.dt) % self.L
    
    def run(self, duration: float, save_interval: float = 0.5):
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration (s)
            save_interval: Interval for saving data (s)
        """
        n_steps = int(duration / self.dt)
        save_every = int(save_interval / self.dt)
        
        for step in range(n_steps):
            self.step()
            
            # Save data at specified interval
            if step % save_every == 0:
                t = step * self.dt
                self.history['time'].append(t)
                self.history['positions'].append([v.x for v in self.vehicles])
                self.history['velocities'].append([v.v for v in self.vehicles])
                self.history['lanes'].append([v.lane for v in self.vehicles])
                self.history['accelerations'].append([v.a for v in self.vehicles])
    
    def get_trajectory_data(self) -> pd.DataFrame:
        """
        Convert history to DataFrame for analysis.
        
        Returns:
            DataFrame with columns: time, vehicle_id, position, velocity, lane, acceleration
        """
        records = []
        
        for i, t in enumerate(self.history['time']):
            for vid in range(self.n_vehicles):
                records.append({
                    'time': t,
                    'vehicle_id': vid,
                    'position': self.history['positions'][i][vid],
                    'velocity': self.history['velocities'][i][vid],
                    'lane': self.history['lanes'][i][vid],
                    'acceleration': self.history['accelerations'][i][vid]
                })
        
        return pd.DataFrame(records)
    
    def calculate_macroscopic_measures(self, window_length: float = 500.0) -> pd.DataFrame:
        """
        Calculate macroscopic traffic measures (speed, flow, density) over time.
        
        Args:
            window_length: Length of measurement window (m)
        
        Returns:
            DataFrame with macroscopic measures
        """
        records = []
        
        for i, t in enumerate(self.history['time']):
            for lane in [0, 1]:
                # Get vehicles in this lane
                lane_vehicles = [(vid, self.history['positions'][i][vid], 
                                self.history['velocities'][i][vid]) 
                               for vid in range(self.n_vehicles) 
                               if self.history['lanes'][i][vid] == lane]
                
                if lane_vehicles:
                    # Calculate measures
                    n = len(lane_vehicles)
                    density = n / (window_length / 1000)  # veh/km
                    mean_speed = np.mean([v for _, _, v in lane_vehicles])  # m/s
                    flow = density * mean_speed * 3.6  # veh/h
                    
                    records.append({
                        'time': t,
                        'lane': lane,
                        'speed': mean_speed,
                        'density': density,
                        'flow': flow,
                        'count': n
                    })
        
        df = pd.DataFrame(records)
        
        # Also calculate aggregate measures
        agg_records = []
        for t in self.history['time']:
            df_t = df[df['time'] == t]
            if not df_t.empty:
                total_vehicles = df_t['count'].sum()
                avg_speed = (df_t['speed'] * df_t['count']).sum() / total_vehicles
                total_density = total_vehicles / (self.L / 1000)
                total_flow = total_density * avg_speed * 3.6
                
                agg_records.append({
                    'time': t,
                    'lane': 'aggregate',
                    'speed': avg_speed,
                    'density': total_density,
                    'flow': total_flow,
                    'count': total_vehicles
                })
        
        df_agg = pd.DataFrame(agg_records)
        return pd.concat([df, df_agg], ignore_index=True)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_trajectories(df: pd.DataFrame, road_length: float, title: str = "Vehicle Trajectories"):
    """
    Plot space-time trajectories for all vehicles.
    
    Args:
        df: Trajectory DataFrame
        road_length: Length of ring road (m)
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Plot trajectory for each vehicle, colored by lane
    for vid in df['vehicle_id'].unique():
        df_veh = df[df['vehicle_id'] == vid]
        
        # Determine dominant lane for coloring
        dominant_lane = df_veh['lane'].mode()[0]
        color = 'blue' if dominant_lane == 0 else 'red'
        
        fig.add_trace(go.Scatter(
            x=df_veh['time'],
            y=df_veh['position'],
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.6,
            showlegend=False,
            hovertemplate='Vehicle %d<br>Time: %%{x:.1f} s<br>Position: %%{y:.1f} m<extra></extra>' % vid
        ))
    
    # Add dummy traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Lane 0'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2),
        name='Lane 1'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Position (m)',
        template='plotly_white',
        hovermode='closest',
        height=500
    )
    
    return fig


def plot_velocity_profiles(df: pd.DataFrame, title: str = "Velocity Profiles"):
    """
    Plot velocity vs time for all vehicles.
    
    Args:
        df: Trajectory DataFrame
        title: Plot title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for vid in df['vehicle_id'].unique():
        df_veh = df[df['vehicle_id'] == vid]
        dominant_lane = df_veh['lane'].mode()[0]
        color = 'blue' if dominant_lane == 0 else 'red'
        
        fig.add_trace(go.Scatter(
            x=df_veh['time'],
            y=df_veh['velocity'],
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.5,
            showlegend=False,
            hovertemplate='Vehicle %d<br>Time: %%{x:.1f} s<br>Velocity: %%{y:.1f} m/s<extra></extra>' % vid
        ))
    
    # Add dummy traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Lane 0'
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='red', width=2),
        name='Lane 1'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Velocity (m/s)',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_lane_distribution(df: pd.DataFrame, title: str = "Lane Distribution Over Time"):
    """
    Plot stacked area chart of vehicles per lane over time.
    
    Args:
        df: Trajectory DataFrame
        title: Plot title
    
    Returns:
        Plotly figure
    """
    # Count vehicles per lane at each time step
    lane_counts = df.groupby(['time', 'lane']).size().reset_index(name='count')
    
    fig = go.Figure()
    
    for lane in [0, 1]:
        df_lane = lane_counts[lane_counts['lane'] == lane]
        color = 'blue' if lane == 0 else 'red'
        
        fig.add_trace(go.Scatter(
            x=df_lane['time'],
            y=df_lane['count'],
            mode='lines',
            name=f'Lane {lane}',
            line=dict(color=color),
            fill='tonexty' if lane == 1 else 'tozeroy',
            opacity=0.6
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Number of Vehicles',
        template='plotly_white',
        height=350
    )
    
    return fig


def plot_macroscopic_measures(df_macro: pd.DataFrame):
    """
    Plot macroscopic measures (speed, density, flow) over time.
    
    Args:
        df_macro: Macroscopic measures DataFrame
    
    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Average Speed', 'Density', 'Flow'),
        vertical_spacing=0.12
    )
    
    colors = {'aggregate': 'black', 0: 'blue', 1: 'red'}
    
    for lane in ['aggregate', 0, 1]:
        df_lane = df_macro[df_macro['lane'] == lane]
        
        if df_lane.empty:
            continue
        
        color = colors[lane]
        name = 'Total' if lane == 'aggregate' else f'Lane {lane}'
        
        # Speed
        fig.add_trace(go.Scatter(
            x=df_lane['time'],
            y=df_lane['speed'],
            mode='lines',
            name=name,
            line=dict(color=color),
            showlegend=True
        ), row=1, col=1)
        
        # Density
        fig.add_trace(go.Scatter(
            x=df_lane['time'],
            y=df_lane['density'],
            mode='lines',
            name=name,
            line=dict(color=color),
            showlegend=False
        ), row=2, col=1)
        
        # Flow
        fig.add_trace(go.Scatter(
            x=df_lane['time'],
            y=df_lane['flow'],
            mode='lines',
            name=name,
            line=dict(color=color),
            showlegend=False
        ), row=3, col=1)
    
    fig.update_xaxes(title_text='Time (s)', row=3, col=1)
    fig.update_yaxes(title_text='Speed (m/s)', row=1, col=1)
    fig.update_yaxes(title_text='Density (veh/km)', row=2, col=1)
    fig.update_yaxes(title_text='Flow (veh/h)', row=3, col=1)
    
    fig.update_layout(
        height=900,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    return fig


def plot_fundamental_diagram(df_macro: pd.DataFrame):
    """
    Plot fundamental diagram (flow vs density).
    
    Args:
        df_macro: Macroscopic measures DataFrame
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    colors = {'aggregate': 'black', 0: 'blue', 1: 'red'}
    
    for lane in ['aggregate', 0, 1]:
        df_lane = df_macro[df_macro['lane'] == lane]
        
        if df_lane.empty:
            continue
        
        color = colors[lane]
        name = 'Total' if lane == 'aggregate' else f'Lane {lane}'
        
        fig.add_trace(go.Scatter(
            x=df_lane['density'],
            y=df_lane['flow'],
            mode='markers',
            name=name,
            marker=dict(color=color, size=6, opacity=0.6),
            hovertemplate='Density: %{x:.1f} veh/km<br>Flow: %{y:.0f} veh/h<extra></extra>'
        ))
    
    fig.update_layout(
        title='Fundamental Diagram',
        xaxis_title='Density (veh/km)',
        yaxis_title='Flow (veh/h)',
        template='plotly_white',
        height=500
    )
    
    return fig


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Assignment 4: Two-Lane Ring Road Simulation",
        page_icon="🚗",
        layout="wide"
    )
    
    st.markdown("""
    <div style="text-align:center;">
    <h1 style="font-size:2.4em; font-weight:800;">🚗 TrafficLens: Two-Lane Ring Road Simulation</h1>
    <p style="font-size:0.95em; color:gray;">
    Microscopic traffic simulation with car-following models and lane-changing behavior.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Car-Following Model",
        ["IDM", "Gipps", "GM"],
        help="Select the longitudinal car-following model"
    )
    
    # Network parameters
    st.sidebar.subheader("Network Configuration")
    road_length = st.sidebar.slider("Ring Road Length (m)", 500, 2000, 1000, 100)
    n_vehicles = st.sidebar.slider("Number of Vehicles", 20, 100, 40, 5)
    
    # Model-specific parameters
    st.sidebar.subheader(f"{model_type} Parameters")
    
    model_params = {}
    
    if model_type == "IDM":
        model_params['v0'] = st.sidebar.slider("Desired Speed v₀ (m/s)", 20.0, 40.0, 30.0, 1.0)
        model_params['T'] = st.sidebar.slider("Time Headway T (s)", 0.5, 3.0, 1.5, 0.1)
        model_params['s0'] = st.sidebar.slider("Min Spacing s₀ (m)", 1.0, 5.0, 2.0, 0.5)
        model_params['a'] = st.sidebar.slider("Max Acceleration a (m/s²)", 0.5, 2.0, 1.0, 0.1)
        model_params['b'] = st.sidebar.slider("Comfortable Decel b (m/s²)", 0.5, 3.0, 1.5, 0.1)
    
    elif model_type == "Gipps":
        model_params['v0'] = st.sidebar.slider("Desired Speed v₀ (m/s)", 20.0, 40.0, 30.0, 1.0)
        model_params['a'] = st.sidebar.slider("Max Acceleration a (m/s²)", 0.5, 2.0, 1.0, 0.1)
        model_params['b'] = st.sidebar.slider("Max Braking b (m/s²)", -3.0, -0.5, -1.5, 0.1)
        model_params['tau'] = st.sidebar.slider("Reaction Time τ (s)", 0.5, 2.0, 1.0, 0.1)
    
    elif model_type == "GM":
        model_params['alpha'] = st.sidebar.slider("Sensitivity α", 0.1, 1.0, 0.5, 0.05)
        model_params['l'] = st.sidebar.slider("Velocity Exponent l", 0.0, 2.0, 1.0, 0.1)
        model_params['m'] = st.sidebar.slider("Spacing Exponent m", 0.0, 2.0, 0.0, 0.1)
    
    # MOBIL parameters
    st.sidebar.subheader("MOBIL Parameters")
    mobil_params = {
        'p': st.sidebar.slider("Politeness p", 0.0, 1.0, 0.3, 0.1),
        'b_safe': st.sidebar.slider("Safe Decel b_safe (m/s²)", 2.0, 6.0, 4.0, 0.5),
        'a_th': st.sidebar.slider("Accel Threshold a_th (m/s²)", 0.0, 0.5, 0.2, 0.05)
    }
    
    # Simulation settings
    st.sidebar.subheader("Simulation Settings")
    duration = st.sidebar.slider("Duration (s)", 50, 300, 120, 10)
    dt = st.sidebar.select_slider("Time Step (s)", [0.05, 0.1, 0.2, 0.5], 0.1)
    seed = st.sidebar.number_input("Random Seed", 0, 1000, 42, 1)
    
    # Run simulation button
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        with st.spinner("Running simulation..."):
            # Initialize simulation
            sim = RingRoadSimulation(
                road_length=road_length,
                n_vehicles=n_vehicles,
                model_type=model_type,
                model_params=model_params,
                mobil_params=mobil_params,
                dt=dt,
                seed=seed
            )
            
            # Run simulation
            start_time = time.time()
            sim.run(duration=duration, save_interval=0.5)
            elapsed = time.time() - start_time
            
            # Get results
            df_traj = sim.get_trajectory_data()
            df_macro = sim.calculate_macroscopic_measures(window_length=road_length)
            
            # Store in session state
            st.session_state['sim'] = sim
            st.session_state['df_traj'] = df_traj
            st.session_state['df_macro'] = df_macro
            st.session_state['elapsed'] = elapsed
            st.session_state['model_type'] = model_type
        
        st.success(f"✅ Simulation completed in {elapsed:.2f} seconds!")
    
    # Display results if available
    if 'df_traj' in st.session_state:
        df_traj = st.session_state['df_traj']
        df_macro = st.session_state['df_macro']
        sim = st.session_state['sim']
        model_name = st.session_state['model_type']
        
        # Summary statistics
        st.header("📊 Simulation Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model", model_name)
        
        with col2:
            avg_speed = df_macro[df_macro['lane'] == 'aggregate']['speed'].mean()
            st.metric("Avg Speed", f"{avg_speed:.2f} m/s")
        
        with col3:
            avg_density = df_macro[df_macro['lane'] == 'aggregate']['density'].mean()
            st.metric("Avg Density", f"{avg_density:.1f} veh/km")
        
        with col4:
            avg_flow = df_macro[df_macro['lane'] == 'aggregate']['flow'].mean()
            st.metric("Avg Flow", f"{avg_flow:.0f} veh/h")
        
        # Microscopic visualizations
        st.header("🔬 Microscopic Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Trajectories", "Velocities", "Lane Usage"])
        
        with tab1:
            st.plotly_chart(
                plot_trajectories(df_traj, sim.L, f"{model_name} Model: Vehicle Trajectories"),
                use_container_width=True
            )
            st.markdown("""
            **Interpretation:** Space-time trajectories show individual vehicle paths. 
            - **Blue** = Lane 0, **Red** = Lane 1
            - Horizontal stretches indicate free-flow
            - Steep slopes suggest congestion or stop-and-go waves
            """)
        
        with tab2:
            st.plotly_chart(
                plot_velocity_profiles(df_traj, f"{model_name} Model: Velocity Profiles"),
                use_container_width=True
            )
            st.markdown("""
            **Interpretation:** Velocity time-series for each vehicle.
            - Stable models show smooth, consistent speeds
            - Unstable models exhibit oscillations (stop-and-go)
            """)
        
        with tab3:
            st.plotly_chart(
                plot_lane_distribution(df_traj, "Lane Distribution Over Time"),
                use_container_width=True
            )
            st.markdown("""
            **Interpretation:** Number of vehicles per lane over time.
            - Balanced distribution suggests effective lane-changing
            - Asymmetry may indicate lane preference or imbalance
            """)
        
        # Macroscopic visualizations
        st.header("🌐 Macroscopic Analysis")
        
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.plotly_chart(
                plot_macroscopic_measures(df_macro),
                use_container_width=True
            )
            st.markdown("""
            **Interpretation:** Aggregate traffic measures over time.
            - **Speed:** Average velocity (higher = better flow)
            - **Density:** Vehicles per km (higher = more congestion)
            - **Flow:** Vehicles per hour (throughput measure)
            - **Black** = Total, **Blue** = Lane 0, **Red** = Lane 1
            """)
        
        with col_right:
            st.plotly_chart(
                plot_fundamental_diagram(df_macro),
                use_container_width=True
            )
            st.markdown("""
            **Fundamental Diagram:** Relationship between flow and density.
            - **Rising branch:** Free-flow regime
            - **Peak:** Maximum flow (capacity)
            - **Falling branch:** Congested regime
            - Scatter indicates dynamic transitions
            """)
        
        # Model comparison insights
        st.header("💡 Key Observations")
        
        if model_name == "IDM":
            st.markdown("""
            ### IDM (Intelligent Driver Model)
            - **Stability:** Generally stable with smooth acceleration/deceleration
            - **Behavior:** Balances free-flow desire with safe following
            - **Stop-and-Go:** Emerges only at high densities or with aggressive parameters
            - **Lane-Changing:** MOBIL integrates well with IDM's continuous acceleration
            """)
        
        elif model_name == "Gipps":
            st.markdown("""
            ### Gipps Model
            - **Stability:** Very stable due to explicit safety considerations
            - **Behavior:** Conservative, safety-first approach
            - **Stop-and-Go:** Rare; model prioritizes collision avoidance
            - **Lane-Changing:** May be less frequent due to cautious nature
            """)
        
        elif model_name == "GM":
            st.markdown("""
            ### GM (Gazis-Herman-Rothery) Model
            - **Stability:** Can be unstable depending on parameters (α, l, m)
            - **Behavior:** Simplified stimulus-response; less realistic
            - **Stop-and-Go:** Prone to oscillations and instability
            - **Lane-Changing:** May amplify instabilities through lane interactions
            """)
        
        st.markdown("""
        ---
        ### General Insights
        
        **Stability Analysis:**
        - IDM and Gipps are generally string-stable under typical parameters
        - GM model can exhibit string instability, leading to growing oscillations
        - Lane-changing can either dampen or amplify instabilities depending on MOBIL parameters
        
        **Lane-Changing Impact:**
        - MOBIL balances individual incentive with collective safety
        - Higher politeness (p) reduces lane-change frequency but improves overall flow
        - Lane changes can relieve local congestion but may disrupt followers
        
        **Fundamental Diagram:**
        - Capacity (peak flow) depends on model parameters and lane-changing behavior
        - Two-lane systems can achieve nearly double single-lane capacity
        - Scatter in FD reflects transient dynamics and lane asymmetries
        """)
        
        # Data export
        st.header("💾 Export Data")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            csv_traj = df_traj.to_csv(index=False)
            st.download_button(
                label="📥 Download Trajectory Data (CSV)",
                data=csv_traj,
                file_name=f"trajectories_{model_name}_{int(time.time())}.csv",
                mime="text/csv"
            )
        
        with col_exp2:
            csv_macro = df_macro.to_csv(index=False)
            st.download_button(
                label="📥 Download Macroscopic Data (CSV)",
                data=csv_macro,
                file_name=f"macroscopic_{model_name}_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    else:
        # Initial instructions
        st.info("👈 Configure simulation parameters in the sidebar and click **Run Simulation** to begin.")
        
        st.markdown("""
        ### About This Simulation
        
        This application implements a **two-lane ring road** with microscopic vehicle dynamics:
        
        #### Car-Following Models
        1. **IDM (Intelligent Driver Model):** Balances desired velocity with safe following distance
        2. **Gipps Model:** Explicitly considers collision avoidance and reaction time
        3. **GM (Gazis-Herman-Rothery):** Classical stimulus-response model
        
        #### Lane-Changing: MOBIL
        - **Minimizing Overall Braking Induced by Lane changes**
        - Considers ego vehicle incentive and impact on new follower
        - Balances individual benefit with collective safety
        
        #### Simulation Features
        - Periodic boundary conditions (ring topology)
        - Zero vehicle length assumption
        - Uniform initial spacing with velocity perturbations
        - Explicit time-stepping with configurable dt
        
        #### Visualizations
        - **Microscopic:** Individual trajectories, velocities, lane choices
        - **Macroscopic:** Aggregate speed, density, flow, fundamental diagram
        - **Interactive:** Hover for details, zoom, pan
        
        ---
        **Ready to explore?** Adjust parameters and run your first simulation! 🚀
        """)


if __name__ == "__main__":
    main()

