"""
Visualization utilities for radar perception.

This module provides various plotting and visualization functions
for radar data, detections, tracks, and analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional, Tuple, Dict, Any, Union
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from .detection import Detection
from .tracking import Track, TrackState
from .datasets import RadarFrame


# Set default style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_range_doppler_map(rd_map: np.ndarray,
                          range_resolution: float = 0.2,
                          velocity_resolution: float = 0.1,
                          title: str = "Range-Doppler Map",
                          cmap: str = 'viridis',
                          db_scale: bool = True,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot range-Doppler map.
    
    Args:
        rd_map: Range-Doppler map (range_bins x doppler_bins)
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        title: Plot title
        cmap: Colormap
        db_scale: Whether to plot in dB scale
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to dB scale if requested
    plot_data = rd_map.copy()
    if db_scale:
        plot_data = 20 * np.log10(np.maximum(np.abs(plot_data), 1e-12))
        cbar_label = "Magnitude (dB)"
    else:
        plot_data = np.abs(plot_data)
        cbar_label = "Magnitude"
    
    # Create axis labels
    num_range_bins, num_doppler_bins = rd_map.shape
    range_axis = np.arange(num_range_bins) * range_resolution
    velocity_axis = (np.arange(num_doppler_bins) - num_doppler_bins // 2) * velocity_resolution
    
    # Plot
    im = ax.imshow(plot_data.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[range_axis[0], range_axis[-1], 
                          velocity_axis[0], velocity_axis[-1]])
    
    ax.set_xlabel('Range (m)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    
    plt.tight_layout()
    return fig


def plot_detections_on_rd_map(rd_map: np.ndarray,
                             detections: List[Detection],
                             range_resolution: float = 0.2,
                             velocity_resolution: float = 0.1,
                             title: str = "Range-Doppler Map with Detections",
                             figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot range-Doppler map with detections overlaid.
    
    Args:
        rd_map: Range-Doppler map
        detections: List of detections
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Plot range-Doppler map
    fig = plot_range_doppler_map(rd_map, range_resolution, velocity_resolution, 
                                title, figsize=figsize)
    ax = fig.gca()
    
    # Overlay detections
    if detections:
        det_ranges = [det.range_m for det in detections]
        det_velocities = [det.velocity_mps for det in detections]
        det_snrs = [det.snr_db for det in detections]
        
        scatter = ax.scatter(det_ranges, det_velocities, c=det_snrs, 
                           cmap='Reds', s=100, marker='x', linewidths=2,
                           label=f'{len(detections)} detections')
        
        # Add colorbar for SNR
        cbar = plt.colorbar(scatter, ax=ax, label='SNR (dB)', shrink=0.8)
        ax.legend()
    
    return fig


def plot_polar_detections(detections: List[Detection],
                         max_range: float = 100,
                         title: str = "Polar Detection Plot",
                         figsize: Tuple[int, int] = (8, 8)) -> plt.Figure:
    """
    Plot detections in polar coordinates.
    
    Args:
        detections: List of detections
        max_range: Maximum range for plot
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    if detections:
        angles = [np.radians(det.angle_deg) if det.angle_deg is not None else 0 
                 for det in detections]
        ranges = [det.range_m for det in detections]
        velocities = [det.velocity_mps for det in detections]
        
        scatter = ax.scatter(angles, ranges, c=velocities, cmap='RdBu', 
                           s=100, alpha=0.7)
        
        # Color bar
        cbar = plt.colorbar(scatter, ax=ax, label='Velocity (m/s)', shrink=0.8)
        
        # Add velocity direction arrows
        for det in detections:
            if det.angle_deg is not None:
                angle_rad = np.radians(det.angle_deg)
                # Arrow length proportional to velocity
                arrow_length = abs(det.velocity_mps) * 2
                ax.annotate('', xy=(angle_rad, det.range_m + arrow_length),
                           xytext=(angle_rad, det.range_m),
                           arrowprops=dict(arrowstyle='->', 
                                         color='red' if det.velocity_mps > 0 else 'blue',
                                         lw=1.5))
    
    ax.set_ylim(0, max_range)
    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_cartesian_detections(detections: List[Detection],
                             tracks: Optional[List[Track]] = None,
                             ego_position: Tuple[float, float] = (0, 0),
                             title: str = "Cartesian Detection Plot",
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot detections and tracks in Cartesian coordinates.
    
    Args:
        detections: List of detections
        tracks: Optional list of tracks
        ego_position: Ego vehicle position (x, y)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot detections
    if detections:
        det_x = []
        det_y = []
        det_velocities = []
        
        for det in detections:
            if det.angle_deg is not None:
                x = det.range_m * np.cos(np.radians(det.angle_deg))
                y = det.range_m * np.sin(np.radians(det.angle_deg))
                det_x.append(x)
                det_y.append(y)
                det_velocities.append(det.velocity_mps)
        
        if det_x:
            scatter = ax.scatter(det_x, det_y, c=det_velocities, cmap='RdBu',
                               s=100, alpha=0.7, label='Detections')
            plt.colorbar(scatter, ax=ax, label='Velocity (m/s)')
    
    # Plot tracks
    if tracks:
        track_colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
        
        for i, track in enumerate(tracks):
            color = track_colors[i]
            
            # Plot current position
            ax.scatter(track.position[0], track.position[1], 
                      c=[color], s=150, marker='o', 
                      alpha=0.8, edgecolors='black', linewidth=2)
            
            # Plot velocity arrow
            arrow_scale = 2.0
            ax.arrow(track.position[0], track.position[1],
                    track.velocity[0] * arrow_scale,
                    track.velocity[1] * arrow_scale,
                    head_width=1.0, head_length=1.0, fc=color, ec=color)
            
            # Add track ID
            ax.annotate(f'T{track.id[-3:]}', 
                       (track.position[0], track.position[1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
    
    # Plot ego vehicle
    ego_x, ego_y = ego_position
    ax.scatter(ego_x, ego_y, c='red', s=200, marker='^', 
              label='Ego Vehicle', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    
    plt.tight_layout()
    return fig


def plot_track_trajectories(tracks: List[List[Track]],
                           time_window: int = 50,
                           title: str = "Track Trajectories",
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot track trajectories over time.
    
    Args:
        tracks: List of track lists (one per time step)
        time_window: Number of time steps to show
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group tracks by ID
    track_histories = {}
    for timestep_tracks in tracks[-time_window:]:
        for track in timestep_tracks:
            if track.id not in track_histories:
                track_histories[track.id] = []
            track_histories[track.id].append(track)
    
    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(track_histories)))
    
    for i, (track_id, track_history) in enumerate(track_histories.items()):
        if len(track_history) < 2:
            continue
        
        color = colors[i % len(colors)]
        
        # Extract positions
        positions = np.array([t.position for t in track_history])
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 
               color=color, linewidth=2, alpha=0.7, label=f'Track {track_id[-3:]}')
        
        # Plot current position
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  c=[color], s=100, marker='o', edgecolors='black')
        
        # Plot start position
        ax.scatter(positions[0, 0], positions[0, 1], 
                  c=[color], s=50, marker='s', alpha=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.axis('equal')
    
    plt.tight_layout()
    return fig


def plot_detection_statistics(detections: List[Detection],
                            title: str = "Detection Statistics",
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot various detection statistics.
    
    Args:
        detections: List of detections
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not detections:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No detections to plot', 
               ha='center', va='center', transform=ax.transAxes)
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data
    ranges = [det.range_m for det in detections]
    velocities = [det.velocity_mps for det in detections]
    snrs = [det.snr_db for det in detections]
    magnitudes = [det.magnitude for det in detections]
    
    # Range histogram
    axes[0, 0].hist(ranges, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Range (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Range Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity histogram
    axes[0, 1].hist(velocities, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Velocity (m/s)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Velocity Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # SNR histogram
    axes[1, 0].hist(snrs, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('SNR (dB)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('SNR Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Range vs Velocity scatter
    scatter = axes[1, 1].scatter(ranges, velocities, c=snrs, cmap='viridis', alpha=0.7)
    axes[1, 1].set_xlabel('Range (m)')
    axes[1, 1].set_ylabel('Velocity (m/s)')
    axes[1, 1].set_title('Range vs Velocity (colored by SNR)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='SNR (dB)')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_range_profile(signal: np.ndarray,
                      range_resolution: float = 0.2,
                      title: str = "Range Profile",
                      db_scale: bool = True,
                      figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot range profile.
    
    Args:
        signal: Range profile signal
        range_resolution: Range resolution in meters
        title: Plot title
        db_scale: Whether to plot in dB scale
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    range_axis = np.arange(len(signal)) * range_resolution
    
    if db_scale:
        plot_signal = 20 * np.log10(np.maximum(np.abs(signal), 1e-12))
        ylabel = "Magnitude (dB)"
    else:
        plot_signal = np.abs(signal)
        ylabel = "Magnitude"
    
    ax.plot(range_axis, plot_signal, linewidth=1.5)
    ax.set_xlabel('Range (m)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_doppler_profile(signal: np.ndarray,
                        velocity_resolution: float = 0.1,
                        title: str = "Doppler Profile",
                        db_scale: bool = True,
                        figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot Doppler profile.
    
    Args:
        signal: Doppler profile signal
        velocity_resolution: Velocity resolution in m/s
        title: Plot title
        db_scale: Whether to plot in dB scale
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    num_bins = len(signal)
    velocity_axis = (np.arange(num_bins) - num_bins // 2) * velocity_resolution
    
    if db_scale:
        plot_signal = 20 * np.log10(np.maximum(np.abs(signal), 1e-12))
        ylabel = "Magnitude (dB)"
    else:
        plot_signal = np.abs(signal)
        ylabel = "Magnitude"
    
    ax.plot(velocity_axis, plot_signal, linewidth=1.5)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_cfar_detection(signal: np.ndarray,
                       threshold: np.ndarray,
                       detections: np.ndarray,
                       title: str = "CFAR Detection",
                       figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot CFAR detection results.
    
    Args:
        signal: Input signal
        threshold: CFAR threshold
        detections: Detection mask
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_axis = np.arange(len(signal))
    
    # Plot signal
    ax.plot(x_axis, np.abs(signal), label='Signal', linewidth=1.5, color='blue')
    
    # Plot threshold
    ax.plot(x_axis, threshold, label='CFAR Threshold', 
           linewidth=1.5, color='red', linestyle='--')
    
    # Mark detections
    detection_indices = np.where(detections)[0]
    if len(detection_indices) > 0:
        ax.scatter(detection_indices, np.abs(signal)[detection_indices],
                  color='red', s=100, marker='x', linewidth=3,
                  label=f'{len(detection_indices)} Detections')
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Magnitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


class RadarAnimator:
    """
    Animated visualization for real-time radar data.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize radar animator.
        
        Args:
            figsize: Figure size
        """
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.animation = None
        self.data_buffer = []
        self.max_buffer_size = 100
    
    def setup_plots(self, plot_types: List[str] = ['rd_map', 'detections']):
        """
        Setup subplot layout.
        
        Args:
            plot_types: Types of plots to include
        """
        num_plots = len(plot_types)
        
        if num_plots == 1:
            self.fig, self.axes = plt.subplots(1, 1, figsize=self.figsize)
            self.axes = [self.axes]
        elif num_plots == 2:
            self.fig, self.axes = plt.subplots(1, 2, figsize=self.figsize)
        elif num_plots <= 4:
            self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
            self.axes = self.axes.flatten()
        else:
            raise ValueError("Too many plot types specified")
        
        self.plot_types = plot_types
    
    def update_frame(self, frame_data: RadarFrame):
        """
        Update animation with new frame data.
        
        Args:
            frame_data: New radar frame
        """
        # Add to buffer
        self.data_buffer.append(frame_data)
        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer.pop(0)
        
        # Clear all axes
        for ax in self.axes:
            ax.clear()
        
        current_frame = self.data_buffer[-1]
        
        # Update each plot type
        for i, plot_type in enumerate(self.plot_types):
            ax = self.axes[i]
            
            if plot_type == 'rd_map':
                im = ax.imshow(20 * np.log10(np.maximum(
                    np.abs(current_frame.range_doppler_map.T), 1e-12)),
                    aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f'Range-Doppler Map (t={current_frame.timestamp:.2f}s)')
                ax.set_xlabel('Range Bin')
                ax.set_ylabel('Doppler Bin')
            
            elif plot_type == 'detections':
                if current_frame.detections:
                    ranges = [det.range_m for det in current_frame.detections]
                    velocities = [det.velocity_mps for det in current_frame.detections]
                    ax.scatter(ranges, velocities, s=100, alpha=0.7)
                
                ax.set_xlabel('Range (m)')
                ax.set_ylabel('Velocity (m/s)')
                ax.set_title(f'Detections ({len(current_frame.detections)})')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def start_animation(self, interval: int = 100):
        """
        Start the animation.
        
        Args:
            interval: Animation interval in milliseconds
        """
        def animate(frame):
            if self.data_buffer:
                self.update_frame(self.data_buffer[-1])
        
        self.animation = FuncAnimation(self.fig, animate, interval=interval, blit=False)
        plt.show()
    
    def save_animation(self, filename: str, frames: List[RadarFrame], 
                      fps: int = 10):
        """
        Save animation to file.
        
        Args:
            filename: Output filename
            frames: List of radar frames
            fps: Frames per second
        """
        def animate(frame_idx):
            if frame_idx < len(frames):
                self.update_frame(frames[frame_idx])
        
        self.animation = FuncAnimation(self.fig, animate, frames=len(frames), 
                                     interval=1000/fps, blit=False)
        self.animation.save(filename, writer='pillow', fps=fps)


def create_radar_dashboard(rd_map: np.ndarray,
                          detections: List[Detection],
                          tracks: Optional[List[Track]] = None,
                          range_resolution: float = 0.2,
                          velocity_resolution: float = 0.1,
                          timestamp: float = 0.0,
                          figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
    """
    Create comprehensive radar dashboard.
    
    Args:
        rd_map: Range-Doppler map
        detections: List of detections
        tracks: Optional list of tracks
        range_resolution: Range resolution
        velocity_resolution: Velocity resolution
        timestamp: Current timestamp
        figsize: Figure size
        
    Returns:
        Dashboard figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Range-Doppler map (large, top-left)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    
    # Plot RD map
    plot_data = 20 * np.log10(np.maximum(np.abs(rd_map), 1e-12))
    im1 = ax1.imshow(plot_data.T, aspect='auto', origin='lower', cmap='viridis')
    
    # Overlay detections
    if detections:
        det_ranges = [det.range_bin for det in detections]
        det_velocities = [det.doppler_bin for det in detections]
        ax1.scatter(det_ranges, det_velocities, c='red', s=100, marker='x', linewidths=2)
    
    ax1.set_xlabel('Range Bin')
    ax1.set_ylabel('Doppler Bin')
    ax1.set_title(f'Range-Doppler Map (t={timestamp:.2f}s)')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Cartesian plot (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    
    if detections:
        det_x = []
        det_y = []
        for det in detections:
            if det.angle_deg is not None:
                x = det.range_m * np.cos(np.radians(det.angle_deg))
                y = det.range_m * np.sin(np.radians(det.angle_deg))
                det_x.append(x)
                det_y.append(y)
        
        if det_x:
            ax2.scatter(det_x, det_y, s=50, alpha=0.7, label='Detections')
    
    if tracks:
        for track in tracks:
            ax2.scatter(track.position[0], track.position[1], s=100, marker='o')
            # Velocity arrow
            ax2.arrow(track.position[0], track.position[1],
                     track.velocity[0] * 2, track.velocity[1] * 2,
                     head_width=1, head_length=1)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Cartesian View')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Detection statistics (middle-right)
    ax3 = fig.add_subplot(gs[1, 2])
    
    if detections:
        snrs = [det.snr_db for det in detections]
        ax3.hist(snrs, bins=10, alpha=0.7, color='orange')
    
    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('Count')
    ax3.set_title('SNR Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Range profile (bottom-left)
    ax4 = fig.add_subplot(gs[2, 0])
    
    range_profile = np.mean(np.abs(rd_map), axis=1)
    range_axis = np.arange(len(range_profile)) * range_resolution
    ax4.plot(range_axis, 20 * np.log10(np.maximum(range_profile, 1e-12)))
    ax4.set_xlabel('Range (m)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.set_title('Range Profile')
    ax4.grid(True, alpha=0.3)
    
    # Doppler profile (bottom-middle)
    ax5 = fig.add_subplot(gs[2, 1])
    
    doppler_profile = np.mean(np.abs(rd_map), axis=0)
    num_bins = len(doppler_profile)
    velocity_axis = (np.arange(num_bins) - num_bins // 2) * velocity_resolution
    ax5.plot(velocity_axis, 20 * np.log10(np.maximum(doppler_profile, 1e-12)))
    ax5.set_xlabel('Velocity (m/s)')
    ax5.set_ylabel('Magnitude (dB)')
    ax5.set_title('Doppler Profile')
    ax5.grid(True, alpha=0.3)
    
    # Statistics text (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    stats_text = f"""
    Frame Statistics:
    
    Timestamp: {timestamp:.2f} s
    Detections: {len(detections)}
    Tracks: {len(tracks) if tracks else 0}
    
    Max Range: {len(rd_map) * range_resolution:.1f} m
    Max Velocity: {len(rd_map[0]) * velocity_resolution / 2:.1f} m/s
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Radar Perception Dashboard - Frame {timestamp:.2f}s', fontsize=16)
    
    return fig
