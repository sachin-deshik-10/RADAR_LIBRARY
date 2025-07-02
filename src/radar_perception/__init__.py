"""
Radar Perception Library

A comprehensive collection of radar datasets, detection algorithms, 
tracking methods, and sensor fusion techniques for autonomous systems 
and robotics applications.

This library provides:
- Signal processing for FMCW radar systems
- Advanced detection algorithms (CFAR, clustering, adaptive)  
- Multi-target tracking with Kalman filtering
- Multi-sensor fusion capabilities
- Dataset utilities for radar data management
- Comprehensive visualization tools
- Utility functions for radar system analysis

Example usage:
    >>> import radar_perception as rp
    >>> processor = rp.signal_processing.FMCWProcessor(...)
    >>> detections = rp.detection.CFARDetector().detect_2d(range_doppler_map)
    >>> tracks = rp.tracking.MultiTargetTracker().update(detections, timestamp)
"""

__version__ = "1.0.0"
__author__ = "Radar Perception Library Contributors"
__email__ = "contact@radarperception.dev"
__license__ = "MIT"

# Core modules
from . import signal_processing
from . import detection
from . import tracking
from . import fusion
from . import datasets
from . import utils
from . import visualization

# Commonly used classes and functions
from .signal_processing import FMCWProcessor, CFARProcessor
from .detection import Detection, CFARDetector, PeakDetector
from .tracking import Track, TrackState, MultiTargetTracker, KalmanFilter
from .fusion import MultiSensorFusion, CoordinateTransformer
from .datasets import RadarFrame, RadarDataset, SyntheticRadarDataset
from .utils import db_to_linear, linear_to_db, polar_to_cartesian, cartesian_to_polar
from .visualization import plot_range_doppler_map, plot_detections_on_rd_map, create_radar_dashboard

__all__ = [
    # Core modules
    "signal_processing",
    "detection", 
    "tracking",
    "fusion",
    "datasets",
    "utils",
    "visualization",
    
    # Main classes
    "FMCWProcessor",
    "CFARProcessor", 
    "Detection",
    "CFARDetector",
    "PeakDetector",
    "Track",
    "TrackState",
    "MultiTargetTracker",
    "KalmanFilter",
    "MultiSensorFusion",
    "CoordinateTransformer",
    "RadarFrame",
    "RadarDataset",
    "SyntheticRadarDataset",
    
    # Utility functions
    "db_to_linear",
    "linear_to_db", 
    "polar_to_cartesian",
    "cartesian_to_polar",
    
    # Visualization functions
    "plot_range_doppler_map",
    "plot_detections_on_rd_map", 
    "create_radar_dashboard",
]
