# Radar Perception Library - API Reference

## Table of Contents

1. [Overview](#overview)
2. [Signal Processing Module](#signal-processing-module)
3. [Detection Module](#detection-module)
4. [Tracking Module](#tracking-module)
5. [Fusion Module](#fusion-module)
6. [Datasets Module](#datasets-module)
7. [Visualization Module](#visualization-module)
8. [Utils Module](#utils-module)
9. [Examples](#examples)

## Overview

The Radar Perception Library provides a comprehensive set of APIs for radar-based perception tasks. All modules are designed to be modular, efficient, and easy to integrate into existing systems.

### Installation

```bash
pip install radar-perception-library
```

### Quick Start

```python
import radar_perception as rp

# Load data
data = rp.datasets.load_sample_data()

# Process radar signals
processor = rp.signal_processing.FMCWProcessor()
processed = processor.process(data)

# Detect objects
detector = rp.detection.CFARDetector()
detections = detector.detect(processed.range_doppler_map)

# Track objects
tracker = rp.tracking.MultiTargetTracker()
tracks = tracker.update(detections, timestamp=data.timestamp)
```

## Signal Processing Module

### `radar_perception.signal_processing`

#### Classes

#### `FMCWProcessor`

Main class for FMCW radar signal processing.

```python
class FMCWProcessor:
    def __init__(self, config: Dict[str, Any])
```

**Parameters:**

- `config`: Configuration dictionary containing radar parameters

**Configuration Keys:**

- `num_tx`: Number of transmit antennas (int)
- `num_rx`: Number of receive antennas (int)
- `num_chirps`: Number of chirps per frame (int)
- `num_samples`: Number of ADC samples per chirp (int)
- `start_freq`: Start frequency in Hz (float)
- `bandwidth`: Chirp bandwidth in Hz (float)
- `chirp_duration`: Chirp duration in seconds (float)

**Methods:**

##### `process(adc_data: np.ndarray) -> ProcessedRadarFrame`

Process raw ADC data through complete FMCW pipeline.

**Parameters:**

- `adc_data`: Raw ADC data with shape (samples, chirps, rx_antennas)

**Returns:**

- `ProcessedRadarFrame`: Object containing processed radar data

**Example:**

```python
config = {
    'num_tx': 3,
    'num_rx': 4, 
    'num_chirps': 128,
    'num_samples': 256,
    'start_freq': 77e9,
    'bandwidth': 4e9,
    'chirp_duration': 62e-6
}

processor = rp.signal_processing.FMCWProcessor(config)
result = processor.process(adc_data)

# Access processed data
range_doppler_map = result.range_doppler_map
angle_response = result.angle_response
```

##### `set_window_type(window_type: str) -> None`

Set windowing function for FFT operations.

**Parameters:**

- `window_type`: Window type ('hann', 'hamming', 'blackman', 'kaiser')

##### `enable_interference_mitigation(enable: bool) -> None`

Enable/disable interference mitigation.

**Parameters:**

- `enable`: Whether to enable interference mitigation

#### `CFARProcessor`

Constant False Alarm Rate (CFAR) processor for adaptive thresholding.

```python
class CFARProcessor:
    def __init__(self, 
                 guard_cells: int = 4,
                 reference_cells: int = 16, 
                 threshold_factor: float = 10.0,
                 cfar_type: str = 'CA')
```

**Parameters:**

- `guard_cells`: Number of guard cells around test cell
- `reference_cells`: Number of reference cells for noise estimation
- `threshold_factor`: Threshold multiplication factor
- `cfar_type`: CFAR algorithm type ('CA', 'OS', 'GO', 'SO')

**Methods:**

##### `detect_1d(signal: np.ndarray) -> List[Dict]`

Perform 1D CFAR detection on input signal.

**Parameters:**

- `signal`: 1D input signal

**Returns:**

- List of detection dictionaries with keys: 'index', 'value', 'threshold'

##### `detect_2d(rd_map: np.ndarray) -> List[Dict]`

Perform 2D CFAR detection on range-Doppler map.

**Parameters:**

- `rd_map`: 2D range-Doppler map

**Returns:**

- List of detection dictionaries

**Example:**

```python
cfar = rp.signal_processing.CFARProcessor(
    guard_cells=4,
    reference_cells=16,
    threshold_factor=12.0,
    cfar_type='CA'
)

detections = cfar.detect_2d(range_doppler_map)

for detection in detections:
    range_idx = detection['range_index']
    doppler_idx = detection['doppler_index']
    snr = detection['snr']
    print(f"Detection at ({range_idx}, {doppler_idx}) with SNR {snr:.1f} dB")
```

#### `BeamFormer`

Digital beamforming for angle estimation.

```python
class BeamFormer:
    def __init__(self, 
                 num_antennas: int,
                 antenna_spacing: float = 0.5,
                 method: str = 'bartlett')
```

**Parameters:**

- `num_antennas`: Number of receive antennas
- `antenna_spacing`: Antenna spacing in wavelengths
- `method`: Beamforming method ('bartlett', 'capon', 'music')

**Methods:**

##### `compute_spectrum(signal: np.ndarray, angles: np.ndarray) -> np.ndarray`

Compute angle spectrum for given signal.

**Parameters:**

- `signal`: Complex signal from antenna array
- `angles`: Array of angles to evaluate (in radians)

**Returns:**

- Angle spectrum values

#### Data Structures

#### `ProcessedRadarFrame`

Container for processed radar data.

**Attributes:**

- `range_doppler_map`: 2D complex range-Doppler map
- `angle_response`: 3D angle response (range, Doppler, angle)
- `timestamp`: Frame timestamp
- `metadata`: Processing metadata

## Detection Module

### `radar_perception.detection`

#### Classes

#### `CFARDetector`

High-level CFAR-based object detector.

```python
class CFARDetector:
    def __init__(self, cfar_config: Dict[str, Any] = None)
```

**Methods:**

##### `detect(rd_map: np.ndarray, **kwargs) -> List[Detection]`

Detect objects in range-Doppler map.

**Parameters:**

- `rd_map`: Range-Doppler map
- `kwargs`: Additional detection parameters

**Returns:**

- List of Detection objects

#### `ClusterDetector`

Clustering-based detector for grouping detection points.

```python
class ClusterDetector:
    def __init__(self,
                 clustering_method: str = 'dbscan',
                 min_points: int = 3,
                 eps: float = 2.0)
```

**Parameters:**

- `clustering_method`: Clustering algorithm ('dbscan', 'kmeans', 'hierarchical')
- `min_points`: Minimum points per cluster
- `eps`: DBSCAN epsilon parameter

##### `cluster_detections(detections: List[Detection]) -> List[DetectionCluster]`

Group individual detections into object clusters.

**Parameters:**

- `detections`: List of individual detection points

**Returns:**

- List of detection clusters

#### `PeakDetector`

Simple peak detection for range-Doppler maps.

```python
class PeakDetector:
    def __init__(self, 
                 threshold: float = 0.5,
                 min_distance: int = 5)
```

#### Data Structures

#### `Detection`

Individual radar detection point.

```python
@dataclass
class Detection:
    range: float          # Range in meters
    doppler: float        # Doppler velocity in m/s  
    angle: float          # Angle in degrees
    amplitude: float      # Detection amplitude
    snr: float           # Signal-to-noise ratio in dB
    timestamp: float     # Detection timestamp
    confidence: float    # Detection confidence [0,1]
```

**Example:**

```python
detection = rp.detection.Detection(
    range=25.5,
    doppler=5.2,
    angle=15.0,
    amplitude=0.8,
    snr=18.5,
    timestamp=time.time(),
    confidence=0.92
)
```

#### `DetectionCluster`

Cluster of detections representing a single object.

```python
@dataclass  
class DetectionCluster:
    detections: List[Detection]
    centroid: Detection
    size: int
    spread: Dict[str, float]  # Range, Doppler, angle spread
    timestamp: float
```

## Tracking Module

### `radar_perception.tracking`

#### Classes

#### `MultiTargetTracker`

Multi-target tracking with Kalman filtering.

```python
class MultiTargetTracker:
    def __init__(self,
                 max_tracks: int = 50,
                 detection_threshold: float = 0.7,
                 deletion_threshold: int = 5,
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0)
```

**Parameters:**

- `max_tracks`: Maximum number of simultaneous tracks
- `detection_threshold`: Minimum confidence for track creation
- `deletion_threshold`: Frames without detection before track deletion
- `process_noise`: Process noise covariance
- `measurement_noise`: Measurement noise covariance

**Methods:**

##### `update(detections: List[Detection], timestamp: float) -> List[Track]`

Update tracker with new detections.

**Parameters:**

- `detections`: List of new detections
- `timestamp`: Current timestamp

**Returns:**

- List of active tracks

**Example:**

```python
tracker = rp.tracking.MultiTargetTracker(
    max_tracks=20,
    detection_threshold=0.8,
    deletion_threshold=3
)

# Update tracker each frame
for frame_data in radar_stream:
    detections = detector.detect(frame_data.rd_map)
    tracks = tracker.update(detections, frame_data.timestamp)
    
    for track in tracks:
        print(f"Track {track.id}: position ({track.state.x:.1f}, {track.state.y:.1f})")
```

##### `get_active_tracks() -> List[Track]`

Get list of currently active tracks.

##### `remove_track(track_id: int) -> bool`

Remove specific track by ID.

#### `KalmanFilter`

Kalman filter implementation for single target tracking.

```python
class KalmanFilter:
    def __init__(self,
                 state_dim: int = 6,  # [x, y, vx, vy, ax, ay]
                 measurement_dim: int = 3)  # [range, angle, doppler]
```

**Methods:**

##### `predict(dt: float) -> np.ndarray`

Predict next state.

##### `update(measurement: np.ndarray) -> np.ndarray`

Update state with measurement.

#### Data Structures

#### `Track`

Individual target track.

```python
@dataclass
class Track:
    id: int
    state: TrackState
    covariance: np.ndarray
    age: int
    hits: int
    confidence: float
    timestamp: float
    history: List[TrackState]
```

#### `TrackState`

Track state vector.

```python
@dataclass
class TrackState:
    x: float          # X position (m)
    y: float          # Y position (m)
    vx: float         # X velocity (m/s)
    vy: float         # Y velocity (m/s)
    ax: float         # X acceleration (m/s²)
    ay: float         # Y acceleration (m/s²)
```

## Fusion Module

### `radar_perception.fusion`

#### Classes

#### `MultiSensorFusion`

Multi-sensor data fusion for radar, camera, and LiDAR.

```python
class MultiSensorFusion:
    def __init__(self,
                 sensors: List[str] = ['radar', 'camera'],
                 fusion_method: str = 'kalman',
                 temporal_alignment: bool = True)
```

**Parameters:**

- `sensors`: List of sensor types to fuse
- `fusion_method`: Fusion algorithm ('kalman', 'particle', 'evidential')
- `temporal_alignment`: Enable temporal synchronization

**Methods:**

##### `fuse_detections(sensor_data: Dict[str, List[Detection]]) -> List[FusedDetection]`

Fuse detections from multiple sensors.

**Parameters:**

- `sensor_data`: Dictionary mapping sensor names to detection lists

**Returns:**

- List of fused detections

**Example:**

```python
fusion = rp.fusion.MultiSensorFusion(
    sensors=['radar', 'camera', 'lidar'],
    fusion_method='kalman'
)

sensor_data = {
    'radar': radar_detections,
    'camera': camera_detections,
    'lidar': lidar_detections
}

fused_detections = fusion.fuse_detections(sensor_data)
```

#### `CoordinateTransformer`

Transform between different coordinate systems.

```python
class CoordinateTransformer:
    def __init__(self, 
                 origin: Tuple[float, float, float] = (0, 0, 0),
                 rotation: Tuple[float, float, float] = (0, 0, 0))
```

**Methods:**

##### `radar_to_cartesian(range_m: float, angle_deg: float) -> Tuple[float, float]`

Convert radar polar coordinates to Cartesian.

##### `transform_detection(detection: Detection, target_frame: str) -> Detection`

Transform detection to target coordinate frame.

#### Data Structures

#### `FusedDetection`

Multi-sensor fused detection.

```python
@dataclass
class FusedDetection:
    position: np.ndarray     # 3D position
    velocity: np.ndarray     # 3D velocity
    classification: str      # Object class
    confidence: float        # Fusion confidence
    contributing_sensors: List[str]
    covariance: np.ndarray
    timestamp: float
```

## Datasets Module

### `radar_perception.datasets`

#### Classes

#### `RadarDataset`

Base class for radar datasets.

```python
class RadarDataset:
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transforms: List[Callable] = None)
```

**Methods:**

##### `__len__() -> int`

Get dataset size.

##### `__getitem__(idx: int) -> RadarFrame`

Get dataset item by index.

##### `get_frame_by_timestamp(timestamp: float) -> RadarFrame`

Get frame closest to given timestamp.

#### `SyntheticRadarDataset`

Generate synthetic radar data for testing.

```python
class SyntheticRadarDataset:
    def __init__(self,
                 num_frames: int = 1000,
                 scenario: str = 'highway',
                 noise_level: float = 0.1)
```

**Methods:**

##### `generate_frame(frame_idx: int) -> RadarFrame`

Generate synthetic radar frame.

**Example:**

```python
# Load real dataset
dataset = rp.datasets.RadarDataset(
    data_path='/path/to/nuScenes',
    split='train'
)

# Or generate synthetic data
synthetic_dataset = rp.datasets.SyntheticRadarDataset(
    num_frames=500,
    scenario='urban',
    noise_level=0.2
)

# Use with PyTorch DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True
)
```

#### Data Structures

#### `RadarFrame`

Single radar frame container.

```python
@dataclass
class RadarFrame:
    adc_data: np.ndarray           # Raw ADC data
    range_doppler_map: np.ndarray  # Processed RD map
    annotations: List[Annotation]   # Ground truth annotations
    timestamp: float
    metadata: Dict[str, Any]
```

#### `Annotation`

Ground truth annotation for radar frame.

```python
@dataclass
class Annotation:
    object_id: int
    class_name: str
    position: np.ndarray    # 3D position
    velocity: np.ndarray    # 3D velocity  
    dimensions: np.ndarray  # [length, width, height]
    yaw: float             # Orientation in radians
    visibility: float      # Visibility score [0,1]
```

## Visualization Module

### `radar_perception.visualization`

#### Functions

#### `plot_range_doppler_map(rd_map: np.ndarray, **kwargs) -> matplotlib.figure.Figure`

Plot range-Doppler map with detections.

**Parameters:**

- `rd_map`: Range-Doppler map to plot
- `detections`: Optional list of detections to overlay
- `colormap`: Colormap for plot ('viridis', 'jet', 'hot')
- `title`: Plot title

**Example:**

```python
fig = rp.visualization.plot_range_doppler_map(
    rd_map=processed_frame.range_doppler_map,
    detections=detections,
    colormap='viridis',
    title='Frame 1234'
)
plt.show()
```

#### `plot_detections_on_rd_map(rd_map: np.ndarray, detections: List[Detection]) -> matplotlib.figure.Figure`

Overlay detections on range-Doppler map.

#### `create_radar_dashboard(data: Dict[str, Any]) -> plotly.graph_objects.Figure`

Create interactive radar dashboard with Plotly.

**Parameters:**

- `data`: Dictionary containing radar data to display

**Returns:**

- Interactive Plotly figure

#### `animate_tracks(tracks: List[Track], duration: float = 10.0) -> matplotlib.animation.Animation`

Create animated visualization of track evolution.

#### `plot_3d_detections(detections: List[Detection]) -> matplotlib.figure.Figure`

Create 3D visualization of radar detections.

**Example:**

```python
# Create dashboard
dashboard_data = {
    'range_doppler_map': rd_map,
    'detections': detections,
    'tracks': tracks,
    'metadata': frame.metadata
}

fig = rp.visualization.create_radar_dashboard(dashboard_data)
fig.show()

# Animate tracks
animation = rp.visualization.animate_tracks(tracks, duration=15.0)
plt.show()
```

## Utils Module

### `radar_perception.utils`

#### Conversion Functions

#### `db_to_linear(db_value: float) -> float`

Convert decibel value to linear scale.

#### `linear_to_db(linear_value: float) -> float`

Convert linear value to decibel scale.

#### `polar_to_cartesian(range_m: float, angle_deg: float) -> Tuple[float, float]`

Convert polar coordinates to Cartesian.

#### `cartesian_to_polar(x: float, y: float) -> Tuple[float, float]`

Convert Cartesian coordinates to polar.

#### `doppler_to_velocity(doppler_shift: float, wavelength: float) -> float`

Convert Doppler shift to velocity.

**Example:**

```python
# Convert SNR from dB to linear
snr_linear = rp.utils.db_to_linear(snr_db)

# Convert detection coordinates
x, y = rp.utils.polar_to_cartesian(range_m=25.0, angle_deg=30.0)

# Calculate velocity from Doppler
velocity = rp.utils.doppler_to_velocity(
    doppler_shift=1000.0,  # Hz
    wavelength=0.004       # 77 GHz wavelength
)
```

#### Signal Processing Utilities

#### `generate_chirp(config: Dict[str, Any]) -> np.ndarray`

Generate FMCW chirp signal.

#### `add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray`

Add Additive White Gaussian Noise to signal.

#### `estimate_noise_floor(signal: np.ndarray) -> float`

Estimate noise floor of signal.

#### Geometric Functions

#### `calculate_distance_3d(point1: np.ndarray, point2: np.ndarray) -> float`

Calculate 3D Euclidean distance between points.

#### `rotate_point_2d(point: np.ndarray, angle: float) -> np.ndarray`

Rotate 2D point by given angle.

#### `transform_points(points: np.ndarray, transformation_matrix: np.ndarray) -> np.ndarray`

Apply transformation matrix to set of points.

## Examples

### Complete Processing Pipeline

```python
import radar_perception as rp
import numpy as np

# Configuration
radar_config = {
    'num_tx': 3,
    'num_rx': 4,
    'num_chirps': 128,
    'num_samples': 256,
    'start_freq': 77e9,
    'bandwidth': 4e9,
    'chirp_duration': 62e-6
}

# Initialize components
processor = rp.signal_processing.FMCWProcessor(radar_config)
detector = rp.detection.CFARDetector()
tracker = rp.tracking.MultiTargetTracker()

# Load data
dataset = rp.datasets.RadarDataset('/path/to/data')

# Process sequence
for frame_idx, frame in enumerate(dataset):
    # Signal processing
    processed = processor.process(frame.adc_data)
    
    # Object detection
    detections = detector.detect(processed.range_doppler_map)
    
    # Multi-target tracking
    tracks = tracker.update(detections, frame.timestamp)
    
    # Visualization
    if frame_idx % 10 == 0:  # Every 10th frame
        fig = rp.visualization.plot_range_doppler_map(
            processed.range_doppler_map,
            detections=detections
        )
        fig.savefig(f'frame_{frame_idx:04d}.png')
    
    # Print results
    print(f"Frame {frame_idx}: {len(detections)} detections, {len(tracks)} tracks")
```

### Multi-Sensor Fusion Example

```python
# Multi-sensor fusion pipeline
fusion = rp.fusion.MultiSensorFusion(['radar', 'camera'])
transformer = rp.fusion.CoordinateTransformer()

# Process multi-sensor data
for radar_frame, camera_frame in zip(radar_data, camera_data):
    # Process each sensor
    radar_detections = radar_detector.detect(radar_frame)
    camera_detections = camera_detector.detect(camera_frame)
    
    # Transform to common coordinate frame
    radar_detections_transformed = [
        transformer.transform_detection(det, 'vehicle_frame') 
        for det in radar_detections
    ]
    
    # Fuse detections
    sensor_data = {
        'radar': radar_detections_transformed,
        'camera': camera_detections
    }
    
    fused_detections = fusion.fuse_detections(sensor_data)
    
    print(f"Fused {len(fused_detections)} objects from "
          f"{len(radar_detections)} radar + {len(camera_detections)} camera")
```

### Real-time Processing

```python
# Real-time processing setup
from radar_perception.real_time import RealTimeProcessor

# Configure real-time processor
rt_processor = RealTimeProcessor(
    radar_config=radar_config,
    max_processing_time_ms=50,
    enable_gpu=True
)

# Start processing
rt_processor.start()

try:
    while True:
        # Get latest results
        results = rt_processor.get_latest_results()
        
        if results:
            tracks = results['tracks']
            for track in tracks:
                print(f"Track {track.id}: speed {track.speed:.1f} m/s")
        
        time.sleep(0.1)  # 10 Hz update rate
        
except KeyboardInterrupt:
    rt_processor.stop()
```

This API reference provides comprehensive documentation for all major components of the radar perception library, with practical examples and clear parameter descriptions for effective usage.
