"""
Sensor fusion algorithms for multi-modal radar perception.

This module provides implementations for fusing data from multiple radar sensors
and other sensor modalities to improve perception accuracy and robustness.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

from .detection import Detection
from .tracking import Track, TrackState


class SensorType(Enum):
    """Sensor type enumeration."""
    RADAR = "radar"
    LIDAR = "lidar"
    CAMERA = "camera"
    IMU = "imu"
    GPS = "gps"


@dataclass
class SensorMeasurement:
    """Generic sensor measurement."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Any
    confidence: float = 1.0
    uncertainty: Optional[np.ndarray] = None


@dataclass
class SensorCalibration:
    """Sensor calibration parameters."""
    sensor_id: str
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [x, y, z]
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [roll, pitch, yaw]
    intrinsic_params: Optional[Dict[str, float]] = None
    extrinsic_matrix: Optional[np.ndarray] = None


class CoordinateTransformer:
    """
    Coordinate transformation utilities for multi-sensor fusion.
    
    Handles transformations between different sensor coordinate frames
    and conversion to a common world coordinate system.
    """
    
    def __init__(self, reference_frame: str = "world"):
        """
        Initialize coordinate transformer.
        
        Args:
            reference_frame: Name of the reference coordinate frame
        """
        self.reference_frame = reference_frame
        self.sensor_calibrations: Dict[str, SensorCalibration] = {}
    
    def add_sensor_calibration(self, calibration: SensorCalibration):
        """Add sensor calibration parameters."""
        self.sensor_calibrations[calibration.sensor_id] = calibration
    
    def transform_to_world(self, points: np.ndarray, sensor_id: str) -> np.ndarray:
        """
        Transform points from sensor frame to world frame.
        
        Args:
            points: Points in sensor frame (Nx3)
            sensor_id: ID of the sensor
            
        Returns:
            Points in world frame (Nx3)
        """
        if sensor_id not in self.sensor_calibrations:
            raise ValueError(f"No calibration found for sensor {sensor_id}")
        
        calib = self.sensor_calibrations[sensor_id]
        
        # Create transformation matrix
        rotation = Rotation.from_euler('xyz', calib.orientation).as_matrix()
        translation = calib.position
        
        # Apply transformation
        transformed_points = (rotation @ points.T).T + translation
        
        return transformed_points
    
    def transform_detections_to_world(self, detections: List[Detection], 
                                    sensor_id: str) -> List[Detection]:
        """Transform detections from sensor frame to world frame."""
        if not detections:
            return []
        
        # Extract positions
        positions = np.array([[det.range_m * np.cos(np.radians(det.angle_deg or 0)),
                             det.range_m * np.sin(np.radians(det.angle_deg or 0)),
                             0.0] for det in detections])
        
        # Transform to world frame
        world_positions = self.transform_to_world(positions, sensor_id)
        
        # Update detections
        transformed_detections = []
        for i, det in enumerate(detections):
            world_pos = world_positions[i]
            transformed_det = Detection(
                range_bin=det.range_bin,
                doppler_bin=det.doppler_bin,
                angle_bin=det.angle_bin,
                range_m=np.linalg.norm(world_pos[:2]),
                velocity_mps=det.velocity_mps,
                angle_deg=np.degrees(np.arctan2(world_pos[1], world_pos[0])),
                snr_db=det.snr_db,
                magnitude=det.magnitude,
                timestamp=det.timestamp
            )
            transformed_detections.append(transformed_det)
        
        return transformed_detections


class TemporalAligner:
    """
    Temporal alignment for multi-sensor measurements.
    
    Handles synchronization and interpolation of measurements from
    sensors with different sampling rates and timing.
    """
    
    def __init__(self, max_time_difference: float = 0.1):
        """
        Initialize temporal aligner.
        
        Args:
            max_time_difference: Maximum allowed time difference for alignment
        """
        self.max_time_difference = max_time_difference
        self.measurement_buffers: Dict[str, List[SensorMeasurement]] = {}
    
    def add_measurement(self, measurement: SensorMeasurement):
        """Add a measurement to the buffer."""
        sensor_id = measurement.sensor_id
        if sensor_id not in self.measurement_buffers:
            self.measurement_buffers[sensor_id] = []
        
        self.measurement_buffers[sensor_id].append(measurement)
        
        # Keep buffer size reasonable
        if len(self.measurement_buffers[sensor_id]) > 100:
            self.measurement_buffers[sensor_id] = self.measurement_buffers[sensor_id][-50:]
    
    def get_synchronized_measurements(self, target_time: float) -> Dict[str, SensorMeasurement]:
        """
        Get synchronized measurements closest to target time.
        
        Args:
            target_time: Target synchronization time
            
        Returns:
            Dictionary of sensor_id -> synchronized measurement
        """
        synchronized = {}
        
        for sensor_id, measurements in self.measurement_buffers.items():
            if not measurements:
                continue
            
            # Find closest measurement in time
            time_diffs = [abs(m.timestamp - target_time) for m in measurements]
            closest_idx = np.argmin(time_diffs)
            
            if time_diffs[closest_idx] <= self.max_time_difference:
                synchronized[sensor_id] = measurements[closest_idx]
        
        return synchronized
    
    def interpolate_measurement(self, sensor_id: str, target_time: float) -> Optional[SensorMeasurement]:
        """
        Interpolate measurement at target time.
        
        Args:
            sensor_id: ID of the sensor
            target_time: Target time for interpolation
            
        Returns:
            Interpolated measurement or None if not possible
        """
        if sensor_id not in self.measurement_buffers:
            return None
        
        measurements = self.measurement_buffers[sensor_id]
        if len(measurements) < 2:
            return None
        
        # Find surrounding measurements
        times = [m.timestamp for m in measurements]
        
        # Find measurements before and after target time
        before_idx = None
        after_idx = None
        
        for i, t in enumerate(times):
            if t <= target_time:
                before_idx = i
            elif t > target_time and after_idx is None:
                after_idx = i
                break
        
        if before_idx is None or after_idx is None:
            return None
        
        # Simple linear interpolation for position data
        m1, m2 = measurements[before_idx], measurements[after_idx]
        alpha = (target_time - m1.timestamp) / (m2.timestamp - m1.timestamp)
        
        # This is a simplified interpolation - would need sensor-specific logic
        return m1  # Return closest for now


class AssociationMatrix:
    """
    Association matrix for multi-sensor data association.
    
    Computes similarity/distance metrics between measurements from
    different sensors for data association.
    """
    
    def __init__(self, max_association_distance: float = 5.0):
        """
        Initialize association matrix.
        
        Args:
            max_association_distance: Maximum distance for valid associations
        """
        self.max_association_distance = max_association_distance
    
    def compute_spatial_distance(self, measurements1: List[Detection], 
                                measurements2: List[Detection]) -> np.ndarray:
        """
        Compute spatial distance matrix between two sets of measurements.
        
        Args:
            measurements1: First set of measurements
            measurements2: Second set of measurements
            
        Returns:
            Distance matrix (len(measurements1) x len(measurements2))
        """
        if not measurements1 or not measurements2:
            return np.array([])
        
        # Extract positions
        pos1 = np.array([[m.range_m * np.cos(np.radians(m.angle_deg or 0)),
                         m.range_m * np.sin(np.radians(m.angle_deg or 0))] 
                        for m in measurements1])
        pos2 = np.array([[m.range_m * np.cos(np.radians(m.angle_deg or 0)),
                         m.range_m * np.sin(np.radians(m.angle_deg or 0))] 
                        for m in measurements2])
        
        # Compute pairwise distances
        distances = cdist(pos1, pos2, metric='euclidean')
        
        # Set invalid associations to infinity
        distances[distances > self.max_association_distance] = np.inf
        
        return distances
    
    def compute_velocity_similarity(self, measurements1: List[Detection], 
                                  measurements2: List[Detection]) -> np.ndarray:
        """
        Compute velocity similarity matrix between two sets of measurements.
        
        Args:
            measurements1: First set of measurements
            measurements2: Second set of measurements
            
        Returns:
            Velocity difference matrix
        """
        if not measurements1 or not measurements2:
            return np.array([])
        
        vel1 = np.array([m.velocity_mps for m in measurements1])
        vel2 = np.array([m.velocity_mps for m in measurements2])
        
        # Compute pairwise velocity differences
        vel_diff = np.abs(vel1[:, np.newaxis] - vel2[np.newaxis, :])
        
        return vel_diff
    
    def compute_combined_cost(self, measurements1: List[Detection], 
                            measurements2: List[Detection],
                            spatial_weight: float = 0.7,
                            velocity_weight: float = 0.3) -> np.ndarray:
        """
        Compute combined association cost matrix.
        
        Args:
            measurements1: First set of measurements
            measurements2: Second set of measurements
            spatial_weight: Weight for spatial distance
            velocity_weight: Weight for velocity difference
            
        Returns:
            Combined cost matrix
        """
        spatial_dist = self.compute_spatial_distance(measurements1, measurements2)
        velocity_diff = self.compute_velocity_similarity(measurements1, measurements2)
        
        if spatial_dist.size == 0:
            return np.array([])
        
        # Normalize velocity differences
        if velocity_diff.size > 0:
            velocity_diff = velocity_diff / np.max(velocity_diff) if np.max(velocity_diff) > 0 else velocity_diff
        
        # Combine costs
        combined_cost = spatial_weight * spatial_dist + velocity_weight * velocity_diff
        
        return combined_cost


class MultiSensorFusion:
    """
    Main multi-sensor fusion system.
    
    Integrates measurements from multiple radar sensors and other modalities
    to provide improved perception accuracy and robustness.
    """
    
    def __init__(self,
                 max_association_distance: float = 5.0,
                 temporal_window: float = 0.1,
                 confidence_threshold: float = 0.5):
        """
        Initialize multi-sensor fusion system.
        
        Args:
            max_association_distance: Maximum distance for data association
            temporal_window: Time window for temporal alignment
            confidence_threshold: Minimum confidence for measurement inclusion
        """
        self.coordinate_transformer = CoordinateTransformer()
        self.temporal_aligner = TemporalAligner(temporal_window)
        self.association_matrix = AssociationMatrix(max_association_distance)
        self.confidence_threshold = confidence_threshold
        
        self.sensor_weights: Dict[str, float] = {}
        self.fusion_history: List[Dict] = []
    
    def add_sensor(self, sensor_id: str, calibration: SensorCalibration, weight: float = 1.0):
        """
        Add a sensor to the fusion system.
        
        Args:
            sensor_id: Unique sensor identifier
            calibration: Sensor calibration parameters
            weight: Sensor weight for fusion (0-1)
        """
        self.coordinate_transformer.add_sensor_calibration(calibration)
        self.sensor_weights[sensor_id] = weight
    
    def process_measurements(self, measurements: List[SensorMeasurement]) -> List[Detection]:
        """
        Process and fuse measurements from multiple sensors.
        
        Args:
            measurements: List of sensor measurements
            
        Returns:
            Fused detection list
        """
        # Filter measurements by confidence
        valid_measurements = [m for m in measurements 
                            if m.confidence >= self.confidence_threshold]
        
        # Group measurements by sensor
        sensor_measurements = {}
        for measurement in valid_measurements:
            sensor_id = measurement.sensor_id
            if sensor_id not in sensor_measurements:
                sensor_measurements[sensor_id] = []
            sensor_measurements[sensor_id].append(measurement)
        
        # Transform all measurements to world frame
        world_detections = {}
        for sensor_id, sensor_data in sensor_measurements.items():
            if sensor_data and hasattr(sensor_data[0].data, '__iter__'):
                # Assume data contains Detection objects
                detections = [item for measurement in sensor_data 
                            for item in (measurement.data if isinstance(measurement.data, list) 
                                       else [measurement.data])]
                world_detections[sensor_id] = self.coordinate_transformer.transform_detections_to_world(
                    detections, sensor_id
                )
        
        # Perform data association and fusion
        fused_detections = self._fuse_detections(world_detections)
        
        return fused_detections
    
    def _fuse_detections(self, sensor_detections: Dict[str, List[Detection]]) -> List[Detection]:
        """
        Fuse detections from multiple sensors.
        
        Args:
            sensor_detections: Dictionary of sensor_id -> detections
            
        Returns:
            List of fused detections
        """
        if not sensor_detections:
            return []
        
        # If only one sensor, return its detections
        if len(sensor_detections) == 1:
            return list(sensor_detections.values())[0]
        
        # Multi-sensor fusion
        all_detections = []
        sensor_ids = list(sensor_detections.keys())
        
        # Start with first sensor's detections
        fused_detections = sensor_detections[sensor_ids[0]].copy()
        used_sensors = {sensor_ids[0]}
        
        # Associate and fuse with other sensors
        for sensor_id in sensor_ids[1:]:
            current_detections = sensor_detections[sensor_id]
            
            if not current_detections:
                continue
            
            # Compute association matrix
            cost_matrix = self.association_matrix.compute_combined_cost(
                fused_detections, current_detections
            )
            
            if cost_matrix.size == 0:
                fused_detections.extend(current_detections)
                continue
            
            # Simple nearest neighbor association
            associations = []
            for i in range(cost_matrix.shape[0]):
                min_cost_idx = np.argmin(cost_matrix[i, :])
                if cost_matrix[i, min_cost_idx] < np.inf:
                    associations.append((i, min_cost_idx))
            
            # Fuse associated detections
            new_fused = []
            used_current = set()
            
            for fused_idx, current_idx in associations:
                fused_det = self._fuse_detection_pair(
                    fused_detections[fused_idx], 
                    current_detections[current_idx],
                    self.sensor_weights.get(sensor_id, 1.0)
                )
                new_fused.append(fused_det)
                used_current.add(current_idx)
            
            # Add unassociated detections from current sensor
            for i, det in enumerate(current_detections):
                if i not in used_current:
                    new_fused.append(det)
            
            # Add unassociated detections from fused set
            for i, det in enumerate(fused_detections):
                if i not in [assoc[0] for assoc in associations]:
                    new_fused.append(det)
            
            fused_detections = new_fused
            used_sensors.add(sensor_id)
        
        return fused_detections
    
    def _fuse_detection_pair(self, det1: Detection, det2: Detection, weight2: float = 1.0) -> Detection:
        """
        Fuse two detections using weighted averaging.
        
        Args:
            det1: First detection
            det2: Second detection
            weight2: Weight for second detection
            
        Returns:
            Fused detection
        """
        weight1 = 1.0
        total_weight = weight1 + weight2
        
        # Weighted average of positions and velocities
        fused_range = (weight1 * det1.range_m + weight2 * det2.range_m) / total_weight
        fused_velocity = (weight1 * det1.velocity_mps + weight2 * det2.velocity_mps) / total_weight
        
        # Take maximum SNR and magnitude
        fused_snr = max(det1.snr_db, det2.snr_db)
        fused_magnitude = max(det1.magnitude, det2.magnitude)
        
        # Average angle if both are available
        fused_angle = None
        if det1.angle_deg is not None and det2.angle_deg is not None:
            fused_angle = (weight1 * det1.angle_deg + weight2 * det2.angle_deg) / total_weight
        elif det1.angle_deg is not None:
            fused_angle = det1.angle_deg
        elif det2.angle_deg is not None:
            fused_angle = det2.angle_deg
        
        fused_detection = Detection(
            range_bin=det1.range_bin,  # Keep original bin indices
            doppler_bin=det1.doppler_bin,
            range_m=fused_range,
            velocity_mps=fused_velocity,
            angle_deg=fused_angle,
            snr_db=fused_snr,
            magnitude=fused_magnitude,
            timestamp=max(det1.timestamp, det2.timestamp)
        )
        
        return fused_detection
    
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get statistics about the fusion process."""
        return {
            'num_sensors': len(self.sensor_weights),
            'sensor_weights': self.sensor_weights.copy(),
            'fusion_history_length': len(self.fusion_history),
            'last_fusion_time': self.fusion_history[-1]['timestamp'] if self.fusion_history else None
        }


class TrackLevelFusion:
    """
    Track-level fusion for combining tracks from multiple sensors.
    
    Operates at a higher level than detection fusion, combining
    track estimates from different sensors.
    """
    
    def __init__(self, association_threshold: float = 10.0):
        """
        Initialize track-level fusion.
        
        Args:
            association_threshold: Maximum distance for track association
        """
        self.association_threshold = association_threshold
        self.master_tracks: Dict[str, Track] = {}
        self.track_associations: Dict[str, List[str]] = {}  # master_track_id -> [sensor_track_ids]
    
    def fuse_tracks(self, sensor_tracks: Dict[str, List[Track]]) -> List[Track]:
        """
        Fuse tracks from multiple sensors.
        
        Args:
            sensor_tracks: Dictionary of sensor_id -> track list
            
        Returns:
            List of fused tracks
        """
        # Implementation would involve track-to-track association
        # and covariance intersection or similar fusion techniques
        
        # Simplified version - return tracks from primary sensor
        if sensor_tracks:
            primary_sensor = list(sensor_tracks.keys())[0]
            return sensor_tracks[primary_sensor]
        
        return []
