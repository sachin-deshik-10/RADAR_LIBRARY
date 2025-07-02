"""
Detection algorithms for radar perception.

This module provides implementations of various detection algorithms
commonly used in radar systems for target detection and parameter estimation.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import scipy.signal
from scipy.cluster.hierarchy import linkage, fcluster


@dataclass
class Detection:
    """Represents a radar detection."""
    range_bin: int
    doppler_bin: int
    angle_bin: Optional[int] = None
    range_m: float = 0.0
    velocity_mps: float = 0.0
    angle_deg: Optional[float] = None
    snr_db: float = 0.0
    magnitude: float = 0.0
    timestamp: float = 0.0


class CFARDetector:
    """
    Constant False Alarm Rate (CFAR) detector for radar applications.
    
    Implements Cell Averaging CFAR (CA-CFAR) and Ordered Statistics CFAR (OS-CFAR)
    algorithms for adaptive threshold detection in radar systems.
    """
    
    def __init__(self, 
                 guard_cells: int = 2,
                 training_cells: int = 16,
                 false_alarm_rate: float = 1e-6,
                 cfar_type: str = 'CA'):
        """
        Initialize CFAR detector.
        
        Args:
            guard_cells: Number of guard cells on each side
            training_cells: Number of training cells on each side
            false_alarm_rate: Target false alarm rate
            cfar_type: Type of CFAR ('CA' for Cell Averaging, 'OS' for Ordered Statistics)
        """
        self.guard_cells = guard_cells
        self.training_cells = training_cells
        self.false_alarm_rate = false_alarm_rate
        self.cfar_type = cfar_type
        
        # Calculate threshold factor based on false alarm rate
        if cfar_type == 'CA':
            self.threshold_factor = training_cells * 2 * (false_alarm_rate ** (-1/(2*training_cells)) - 1)
        else:  # OS-CFAR
            # Simplified OS-CFAR threshold calculation
            self.threshold_factor = 2 * training_cells * (false_alarm_rate ** (-1/(2*training_cells)) - 1)
    
    def detect_1d(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform 1D CFAR detection on input signal.
        
        Args:
            signal: Input signal (typically range profile or Doppler profile)
            
        Returns:
            Tuple of (detections_mask, threshold_values)
        """
        signal_power = np.abs(signal) ** 2
        detections = np.zeros_like(signal_power, dtype=bool)
        thresholds = np.zeros_like(signal_power)
        
        total_window = 2 * (self.guard_cells + self.training_cells) + 1
        
        for i in range(len(signal_power)):
            # Define training cell indices
            start_idx = max(0, i - self.guard_cells - self.training_cells)
            end_idx = min(len(signal_power), i + self.guard_cells + self.training_cells + 1)
            
            # Exclude guard cells and cell under test
            left_training = signal_power[start_idx:max(0, i - self.guard_cells)]
            right_training = signal_power[min(len(signal_power), i + self.guard_cells + 1):end_idx]
            
            training_samples = np.concatenate([left_training, right_training])
            
            if len(training_samples) > 0:
                if self.cfar_type == 'CA':
                    noise_level = np.mean(training_samples)
                else:  # OS-CFAR
                    # Use median for ordered statistics
                    noise_level = np.median(training_samples)
                
                threshold = noise_level * self.threshold_factor
                thresholds[i] = threshold
                
                if signal_power[i] > threshold:
                    detections[i] = True
        
        return detections, thresholds
    
    def detect_2d(self, range_doppler_map: np.ndarray) -> List[Detection]:
        """
        Perform 2D CFAR detection on range-Doppler map.
        
        Args:
            range_doppler_map: 2D array (range_bins x doppler_bins)
            
        Returns:
            List of Detection objects
        """
        detections = []
        power_map = np.abs(range_doppler_map) ** 2
        
        for range_idx in range(power_map.shape[0]):
            # Perform 1D CFAR along Doppler dimension
            range_profile = power_map[range_idx, :]
            doppler_detections, _ = self.detect_1d(range_profile)
            
            for doppler_idx in np.where(doppler_detections)[0]:
                detection = Detection(
                    range_bin=range_idx,
                    doppler_bin=doppler_idx,
                    magnitude=power_map[range_idx, doppler_idx],
                    snr_db=10 * np.log10(power_map[range_idx, doppler_idx] / np.mean(power_map))
                )
                detections.append(detection)
        
        return detections


class PeakDetector:
    """Peak detection algorithms for radar applications."""
    
    @staticmethod
    def find_peaks_2d(data: np.ndarray, 
                      min_distance: int = 3,
                      threshold_abs: Optional[float] = None,
                      threshold_rel: float = 0.1) -> List[Tuple[int, int]]:
        """
        Find peaks in 2D data using local maximum detection.
        
        Args:
            data: 2D input data
            min_distance: Minimum distance between peaks
            threshold_abs: Absolute threshold for peak detection
            threshold_rel: Relative threshold (fraction of maximum)
            
        Returns:
            List of (row, col) peak positions
        """
        if threshold_abs is None:
            threshold_abs = threshold_rel * np.max(data)
        
        # Find local maxima
        peaks = []
        rows, cols = data.shape
        
        for i in range(min_distance, rows - min_distance):
            for j in range(min_distance, cols - min_distance):
                if data[i, j] > threshold_abs:
                    # Check if it's a local maximum
                    local_region = data[i-min_distance:i+min_distance+1, 
                                      j-min_distance:j+min_distance+1]
                    if data[i, j] == np.max(local_region):
                        peaks.append((i, j))
        
        return peaks


class ClusteringDetector:
    """Clustering-based detection for grouping nearby detections."""
    
    def __init__(self, distance_threshold: float = 2.0):
        """
        Initialize clustering detector.
        
        Args:
            distance_threshold: Maximum distance for clustering
        """
        self.distance_threshold = distance_threshold
    
    def cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """
        Cluster detections based on spatial proximity.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of detection clusters
        """
        if len(detections) < 2:
            return [detections] if detections else []
        
        # Create feature matrix (range_bin, doppler_bin)
        features = np.array([[det.range_bin, det.doppler_bin] for det in detections])
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(features, method='ward')
        cluster_labels = fcluster(linkage_matrix, 
                                self.distance_threshold, 
                                criterion='distance')
        
        # Group detections by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])
        
        return list(clusters.values())
    
    def merge_clustered_detections(self, 
                                 detection_clusters: List[List[Detection]]) -> List[Detection]:
        """
        Merge clustered detections into single detections per cluster.
        
        Args:
            detection_clusters: List of detection clusters
            
        Returns:
            List of merged detections
        """
        merged_detections = []
        
        for cluster in detection_clusters:
            if not cluster:
                continue
            
            # Calculate weighted centroid based on magnitude
            total_magnitude = sum(det.magnitude for det in cluster)
            
            if total_magnitude > 0:
                weighted_range = sum(det.range_bin * det.magnitude for det in cluster) / total_magnitude
                weighted_doppler = sum(det.doppler_bin * det.magnitude for det in cluster) / total_magnitude
                max_snr = max(det.snr_db for det in cluster)
                
                merged_detection = Detection(
                    range_bin=int(round(weighted_range)),
                    doppler_bin=int(round(weighted_doppler)),
                    magnitude=total_magnitude,
                    snr_db=max_snr,
                    timestamp=cluster[0].timestamp
                )
                merged_detections.append(merged_detection)
        
        return merged_detections


class AdaptiveDetector:
    """Adaptive detection algorithms that adjust parameters based on environment."""
    
    def __init__(self, adaptation_rate: float = 0.1):
        """
        Initialize adaptive detector.
        
        Args:
            adaptation_rate: Rate of adaptation (0-1)
        """
        self.adaptation_rate = adaptation_rate
        self.noise_level_estimate = 1.0
        self.false_alarm_history = []
    
    def update_noise_estimate(self, signal: np.ndarray):
        """Update noise level estimate based on current signal."""
        # Estimate noise as lower percentile of signal power
        signal_power = np.abs(signal) ** 2
        current_noise = np.percentile(signal_power, 25)  # 25th percentile
        
        # Exponential moving average
        self.noise_level_estimate = (1 - self.adaptation_rate) * self.noise_level_estimate + \
                                  self.adaptation_rate * current_noise
    
    def adaptive_threshold(self, target_false_alarm_rate: float = 1e-6) -> float:
        """Calculate adaptive threshold based on current noise estimate."""
        # Simple adaptive threshold calculation
        threshold_factor = -np.log(target_false_alarm_rate)
        return self.noise_level_estimate * threshold_factor


def convert_detections_to_physical(detections: List[Detection],
                                 range_resolution: float,
                                 velocity_resolution: float,
                                 angle_resolution: Optional[float] = None) -> List[Detection]:
    """
    Convert detection bin indices to physical units.
    
    Args:
        detections: List of detections with bin indices
        range_resolution: Range resolution in meters
        velocity_resolution: Velocity resolution in m/s
        angle_resolution: Angle resolution in degrees (optional)
        
    Returns:
        List of detections with physical units
    """
    converted_detections = []
    
    for det in detections:
        converted_det = Detection(
            range_bin=det.range_bin,
            doppler_bin=det.doppler_bin,
            angle_bin=det.angle_bin,
            range_m=det.range_bin * range_resolution,
            velocity_mps=det.doppler_bin * velocity_resolution,
            snr_db=det.snr_db,
            magnitude=det.magnitude,
            timestamp=det.timestamp
        )
        
        if angle_resolution is not None and det.angle_bin is not None:
            converted_det.angle_deg = det.angle_bin * angle_resolution
        
        converted_detections.append(converted_det)
    
    return converted_detections
