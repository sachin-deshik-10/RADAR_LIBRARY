"""
Signal Processing Module

Core FMCW radar signal processing algorithms including:
- Range-Doppler processing
- CFAR detection
- Point cloud generation
- Calibration and filtering
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from scipy import signal
from scipy.fft import fft, fft2


class FMCWProcessor:
    """FMCW radar signal processor."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FMCW processor with configuration.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self.sample_rate = config.get('sample_rate', 1e6)
        self.bandwidth = config.get('bandwidth', 4e9)
        self.num_chirps = config.get('num_chirps', 128)
        self.num_samples = config.get('num_samples', 256)
        
    def range_processing(self, adc_data: np.ndarray) -> np.ndarray:
        """
        Perform range FFT on ADC data.
        
        Args:
            adc_data: Raw ADC data [samples, chirps, rx_antennas]
            
        Returns:
            Range FFT data
        """
        # Apply window function
        window = signal.windows.hann(self.num_samples)
        windowed_data = adc_data * window[:, None, None]
        
        # Range FFT
        range_fft = fft(windowed_data, axis=0)
        
        return range_fft
    
    def doppler_processing(self, range_data: np.ndarray) -> np.ndarray:
        """
        Perform Doppler FFT on range data.
        
        Args:
            range_data: Range FFT data
            
        Returns:
            Range-Doppler map
        """
        # Apply window function
        window = signal.windows.hann(self.num_chirps)
        windowed_data = range_data * window[None, :, None]
        
        # Doppler FFT
        doppler_fft = fft(windowed_data, axis=1)
        
        return doppler_fft
    
    def cfar_detection(self, rd_map: np.ndarray, 
                      pfa: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        CFAR detection on range-Doppler map.
        
        Args:
            rd_map: Range-Doppler magnitude map
            pfa: Probability of false alarm
            
        Returns:
            Detection mask and threshold map
        """
        # CA-CFAR parameters
        guard_cells = 4
        reference_cells = 16
        
        detections = np.zeros_like(rd_map, dtype=bool)
        threshold_map = np.zeros_like(rd_map)
        
        for i in range(guard_cells + reference_cells, 
                      rd_map.shape[0] - guard_cells - reference_cells):
            for j in range(guard_cells + reference_cells,
                          rd_map.shape[1] - guard_cells - reference_cells):
                
                # Extract reference cells
                ref_window = rd_map[i-guard_cells-reference_cells:i+guard_cells+reference_cells+1,
                                   j-guard_cells-reference_cells:j+guard_cells+reference_cells+1]
                
                # Exclude guard cells
                ref_window[guard_cells:guard_cells*2+1, guard_cells:guard_cells*2+1] = 0
                
                # Calculate threshold
                noise_level = np.mean(ref_window[ref_window > 0])
                threshold = noise_level * (pfa ** (-1/reference_cells) - 1)
                
                threshold_map[i, j] = threshold
                detections[i, j] = rd_map[i, j] > threshold
        
        return detections, threshold_map


def generate_point_cloud(detections: np.ndarray, 
                        rd_map: np.ndarray,
                        config: Dict[str, Any]) -> np.ndarray:
    """
    Generate point cloud from detections.
    
    Args:
        detections: Detection mask
        rd_map: Range-Doppler map
        config: Processing configuration
        
    Returns:
        Point cloud [N, 4] with [x, y, z, intensity]
    """
    range_resolution = config.get('range_resolution', 0.1)
    velocity_resolution = config.get('velocity_resolution', 0.1)
    
    # Find detection indices
    range_indices, doppler_indices = np.where(detections)
    
    if len(range_indices) == 0:
        return np.array([]).reshape(0, 4)
    
    # Convert to physical coordinates
    ranges = range_indices * range_resolution
    velocities = doppler_indices * velocity_resolution
    intensities = rd_map[range_indices, doppler_indices]
    
    # Assume targets are in front of radar (simplified)
    x = ranges
    y = np.zeros_like(ranges)
    z = np.zeros_like(ranges)
    
    point_cloud = np.column_stack([x, y, z, intensities])
    
    return point_cloud
