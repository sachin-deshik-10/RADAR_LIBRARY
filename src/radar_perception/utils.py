"""
Utility functions for radar perception.

This module provides various utility functions for data processing,
coordinate transformations, and common operations in radar perception.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import math
from scipy import constants
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def db_to_linear(db_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert decibel values to linear scale.
    
    Args:
        db_value: Value(s) in decibels
        
    Returns:
        Linear scale value(s)
    """
    return 10 ** (db_value / 10)


def linear_to_db(linear_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert linear values to decibel scale.
    
    Args:
        linear_value: Linear scale value(s)
        
    Returns:
        Value(s) in decibels
    """
    return 10 * np.log10(np.maximum(linear_value, 1e-12))  # Avoid log(0)


def range_to_bin(range_m: float, range_resolution: float) -> int:
    """
    Convert range in meters to range bin index.
    
    Args:
        range_m: Range in meters
        range_resolution: Range resolution in meters
        
    Returns:
        Range bin index
    """
    return int(round(range_m / range_resolution))


def bin_to_range(range_bin: int, range_resolution: float) -> float:
    """
    Convert range bin index to range in meters.
    
    Args:
        range_bin: Range bin index
        range_resolution: Range resolution in meters
        
    Returns:
        Range in meters
    """
    return range_bin * range_resolution


def velocity_to_bin(velocity_mps: float, velocity_resolution: float, 
                   num_doppler_bins: int) -> int:
    """
    Convert velocity in m/s to Doppler bin index.
    
    Args:
        velocity_mps: Velocity in m/s
        velocity_resolution: Velocity resolution in m/s
        num_doppler_bins: Number of Doppler bins
        
    Returns:
        Doppler bin index
    """
    doppler_bin = int(round(velocity_mps / velocity_resolution))
    # Shift to account for zero-velocity at center
    return doppler_bin + num_doppler_bins // 2


def bin_to_velocity(doppler_bin: int, velocity_resolution: float, 
                   num_doppler_bins: int) -> float:
    """
    Convert Doppler bin index to velocity in m/s.
    
    Args:
        doppler_bin: Doppler bin index
        velocity_resolution: Velocity resolution in m/s
        num_doppler_bins: Number of Doppler bins
        
    Returns:
        Velocity in m/s
    """
    # Shift to account for zero-velocity at center
    velocity_bin = doppler_bin - num_doppler_bins // 2
    return velocity_bin * velocity_resolution


def angle_to_bin(angle_deg: float, angle_resolution: float, 
                num_angle_bins: int) -> int:
    """
    Convert angle in degrees to angle bin index.
    
    Args:
        angle_deg: Angle in degrees
        angle_resolution: Angle resolution in degrees
        num_angle_bins: Number of angle bins
        
    Returns:
        Angle bin index
    """
    angle_bin = int(round(angle_deg / angle_resolution))
    # Shift to account for zero angle at center
    return angle_bin + num_angle_bins // 2


def bin_to_angle(angle_bin: int, angle_resolution: float, 
                num_angle_bins: int) -> float:
    """
    Convert angle bin index to angle in degrees.
    
    Args:
        angle_bin: Angle bin index
        angle_resolution: Angle resolution in degrees
        num_angle_bins: Number of angle bins
        
    Returns:
        Angle in degrees
    """
    # Shift to account for zero angle at center
    centered_bin = angle_bin - num_angle_bins // 2
    return centered_bin * angle_resolution


def polar_to_cartesian(range_m: float, angle_deg: float) -> Tuple[float, float]:
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Args:
        range_m: Range in meters
        angle_deg: Angle in degrees
        
    Returns:
        Tuple of (x, y) coordinates in meters
    """
    angle_rad = math.radians(angle_deg)
    x = range_m * math.cos(angle_rad)
    y = range_m * math.sin(angle_rad)
    return x, y


def cartesian_to_polar(x: float, y: float) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        x: X coordinate in meters
        y: Y coordinate in meters
        
    Returns:
        Tuple of (range_m, angle_deg)
    """
    range_m = math.sqrt(x**2 + y**2)
    angle_deg = math.degrees(math.atan2(y, x))
    return range_m, angle_deg


def calculate_doppler_frequency(velocity_mps: float, frequency_hz: float) -> float:
    """
    Calculate Doppler frequency shift.
    
    Args:
        velocity_mps: Radial velocity in m/s (positive = approaching)
        frequency_hz: Transmit frequency in Hz
        
    Returns:
        Doppler frequency shift in Hz
    """
    c = constants.speed_of_light  # Speed of light in m/s
    return 2 * frequency_hz * velocity_mps / c


def calculate_velocity_from_doppler(doppler_freq_hz: float, frequency_hz: float) -> float:
    """
    Calculate velocity from Doppler frequency shift.
    
    Args:
        doppler_freq_hz: Doppler frequency shift in Hz
        frequency_hz: Transmit frequency in Hz
        
    Returns:
        Radial velocity in m/s
    """
    c = constants.speed_of_light  # Speed of light in m/s
    return doppler_freq_hz * c / (2 * frequency_hz)


def calculate_range_resolution(bandwidth_hz: float) -> float:
    """
    Calculate range resolution from bandwidth.
    
    Args:
        bandwidth_hz: Signal bandwidth in Hz
        
    Returns:
        Range resolution in meters
    """
    c = constants.speed_of_light  # Speed of light in m/s
    return c / (2 * bandwidth_hz)


def calculate_velocity_resolution(coherent_processing_interval: float, 
                                frequency_hz: float) -> float:
    """
    Calculate velocity resolution from coherent processing interval.
    
    Args:
        coherent_processing_interval: CPI duration in seconds
        frequency_hz: Transmit frequency in Hz
        
    Returns:
        Velocity resolution in m/s
    """
    c = constants.speed_of_light  # Speed of light in m/s
    wavelength = c / frequency_hz
    return wavelength / (2 * coherent_processing_interval)


def calculate_maximum_unambiguous_range(pulse_repetition_freq: float) -> float:
    """
    Calculate maximum unambiguous range.
    
    Args:
        pulse_repetition_freq: Pulse repetition frequency in Hz
        
    Returns:
        Maximum unambiguous range in meters
    """
    c = constants.speed_of_light  # Speed of light in m/s
    return c / (2 * pulse_repetition_freq)


def calculate_maximum_unambiguous_velocity(pulse_repetition_freq: float, 
                                         frequency_hz: float) -> float:
    """
    Calculate maximum unambiguous velocity.
    
    Args:
        pulse_repetition_freq: Pulse repetition frequency in Hz
        frequency_hz: Transmit frequency in Hz
        
    Returns:
        Maximum unambiguous velocity in m/s
    """
    c = constants.speed_of_light  # Speed of light in m/s
    wavelength = c / frequency_hz
    return wavelength * pulse_repetition_freq / 4


def apply_windowing(signal: np.ndarray, window_type: str = 'hann') -> np.ndarray:
    """
    Apply windowing function to signal.
    
    Args:
        signal: Input signal
        window_type: Type of window ('hann', 'hamming', 'blackman', 'kaiser')
        
    Returns:
        Windowed signal
    """
    from scipy.signal import get_window
    
    if signal.ndim == 1:
        window = get_window(window_type, len(signal))
        return signal * window
    elif signal.ndim == 2:
        # Apply window to each dimension
        window_rows = get_window(window_type, signal.shape[0])
        window_cols = get_window(window_type, signal.shape[1])
        window_2d = np.outer(window_rows, window_cols)
        return signal * window_2d
    else:
        raise ValueError("Windowing only supported for 1D and 2D signals")


def zero_pad_signal(signal: np.ndarray, target_length: int, axis: int = -1) -> np.ndarray:
    """
    Zero-pad signal to target length.
    
    Args:
        signal: Input signal
        target_length: Target length after padding
        axis: Axis along which to pad
        
    Returns:
        Zero-padded signal
    """
    current_length = signal.shape[axis]
    if current_length >= target_length:
        return signal
    
    pad_width = [(0, 0) for _ in range(signal.ndim)]
    pad_width[axis] = (0, target_length - current_length)
    
    return np.pad(signal, pad_width, mode='constant', constant_values=0)


def interpolate_missing_data(data: np.ndarray, 
                           invalid_mask: np.ndarray,
                           method: str = 'linear') -> np.ndarray:
    """
    Interpolate missing or invalid data points.
    
    Args:
        data: Input data array
        invalid_mask: Boolean mask indicating invalid data points
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        Data with interpolated values
    """
    if np.all(invalid_mask):
        return data  # Cannot interpolate if all data is invalid
    
    result = data.copy()
    
    if data.ndim == 1:
        valid_indices = np.where(~invalid_mask)[0]
        invalid_indices = np.where(invalid_mask)[0]
        
        if len(valid_indices) > 1 and len(invalid_indices) > 0:
            # Only interpolate within the range of valid data
            min_valid = np.min(valid_indices)
            max_valid = np.max(valid_indices)
            
            interp_indices = invalid_indices[
                (invalid_indices >= min_valid) & (invalid_indices <= max_valid)
            ]
            
            if len(interp_indices) > 0:
                f = interp1d(valid_indices, data[valid_indices], 
                           kind=method, bounds_error=False, fill_value='extrapolate')
                result[interp_indices] = f(interp_indices)
    
    elif data.ndim == 2:
        # Interpolate row-wise and column-wise
        for i in range(data.shape[0]):
            row_mask = invalid_mask[i, :]
            if not np.all(row_mask) and np.any(row_mask):
                result[i, :] = interpolate_missing_data(data[i, :], row_mask, method)
        
        for j in range(data.shape[1]):
            col_mask = invalid_mask[:, j]
            if not np.all(col_mask) and np.any(col_mask):
                result[:, j] = interpolate_missing_data(result[:, j], col_mask, method)
    
    return result


def sliding_window_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply sliding window average to signal.
    
    Args:
        signal: Input signal
        window_size: Size of averaging window
        
    Returns:
        Smoothed signal
    """
    if window_size <= 1:
        return signal
    
    if signal.ndim == 1:
        # Use convolution for 1D
        kernel = np.ones(window_size) / window_size
        return np.convolve(signal, kernel, mode='same')
    
    elif signal.ndim == 2:
        # Apply to each row and column
        result = signal.copy()
        
        # Row-wise smoothing
        for i in range(signal.shape[0]):
            kernel = np.ones(min(window_size, signal.shape[1])) / min(window_size, signal.shape[1])
            result[i, :] = np.convolve(signal[i, :], kernel, mode='same')
        
        # Column-wise smoothing
        for j in range(signal.shape[1]):
            kernel = np.ones(min(window_size, signal.shape[0])) / min(window_size, signal.shape[0])
            result[:, j] = np.convolve(result[:, j], kernel, mode='same')
        
        return result
    
    else:
        raise ValueError("Sliding window average only supported for 1D and 2D signals")


def median_filter(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply median filter to remove impulse noise.
    
    Args:
        signal: Input signal
        window_size: Size of median filter window (should be odd)
        
    Returns:
        Filtered signal
    """
    from scipy.signal import medfilt
    
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    return medfilt(signal, kernel_size=window_size)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio.
    
    Args:
        signal: Signal array
        noise: Noise array
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    return 10 * np.log10(snr_linear)


def estimate_noise_floor(signal: np.ndarray, percentile: float = 25) -> float:
    """
    Estimate noise floor from signal using percentile method.
    
    Args:
        signal: Input signal
        percentile: Percentile to use for noise estimation (0-100)
        
    Returns:
        Estimated noise floor level
    """
    signal_power = np.abs(signal) ** 2
    return np.percentile(signal_power, percentile)


def normalize_signal(signal: np.ndarray, method: str = 'max') -> np.ndarray:
    """
    Normalize signal using various methods.
    
    Args:
        signal: Input signal
        method: Normalization method ('max', 'rms', 'std', 'minmax')
        
    Returns:
        Normalized signal
    """
    if method == 'max':
        max_val = np.max(np.abs(signal))
        return signal / max_val if max_val != 0 else signal
    
    elif method == 'rms':
        rms_val = np.sqrt(np.mean(signal ** 2))
        return signal / rms_val if rms_val != 0 else signal
    
    elif method == 'std':
        std_val = np.std(signal)
        return (signal - np.mean(signal)) / std_val if std_val != 0 else signal
    
    elif method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        range_val = max_val - min_val
        return (signal - min_val) / range_val if range_val != 0 else signal
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def find_local_maxima(signal: np.ndarray, 
                     min_height: Optional[float] = None,
                     min_distance: int = 1) -> Tuple[np.ndarray, Dict]:
    """
    Find local maxima in signal.
    
    Args:
        signal: Input signal
        min_height: Minimum height for peaks
        min_distance: Minimum distance between peaks
        
    Returns:
        Tuple of (peak_indices, peak_properties)
    """
    peaks, properties = find_peaks(signal, 
                                  height=min_height,
                                  distance=min_distance)
    return peaks, properties


def circular_correlation(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """
    Compute circular correlation between two signals.
    
    Args:
        signal1: First signal
        signal2: Second signal
        
    Returns:
        Circular correlation
    """
    # Ensure signals are same length
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    # Compute circular correlation via FFT
    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)
    correlation = np.fft.ifft(fft1 * np.conj(fft2))
    
    return np.real(correlation)


def range_doppler_coupling_compensation(range_doppler_map: np.ndarray,
                                      range_migration_compensation: bool = True) -> np.ndarray:
    """
    Compensate for range-Doppler coupling effects.
    
    Args:
        range_doppler_map: Input range-Doppler map
        range_migration_compensation: Whether to apply range migration compensation
        
    Returns:
        Compensated range-Doppler map
    """
    compensated_map = range_doppler_map.copy()
    
    if range_migration_compensation:
        # Simple range migration compensation
        # In practice, this would involve more sophisticated processing
        for doppler_bin in range(compensated_map.shape[1]):
            # Calculate range shift based on Doppler bin
            range_shift = (doppler_bin - compensated_map.shape[1] // 2) * 0.1
            shift_bins = int(round(range_shift))
            
            if shift_bins != 0:
                compensated_map[:, doppler_bin] = np.roll(
                    compensated_map[:, doppler_bin], -shift_bins
                )
    
    return compensated_map


def extract_roi(data: np.ndarray, 
               center: Tuple[int, int], 
               size: Tuple[int, int]) -> np.ndarray:
    """
    Extract Region of Interest (ROI) from 2D data.
    
    Args:
        data: Input 2D data
        center: Center coordinates (row, col)
        size: ROI size (height, width)
        
    Returns:
        Extracted ROI
    """
    center_row, center_col = center
    height, width = size
    
    # Calculate ROI bounds
    row_start = max(0, center_row - height // 2)
    row_end = min(data.shape[0], center_row + height // 2)
    col_start = max(0, center_col - width // 2)
    col_end = min(data.shape[1], center_col + width // 2)
    
    return data[row_start:row_end, col_start:col_end]


class MovingAverage:
    """Moving average filter for real-time processing."""
    
    def __init__(self, window_size: int):
        """
        Initialize moving average filter.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.buffer = []
        self.sum = 0.0
    
    def update(self, value: float) -> float:
        """
        Update moving average with new value.
        
        Args:
            value: New input value
            
        Returns:
            Current moving average
        """
        self.buffer.append(value)
        self.sum += value
        
        if len(self.buffer) > self.window_size:
            removed_value = self.buffer.pop(0)
            self.sum -= removed_value
        
        return self.sum / len(self.buffer)
    
    def reset(self):
        """Reset the moving average filter."""
        self.buffer = []
        self.sum = 0.0


class ExponentialMovingAverage:
    """Exponential moving average filter for real-time processing."""
    
    def __init__(self, alpha: float):
        """
        Initialize exponential moving average filter.
        
        Args:
            alpha: Smoothing factor (0 < alpha <= 1)
        """
        if not 0 < alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1")
        
        self.alpha = alpha
        self.ema_value = None
    
    def update(self, value: float) -> float:
        """
        Update exponential moving average with new value.
        
        Args:
            value: New input value
            
        Returns:
            Current exponential moving average
        """
        if self.ema_value is None:
            self.ema_value = value
        else:
            self.ema_value = self.alpha * value + (1 - self.alpha) * self.ema_value
        
        return self.ema_value
    
    def reset(self):
        """Reset the exponential moving average filter."""
        self.ema_value = None
