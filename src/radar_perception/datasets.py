"""
Dataset utilities for radar perception.

This module provides utilities for loading, processing, and managing
radar datasets commonly used in research and development.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Iterator, Union
from dataclasses import dataclass, field
from pathlib import Path
import h5py
from abc import ABC, abstractmethod

from .detection import Detection
from .tracking import Track


@dataclass
class RadarFrame:
    """Represents a single radar frame with associated data."""
    timestamp: float
    range_doppler_map: np.ndarray
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_adc_data: Optional[np.ndarray] = None
    angle_data: Optional[np.ndarray] = None


@dataclass
class DatasetMetadata:
    """Metadata for radar datasets."""
    name: str
    description: str
    num_frames: int
    frame_rate: float
    range_resolution: float
    velocity_resolution: float
    angle_resolution: Optional[float] = None
    frequency_ghz: float = 77.0
    antenna_config: Optional[Dict] = None
    scenario_info: Optional[Dict] = None


class RadarDataset(ABC):
    """
    Abstract base class for radar datasets.
    
    Provides a common interface for different radar dataset formats
    and sources.
    """
    
    def __init__(self, data_path: str, metadata: Optional[DatasetMetadata] = None):
        """
        Initialize radar dataset.
        
        Args:
            data_path: Path to dataset directory or file
            metadata: Dataset metadata
        """
        self.data_path = Path(data_path)
        self.metadata = metadata
        self._validate_dataset()
    
    @abstractmethod
    def _validate_dataset(self):
        """Validate dataset structure and files."""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return number of frames in dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, index: int) -> RadarFrame:
        """Get frame by index."""
        pass
    
    @abstractmethod
    def get_frame_by_timestamp(self, timestamp: float) -> Optional[RadarFrame]:
        """Get frame by timestamp."""
        pass
    
    def __iter__(self) -> Iterator[RadarFrame]:
        """Iterate over frames."""
        for i in range(len(self)):
            yield self[i]
    
    def get_metadata(self) -> Optional[DatasetMetadata]:
        """Get dataset metadata."""
        return self.metadata
    
    def get_frame_range(self, start_idx: int, end_idx: int) -> List[RadarFrame]:
        """Get a range of frames."""
        return [self[i] for i in range(start_idx, min(end_idx, len(self)))]
    
    def get_time_range(self, start_time: float, end_time: float) -> List[RadarFrame]:
        """Get frames within a time range."""
        frames = []
        for frame in self:
            if start_time <= frame.timestamp <= end_time:
                frames.append(frame)
        return frames


class HDF5RadarDataset(RadarDataset):
    """
    HDF5-based radar dataset implementation.
    
    Supports efficient storage and loading of large radar datasets
    using HDF5 format.
    """
    
    def __init__(self, data_path: str, metadata: Optional[DatasetMetadata] = None):
        """Initialize HDF5 radar dataset."""
        super().__init__(data_path, metadata)
        self.hdf5_file = None
        self._open_dataset()
    
    def _validate_dataset(self):
        """Validate HDF5 dataset structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        if not self.data_path.suffix == '.h5':
            raise ValueError("Expected HDF5 file with .h5 extension")
    
    def _open_dataset(self):
        """Open HDF5 dataset file."""
        try:
            self.hdf5_file = h5py.File(self.data_path, 'r')
            
            # Load metadata if available
            if 'metadata' in self.hdf5_file.attrs and self.metadata is None:
                metadata_dict = json.loads(self.hdf5_file.attrs['metadata'])
                self.metadata = DatasetMetadata(**metadata_dict)
        
        except Exception as e:
            raise RuntimeError(f"Failed to open HDF5 dataset: {e}")
    
    def __len__(self) -> int:
        """Return number of frames."""
        if self.hdf5_file is None:
            return 0
        return self.hdf5_file['range_doppler_maps'].shape[0]
    
    def __getitem__(self, index: int) -> RadarFrame:
        """Get frame by index."""
        if self.hdf5_file is None:
            raise RuntimeError("Dataset not open")
        
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        
        # Load basic data
        timestamp = float(self.hdf5_file['timestamps'][index])
        range_doppler_map = self.hdf5_file['range_doppler_maps'][index]
        
        # Load optional data
        raw_adc_data = None
        if 'raw_adc_data' in self.hdf5_file:
            raw_adc_data = self.hdf5_file['raw_adc_data'][index]
        
        angle_data = None
        if 'angle_data' in self.hdf5_file:
            angle_data = self.hdf5_file['angle_data'][index]
        
        # Load detections if available
        detections = []
        if f'detections_{index}' in self.hdf5_file:
            det_data = self.hdf5_file[f'detections_{index}']
            for i in range(len(det_data)):
                detection = Detection(
                    range_bin=int(det_data[i]['range_bin']),
                    doppler_bin=int(det_data[i]['doppler_bin']),
                    range_m=float(det_data[i]['range_m']),
                    velocity_mps=float(det_data[i]['velocity_mps']),
                    snr_db=float(det_data[i]['snr_db']),
                    magnitude=float(det_data[i]['magnitude']),
                    timestamp=timestamp
                )
                detections.append(detection)
        
        # Load metadata
        metadata = {}
        if f'metadata_{index}' in self.hdf5_file.attrs:
            metadata = json.loads(self.hdf5_file.attrs[f'metadata_{index}'])
        
        return RadarFrame(
            timestamp=timestamp,
            range_doppler_map=range_doppler_map,
            detections=detections,
            raw_adc_data=raw_adc_data,
            angle_data=angle_data,
            metadata=metadata
        )
    
    def get_frame_by_timestamp(self, timestamp: float) -> Optional[RadarFrame]:
        """Get frame by timestamp."""
        if self.hdf5_file is None:
            return None
        
        timestamps = self.hdf5_file['timestamps'][:]
        closest_idx = np.argmin(np.abs(timestamps - timestamp))
        
        if np.abs(timestamps[closest_idx] - timestamp) < 0.1:  # 100ms tolerance
            return self[closest_idx]
        
        return None
    
    def close(self):
        """Close HDF5 file."""
        if self.hdf5_file is not None:
            self.hdf5_file.close()
            self.hdf5_file = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


class DirectoryRadarDataset(RadarDataset):
    """
    Directory-based radar dataset implementation.
    
    Loads radar data from a directory structure with individual files
    for each frame or component.
    """
    
    def __init__(self, data_path: str, 
                 file_pattern: str = "frame_{:06d}.npy",
                 metadata: Optional[DatasetMetadata] = None):
        """
        Initialize directory radar dataset.
        
        Args:
            data_path: Path to dataset directory
            file_pattern: Pattern for frame files
            metadata: Dataset metadata
        """
        self.file_pattern = file_pattern
        self.frame_files = []
        super().__init__(data_path, metadata)
    
    def _validate_dataset(self):
        """Validate directory dataset structure."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_path}")
        
        if not self.data_path.is_dir():
            raise ValueError("Expected directory path")
        
        # Find all frame files
        self.frame_files = sorted(list(self.data_path.glob("frame_*.npy")))
        
        if not self.frame_files:
            raise ValueError("No frame files found in dataset directory")
    
    def __len__(self) -> int:
        """Return number of frames."""
        return len(self.frame_files)
    
    def __getitem__(self, index: int) -> RadarFrame:
        """Get frame by index."""
        if index >= len(self.frame_files):
            raise IndexError(f"Index {index} out of range")
        
        frame_file = self.frame_files[index]
        
        # Load range-Doppler map
        range_doppler_map = np.load(frame_file)
        
        # Extract timestamp from filename or use index
        timestamp = float(index) * (1.0 / self.metadata.frame_rate) if self.metadata else float(index)
        
        # Look for detection file
        det_file = frame_file.parent / f"detections_{frame_file.stem.split('_')[1]}.json"
        detections = []
        if det_file.exists():
            with open(det_file, 'r') as f:
                det_data = json.load(f)
                for det in det_data:
                    detection = Detection(**det, timestamp=timestamp)
                    detections.append(detection)
        
        return RadarFrame(
            timestamp=timestamp,
            range_doppler_map=range_doppler_map,
            detections=detections
        )
    
    def get_frame_by_timestamp(self, timestamp: float) -> Optional[RadarFrame]:
        """Get frame by timestamp."""
        if self.metadata is None:
            return None
        
        frame_idx = int(timestamp * self.metadata.frame_rate)
        if 0 <= frame_idx < len(self):
            return self[frame_idx]
        
        return None


class SyntheticRadarDataset(RadarDataset):
    """
    Synthetic radar dataset generator.
    
    Generates synthetic radar data for testing and development purposes.
    """
    
    def __init__(self, 
                 num_frames: int = 1000,
                 num_targets: int = 3,
                 range_bins: int = 256,
                 doppler_bins: int = 128,
                 noise_level: float = 0.1,
                 metadata: Optional[DatasetMetadata] = None):
        """
        Initialize synthetic radar dataset.
        
        Args:
            num_frames: Number of frames to generate
            num_targets: Number of synthetic targets
            range_bins: Number of range bins
            doppler_bins: Number of Doppler bins
            noise_level: Noise level (0-1)
            metadata: Dataset metadata
        """
        self.num_frames = num_frames
        self.num_targets = num_targets
        self.range_bins = range_bins
        self.doppler_bins = doppler_bins
        self.noise_level = noise_level
        
        # Generate target trajectories
        self.target_trajectories = self._generate_trajectories()
        
        # Create default metadata if not provided
        if metadata is None:
            metadata = DatasetMetadata(
                name="synthetic_radar_dataset",
                description="Synthetic radar dataset for testing",
                num_frames=num_frames,
                frame_rate=10.0,
                range_resolution=0.2,
                velocity_resolution=0.1,
                frequency_ghz=77.0
            )
        
        super().__init__("synthetic", metadata)
    
    def _validate_dataset(self):
        """Validate synthetic dataset parameters."""
        if self.num_frames <= 0:
            raise ValueError("Number of frames must be positive")
        if self.num_targets < 0:
            raise ValueError("Number of targets must be non-negative")
    
    def _generate_trajectories(self) -> List[List[Tuple[float, float, float]]]:
        """Generate synthetic target trajectories."""
        trajectories = []
        
        for target_id in range(self.num_targets):
            trajectory = []
            
            # Random initial position and velocity
            initial_range = np.random.uniform(10, 100)
            initial_velocity = np.random.uniform(-10, 10)
            
            for frame_idx in range(self.num_frames):
                time_step = frame_idx * 0.1  # 10 FPS
                
                # Simple linear motion with some noise
                range_m = initial_range + initial_velocity * time_step + \
                         np.random.normal(0, 0.5)
                velocity_mps = initial_velocity + np.random.normal(0, 0.2)
                
                # Random angle
                angle_deg = np.random.uniform(-60, 60)
                
                trajectory.append((range_m, velocity_mps, angle_deg))
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def __len__(self) -> int:
        """Return number of frames."""
        return self.num_frames
    
    def __getitem__(self, index: int) -> RadarFrame:
        """Generate frame by index."""
        if index >= self.num_frames:
            raise IndexError(f"Index {index} out of range")
        
        timestamp = index * 0.1  # 10 FPS
        
        # Generate range-Doppler map
        range_doppler_map = self._generate_range_doppler_map(index)
        
        # Generate detections
        detections = self._generate_detections(index, timestamp)
        
        return RadarFrame(
            timestamp=timestamp,
            range_doppler_map=range_doppler_map,
            detections=detections,
            metadata={'frame_index': index, 'synthetic': True}
        )
    
    def _generate_range_doppler_map(self, frame_idx: int) -> np.ndarray:
        """Generate synthetic range-Doppler map."""
        # Start with noise
        rd_map = np.random.normal(0, self.noise_level, 
                                (self.range_bins, self.doppler_bins))
        
        # Add targets
        for target_id, trajectory in enumerate(self.target_trajectories):
            range_m, velocity_mps, angle_deg = trajectory[frame_idx]
            
            # Convert to bin indices
            range_bin = int(range_m / self.metadata.range_resolution)
            velocity_bin = int(velocity_mps / self.metadata.velocity_resolution) + self.doppler_bins // 2
            
            # Add target signal
            if (0 <= range_bin < self.range_bins and 
                0 <= velocity_bin < self.doppler_bins):
                # Add target with some spread
                for dr in range(-2, 3):
                    for dv in range(-2, 3):
                        r_idx = range_bin + dr
                        v_idx = velocity_bin + dv
                        if (0 <= r_idx < self.range_bins and 
                            0 <= v_idx < self.doppler_bins):
                            signal_strength = 1.0 * np.exp(-(dr*dr + dv*dv) / 2.0)
                            rd_map[r_idx, v_idx] += signal_strength
        
        return rd_map
    
    def _generate_detections(self, frame_idx: int, timestamp: float) -> List[Detection]:
        """Generate synthetic detections."""
        detections = []
        
        for target_id, trajectory in enumerate(self.target_trajectories):
            range_m, velocity_mps, angle_deg = trajectory[frame_idx]
            
            # Convert to bin indices
            range_bin = int(range_m / self.metadata.range_resolution)
            velocity_bin = int(velocity_mps / self.metadata.velocity_resolution) + self.doppler_bins // 2
            
            if (0 <= range_bin < self.range_bins and 
                0 <= velocity_bin < self.doppler_bins):
                
                detection = Detection(
                    range_bin=range_bin,
                    doppler_bin=velocity_bin,
                    range_m=range_m,
                    velocity_mps=velocity_mps,
                    angle_deg=angle_deg,
                    snr_db=20.0 + np.random.normal(0, 3),
                    magnitude=1.0 + np.random.normal(0, 0.1),
                    timestamp=timestamp
                )
                detections.append(detection)
        
        return detections
    
    def get_frame_by_timestamp(self, timestamp: float) -> Optional[RadarFrame]:
        """Get frame by timestamp."""
        frame_idx = int(timestamp * 10)  # 10 FPS
        if 0 <= frame_idx < self.num_frames:
            return self[frame_idx]
        return None


class DatasetSplitter:
    """Utility class for splitting datasets into train/validation/test sets."""
    
    @staticmethod
    def split_dataset(dataset: RadarDataset, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_seed: Optional[int] = None) -> Tuple[List[int], List[int], List[int]]:
        """
        Split dataset indices into train/validation/test sets.
        
        Args:
            dataset: Radar dataset to split
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        num_frames = len(dataset)
        indices = np.random.permutation(num_frames)
        
        train_end = int(num_frames * train_ratio)
        val_end = train_end + int(num_frames * val_ratio)
        
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()
        
        return train_indices, val_indices, test_indices


def create_hdf5_dataset(output_path: str, 
                       frames: List[RadarFrame],
                       metadata: Optional[DatasetMetadata] = None,
                       compression: str = 'gzip') -> None:
    """
    Create HDF5 dataset from list of frames.
    
    Args:
        output_path: Output HDF5 file path
        frames: List of radar frames
        metadata: Dataset metadata
        compression: HDF5 compression method
    """
    if not frames:
        raise ValueError("No frames provided")
    
    with h5py.File(output_path, 'w') as f:
        num_frames = len(frames)
        
        # Get dimensions from first frame
        rd_shape = frames[0].range_doppler_map.shape
        
        # Create datasets
        timestamps_ds = f.create_dataset('timestamps', (num_frames,), dtype='f')
        rd_maps_ds = f.create_dataset('range_doppler_maps', 
                                     (num_frames,) + rd_shape,
                                     compression=compression,
                                     dtype='f')
        
        # Store data
        for i, frame in enumerate(frames):
            timestamps_ds[i] = frame.timestamp
            rd_maps_ds[i] = frame.range_doppler_map
            
            # Store detections if available
            if frame.detections:
                det_group = f.create_group(f'detections_{i}')
                det_data = []
                for det in frame.detections:
                    det_dict = {
                        'range_bin': det.range_bin,
                        'doppler_bin': det.doppler_bin,
                        'range_m': det.range_m,
                        'velocity_mps': det.velocity_mps,
                        'snr_db': det.snr_db,
                        'magnitude': det.magnitude
                    }
                    det_data.append(det_dict)
                
                # Convert to structured array for HDF5 storage
                if det_data:
                    dt = np.dtype([
                        ('range_bin', 'i4'),
                        ('doppler_bin', 'i4'),
                        ('range_m', 'f4'),
                        ('velocity_mps', 'f4'),
                        ('snr_db', 'f4'),
                        ('magnitude', 'f4')
                    ])
                    det_array = np.array([tuple(d.values()) for d in det_data], dtype=dt)
                    f.create_dataset(f'detections_{i}', data=det_array)
            
            # Store metadata
            if frame.metadata:
                f.attrs[f'metadata_{i}'] = json.dumps(frame.metadata)
        
        # Store dataset metadata
        if metadata:
            metadata_dict = {
                'name': metadata.name,
                'description': metadata.description,
                'num_frames': metadata.num_frames,
                'frame_rate': metadata.frame_rate,
                'range_resolution': metadata.range_resolution,
                'velocity_resolution': metadata.velocity_resolution,
                'frequency_ghz': metadata.frequency_ghz
            }
            f.attrs['metadata'] = json.dumps(metadata_dict)


def load_dataset(data_path: str, 
                dataset_type: str = 'auto',
                **kwargs) -> RadarDataset:
    """
    Load radar dataset from path.
    
    Args:
        data_path: Path to dataset
        dataset_type: Type of dataset ('hdf5', 'directory', 'synthetic', 'auto')
        **kwargs: Additional arguments for dataset constructor
        
    Returns:
        Loaded radar dataset
    """
    data_path = Path(data_path)
    
    if dataset_type == 'auto':
        if data_path.suffix == '.h5':
            dataset_type = 'hdf5'
        elif data_path.is_dir():
            dataset_type = 'directory'
        else:
            raise ValueError(f"Cannot determine dataset type for: {data_path}")
    
    if dataset_type == 'hdf5':
        return HDF5RadarDataset(str(data_path), **kwargs)
    elif dataset_type == 'directory':
        return DirectoryRadarDataset(str(data_path), **kwargs)
    elif dataset_type == 'synthetic':
        return SyntheticRadarDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
