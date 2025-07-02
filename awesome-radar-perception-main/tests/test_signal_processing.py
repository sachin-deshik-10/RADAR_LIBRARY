"""
Test suite for signal processing module.
"""

import pytest
import numpy as np
from radar_perception.signal_processing import FMCWProcessor, generate_point_cloud


class TestFMCWProcessor:
    """Test FMCW processor functionality."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'sample_rate': 1e6,
            'bandwidth': 4e9,
            'num_chirps': 128,
            'num_samples': 256,
            'range_resolution': 0.1,
            'velocity_resolution': 0.1,
        }
    
    @pytest.fixture
    def processor(self, config):
        """Create FMCW processor instance."""
        return FMCWProcessor(config)
    
    @pytest.fixture
    def test_data(self, config):
        """Generate test ADC data."""
        return np.random.rand(config['num_samples'], 
                             config['num_chirps'], 
                             4) + 1j * np.random.rand(config['num_samples'], 
                                                     config['num_chirps'], 
                                                     4)
    
    def test_processor_initialization(self, processor, config):
        """Test processor initialization."""
        assert processor.sample_rate == config['sample_rate']
        assert processor.bandwidth == config['bandwidth']
        assert processor.num_chirps == config['num_chirps']
        assert processor.num_samples == config['num_samples']
    
    def test_range_processing(self, processor, test_data):
        """Test range FFT processing."""
        range_data = processor.range_processing(test_data)
        
        assert range_data.shape == test_data.shape
        assert np.iscomplexobj(range_data)
    
    def test_doppler_processing(self, processor, test_data):
        """Test Doppler FFT processing."""
        range_data = processor.range_processing(test_data)
        rd_data = processor.doppler_processing(range_data)
        
        assert rd_data.shape == test_data.shape
        assert np.iscomplexobj(rd_data)
    
    def test_cfar_detection(self, processor, test_data):
        """Test CFAR detection."""
        range_data = processor.range_processing(test_data)
        rd_data = processor.doppler_processing(range_data)
        rd_magnitude = np.abs(rd_data[:, :, 0])
        
        detections, threshold_map = processor.cfar_detection(rd_magnitude)
        
        assert detections.shape == rd_magnitude.shape
        assert threshold_map.shape == rd_magnitude.shape
        assert detections.dtype == bool
    
    def test_point_cloud_generation_empty(self, config):
        """Test point cloud generation with no detections."""
        detections = np.zeros((100, 100), dtype=bool)
        rd_map = np.random.rand(100, 100)
        
        points = generate_point_cloud(detections, rd_map, config)
        
        assert points.shape == (0, 4)
    
    def test_point_cloud_generation_with_detections(self, config):
        """Test point cloud generation with detections."""
        detections = np.zeros((100, 100), dtype=bool)
        detections[50, 60] = True  # Single detection
        detections[30, 40] = True  # Another detection
        
        rd_map = np.random.rand(100, 100)
        
        points = generate_point_cloud(detections, rd_map, config)
        
        assert points.shape[0] == 2  # Two detections
        assert points.shape[1] == 4  # [x, y, z, intensity]
        assert np.all(points[:, 3] > 0)  # Positive intensities


class TestSignalProcessingUtilities:
    """Test utility functions."""
    
    def test_point_cloud_structure(self):
        """Test point cloud data structure."""
        detections = np.array([[True, False], [False, True]])
        rd_map = np.array([[0.5, 0.2], [0.1, 0.8]])
        config = {'range_resolution': 0.1, 'velocity_resolution': 0.1}
        
        points = generate_point_cloud(detections, rd_map, config)
        
        # Should have 2 points
        assert points.shape[0] == 2
        
        # Check coordinate ranges
        assert np.all(points[:, 0] >= 0)  # Positive range
        assert points.shape[1] == 4  # [x, y, z, intensity]


@pytest.mark.benchmark
def test_processing_performance(benchmark):
    """Benchmark signal processing performance."""
    config = {
        'sample_rate': 1e6,
        'bandwidth': 4e9,
        'num_chirps': 128,
        'num_samples': 256,
    }
    processor = FMCWProcessor(config)
    test_data = np.random.rand(256, 128, 4) + 1j * np.random.rand(256, 128, 4)
    
    def full_processing():
        range_data = processor.range_processing(test_data)
        rd_data = processor.doppler_processing(range_data)
        detections, _ = processor.cfar_detection(np.abs(rd_data[:, :, 0]))
        return detections
    
    result = benchmark(full_processing)
    assert result.shape == (256, 128)
