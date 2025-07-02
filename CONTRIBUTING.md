# Contributing to Radar Perception Library

Thank you for your interest in contributing to the Radar Perception Library! This document provides guidelines for contributing to the project.

## 🚀 How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use the issue template** to provide necessary information
3. **Include relevant details** such as:
   - Operating system and version
   - Python version and dependencies
   - Error messages and stack traces
   - Steps to reproduce the issue

### Submitting Pull Requests

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a clear description

## 📋 Development Setup

### Prerequisites

```bash
# Install Python 3.8+
python --version

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=radar_perception tests/

# Run specific test categories
pytest tests/test_signal_processing.py
pytest tests/test_detection.py
```

## 🎯 Contribution Areas

### High Priority

- [ ] Algorithm implementations
- [ ] Dataset integrations
- [ ] Performance optimizations
- [ ] Documentation improvements

### Medium Priority

- [ ] Example notebooks
- [ ] Visualization tools
- [ ] Benchmarking utilities
- [ ] Edge deployment support

### Future Work

- [ ] Real-time processing
- [ ] Cloud integration
- [ ] Mobile applications
- [ ] Hardware abstractions

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear function names and docstrings
def process_radar_data(raw_data: np.ndarray, config: dict) -> np.ndarray:
    """
    Process raw radar data to extract point cloud.
    
    Args:
        raw_data: Raw ADC data from radar sensor
        config: Processing configuration parameters
        
    Returns:
        Processed point cloud data
    """
    # Implementation here
    pass

# Good: Type hints and error handling
def detect_objects(points: np.ndarray) -> List[Detection]:
    """Detect objects in radar point cloud."""
    if points.size == 0:
        raise ValueError("Input points cannot be empty")
    
    # Process and return detections
    return detections
```

### Documentation Standards

```python
class RadarProcessor:
    """
    Main radar signal processing class.
    
    This class provides methods for processing FMCW radar data
    including range-Doppler processing, CFAR detection, and
    point cloud generation.
    
    Attributes:
        config: Processing configuration
        calibration: Radar calibration parameters
        
    Example:
        >>> processor = RadarProcessor(config)
        >>> detections = processor.process(raw_data)
    """
    
    def __init__(self, config: dict):
        """Initialize radar processor with configuration."""
        self.config = config
```

## 🧪 Testing Guidelines

### Unit Tests

```python
import pytest
import numpy as np
from radar_perception import RadarProcessor

class TestRadarProcessor:
    def test_process_valid_data(self):
        """Test processing with valid input data."""
        processor = RadarProcessor(config={})
        data = np.random.rand(256, 128)
        result = processor.process(data)
        assert result is not None
        
    def test_process_empty_data(self):
        """Test processing with empty input."""
        processor = RadarProcessor(config={})
        with pytest.raises(ValueError):
            processor.process(np.array([]))
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete radar processing pipeline."""
    # Load test data
    data = load_test_dataset('sample_radar_data.h5')
    
    # Process data
    processor = RadarProcessor(config)
    detections = processor.process(data)
    
    # Verify results
    assert len(detections) > 0
    assert all(d.confidence > 0.5 for d in detections)
```

## 📊 Benchmarking

### Performance Tests

```python
import time
import pytest

@pytest.mark.benchmark
def test_processing_speed():
    """Benchmark radar processing speed."""
    processor = RadarProcessor(config)
    data = generate_test_data(size='large')
    
    start_time = time.time()
    result = processor.process(data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    assert processing_time < 0.1  # Should process in < 100ms
```

### Memory Usage

```python
import psutil
import pytest

@pytest.mark.memory
def test_memory_usage():
    """Test memory usage during processing."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    processor = RadarProcessor(config)
    data = generate_test_data(size='large')
    result = processor.process(data)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Should not use more than 500MB additional memory
    assert memory_increase < 500 * 1024 * 1024
```

## 📚 Documentation

### Updating Documentation

1. **API Documentation**: Update docstrings for new functions
2. **User Guides**: Add tutorials for new features
3. **Examples**: Provide working code examples
4. **README**: Update feature lists and usage instructions

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## 🏷️ Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] Update version number in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release tag
- [ ] Publish to PyPI

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn and contribute
- Follow project guidelines and standards

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Pull Requests**: Code contributions and reviews

## 📧 Getting Help

If you need help contributing:

1. Check existing documentation and examples
2. Search GitHub issues and discussions
3. Create a new issue with the "help wanted" label
4. Join our community discussions

Thank you for contributing to the Radar Perception Library! 🎉
