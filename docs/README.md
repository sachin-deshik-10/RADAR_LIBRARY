# Radar Perception Library Documentation

Welcome to the comprehensive documentation for the Radar Perception Library - a modern, high-performance Python library for radar-based perception systems.

## Quick Start

### Installation

```bash
pip install radar-perception
```

### Basic Usage

```python
import radar_perception as rp

# Create FMCW processor
config = {
    'sample_rate': 1e6,
    'bandwidth': 4e9,
    'num_chirps': 128,
    'num_samples': 256
}
processor = rp.FMCWProcessor(config)

# Process radar data
range_doppler_map = processor.process_frame(adc_data)

# Detect targets
detector = rp.CFARDetector(guard_cells=2, training_cells=16)
detections = detector.detect_2d(range_doppler_map)

# Track targets
tracker = rp.MultiTargetTracker()
tracks = tracker.update(detections, timestamp)

# Visualize results
fig = rp.create_radar_dashboard(range_doppler_map, detections, tracks)
```

## Documentation Sections

### API Reference

- [Signal Processing](api/signal_processing.md) - FMCW processing, windowing, calibration
- [Detection](api/detection.md) - CFAR, peak detection, clustering algorithms
- [Tracking](api/tracking.md) - Kalman filtering, multi-target tracking
- [Fusion](api/fusion.md) - Multi-sensor data fusion and coordinate transforms
- [Datasets](api/datasets.md) - Dataset loading, synthetic data generation
- [Utilities](api/utils.md) - Helper functions and coordinate conversions
- [Visualization](api/visualization.md) - Plotting and animation tools

### Tutorials

- [Getting Started](tutorials/getting_started.md) - First steps with the library
- [FMCW Radar Basics](tutorials/fmcw_basics.md) - Understanding FMCW radar processing
- [Detection Algorithms](tutorials/detection.md) - Working with CFAR and other detectors
- [Multi-Target Tracking](tutorials/tracking.md) - Implementing target tracking systems
- [Sensor Fusion](tutorials/fusion.md) - Combining multiple radar sensors
- [Custom Datasets](tutorials/datasets.md) - Loading and creating radar datasets

### Examples

- [Basic Processing Pipeline](examples/basic_pipeline.py) - Complete processing example
- [Real-time Processing](examples/realtime.py) - Live radar data processing
- [Multi-sensor Fusion](examples/multi_sensor.py) - Fusion of multiple radars
- [Custom Detection Algorithm](examples/custom_detector.py) - Implementing custom detectors
- [Performance Benchmarking](examples/benchmarks.py) - Performance analysis tools

### Advanced Topics

- [Algorithm Design](advanced/algorithms.md) - Designing custom algorithms
- [Performance Optimization](advanced/optimization.md) - Optimizing processing pipelines
- [Hardware Integration](advanced/hardware.md) - Interfacing with radar hardware
- [Research Applications](advanced/research.md) - Using the library for research

## Key Features

### 🚀 High Performance

- Optimized NumPy and SciPy implementations
- Vectorized operations for maximum speed
- Memory-efficient processing pipelines
- GPU acceleration support (optional)

### 🔧 Comprehensive Algorithms

- **Signal Processing**: FMCW processing, windowing, calibration
- **Detection**: CFAR variants, adaptive thresholding, clustering
- **Tracking**: Kalman filters, IMM, multi-hypothesis tracking
- **Fusion**: Multi-sensor alignment, temporal synchronization

### 📊 Rich Visualization

- Interactive range-Doppler maps
- Real-time tracking displays
- Comprehensive dashboards
- Animation and export capabilities

### 🛠 Developer Friendly

- Clean, Pythonic API design
- Comprehensive documentation
- Extensive test coverage
- Type hints throughout

### 🔬 Research Ready

- Synthetic data generation
- Benchmarking utilities
- Algorithm comparison tools
- Publication-quality visualizations

## System Requirements

- Python 3.8 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0
- Optional: CuPy for GPU acceleration

## Contributing

We welcome contributions! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and standards
- Testing requirements
- Submitting pull requests

## Support

- 📚 **Documentation**: [https://radar-perception.readthedocs.io](https://radar-perception.readthedocs.io)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/radar-perception/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/radar-perception/discussions)
- 📧 **Email**: <contact@radarperception.dev>

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{radar_perception_2024,
  title={Radar Perception Library: A Comprehensive Python Framework for Radar-Based Perception},
  author={Radar Perception Library Contributors},
  year={2024},
  url={https://github.com/yourusername/radar-perception},
  version={1.0.0}
}
```

---

*Built with ❤️ for the radar perception community*
