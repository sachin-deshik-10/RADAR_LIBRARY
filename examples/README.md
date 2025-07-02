# Radar Perception Examples

This directory contains practical examples and tutorials for using the Radar Perception Library.

## 📁 Directory Structure

```
examples/
├── basic_processing/           # Basic radar signal processing
│   ├── fmcw_processing.py     # FMCW radar processing pipeline
│   ├── cfar_detection.py      # CFAR detection implementation
│   └── point_cloud_gen.py     # Point cloud generation
├── object_detection/          # Object detection examples
│   ├── pointnet_radar.py      # PointNet for radar data
│   ├── range_azimuth_cnn.py   # CNN on range-azimuth maps
│   └── rad_tensor_detection.py # RAD tensor processing
├── sensor_fusion/             # Multi-modal fusion examples
│   ├── radar_camera_fusion.py # Early and late fusion
│   ├── radar_lidar_fusion.py  # Point cloud fusion
│   └── bev_fusion.py          # Bird's eye view fusion
├── tracking/                  # Tracking and SLAM examples
│   ├── kalman_tracking.py     # Kalman filter tracking
│   ├── radar_odometry.py      # Radar-based odometry
│   └── place_recognition.py   # Place recognition
├── datasets/                  # Dataset loading and processing
│   ├── nuscenes_loader.py     # nuScenes dataset utilities
│   ├── carrada_loader.py      # CARRADA dataset utilities
│   └── radiate_loader.py      # RADIATE dataset utilities
└── real_time/                 # Real-time processing examples
    ├── ti_mmwave_streaming.py # TI mmWave real-time processing
    ├── ros_integration.py     # ROS integration example
    └── edge_deployment.py     # Edge device deployment
```

## 🚀 Getting Started

### Prerequisites

```bash
# Install the radar perception library
pip install radar-perception-library

# Install additional dependencies for examples
pip install -r examples_requirements.txt
```

### Running Examples

1. **Basic Processing**: Start with FMCW signal processing

   ```bash
   python examples/basic_processing/fmcw_processing.py
   ```

2. **Object Detection**: Try radar-based object detection

   ```bash
   python examples/object_detection/pointnet_radar.py
   ```

3. **Sensor Fusion**: Explore multi-modal approaches

   ```bash
   python examples/sensor_fusion/radar_camera_fusion.py
   ```

## 📚 Example Categories

### Basic Signal Processing

Learn the fundamentals of radar signal processing including:

- Range-Doppler processing
- CFAR detection algorithms
- Point cloud generation
- Noise filtering and calibration

### Object Detection

Implement various detection approaches:

- Traditional clustering methods
- Deep learning on point clouds
- CNN-based range-azimuth processing
- Multi-frame temporal fusion

### Sensor Fusion

Combine radar with other sensors:

- Radar-camera early/late fusion
- Radar-LiDAR point cloud fusion
- Bird's eye view representations
- Cross-modal learning techniques

### Tracking and SLAM

Build complete perception systems:

- Multi-object tracking with Kalman filters
- Radar odometry and localization
- Place recognition and loop closure
- Long-term mapping applications

## 🔧 Development

### Adding New Examples

1. Create a new directory for your example category
2. Follow the naming convention: `category_name/example_name.py`
3. Include comprehensive documentation and comments
4. Add any required dependencies to `examples_requirements.txt`
5. Update this README with your example description

### Code Style

- Follow PEP 8 Python style guidelines
- Include docstrings for all functions and classes
- Add type hints where appropriate
- Provide clear examples and usage instructions

### Testing Examples

```bash
# Run all example tests
pytest examples/tests/

# Run specific category tests
pytest examples/tests/test_basic_processing.py
```

## 📖 Tutorials

Each example includes:

- **Objective**: What you'll learn
- **Prerequisites**: Required knowledge and setup
- **Step-by-step guide**: Detailed implementation walkthrough
- **Results**: Expected outputs and visualizations
- **Extensions**: Ideas for further development

## 🤝 Contributing

We welcome contributions of new examples! Please:

1. Ensure your example is well-documented
2. Include test data or synthetic data generation
3. Provide clear learning objectives
4. Follow the existing code structure and style

## 📞 Support

If you encounter issues with any examples:

1. Check the prerequisites and dependencies
2. Review the documentation and comments
3. Open an issue on GitHub with details about your setup
4. Join our community discussions for help

---

*These examples are designed to be educational and practical, providing hands-on experience with radar perception techniques.*
