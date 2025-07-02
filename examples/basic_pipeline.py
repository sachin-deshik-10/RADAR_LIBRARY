#!/usr/bin/env python3
"""
Basic Radar Processing Pipeline Example

This example demonstrates a complete radar processing pipeline from
raw ADC data to final visualizations including:
1. FMCW signal processing
2. Detection using CFAR
3. Multi-target tracking
4. Visualization and analysis

Author: Radar Perception Library Contributors
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# Import radar perception library
import sys
sys.path.append('../src')
import radar_perception as rp


def generate_synthetic_adc_data(config):
    """Generate synthetic ADC data with multiple targets."""
    num_samples = config['num_samples']
    num_chirps = config['num_chirps']
    num_rx = config.get('num_rx_antennas', 4)
    
    # Create synthetic multi-target scenario
    targets = [
        {'range': 25.0, 'velocity': 5.0, 'rcs': 10.0, 'angle': 0.0},
        {'range': 45.0, 'velocity': -3.0, 'rcs': 15.0, 'angle': 15.0},
        {'range': 70.0, 'velocity': 8.0, 'rcs': 8.0, 'angle': -10.0}
    ]
    
    # Generate clean signal + noise
    adc_data = np.random.normal(0, 0.1, (num_samples, num_chirps, num_rx)) + \
               1j * np.random.normal(0, 0.1, (num_samples, num_chirps, num_rx))
    
    # Add target signals (simplified model)
    for target in targets:
        # Convert physical parameters to bin indices
        range_bin = int(target['range'] / config['range_resolution'])
        velocity_bin = int(target['velocity'] / config['velocity_resolution']) + num_chirps // 2
        
        if 0 <= range_bin < num_samples and 0 <= velocity_bin < num_chirps:
            # Add target signal with some spread
            amplitude = np.sqrt(target['rcs'])
            for dr in range(-2, 3):
                for dv in range(-2, 3):
                    r_idx = range_bin + dr
                    v_idx = velocity_bin + dv
                    if 0 <= r_idx < num_samples and 0 <= v_idx < num_chirps:
                        signal_amp = amplitude * np.exp(-(dr*dr + dv*dv) / 2.0)
                        adc_data[r_idx, v_idx, :] += signal_amp * (1 + 0.1j)
    
    return adc_data


def main():
    """Main processing pipeline."""
    print("🎯 Radar Perception Library - Basic Processing Pipeline Example")
    print("=" * 60)
    
    # 1. Configuration
    print("\n📋 Step 1: Configuration")
    config = {
        'sample_rate': 10e6,        # 10 MHz
        'bandwidth': 4e9,           # 4 GHz
        'num_chirps': 128,          # Number of chirps per frame
        'num_samples': 256,         # Samples per chirp
        'num_rx_antennas': 4,       # Number of RX antennas
        'range_resolution': 0.2,    # Range resolution (m)
        'velocity_resolution': 0.1, # Velocity resolution (m/s)
        'frame_rate': 10.0          # Frame rate (Hz)
    }
    
    print(f"   • Sample Rate: {config['sample_rate']/1e6:.1f} MHz")
    print(f"   • Bandwidth: {config['bandwidth']/1e9:.1f} GHz")
    print(f"   • Frame Size: {config['num_samples']} x {config['num_chirps']}")
    print(f"   • Range Resolution: {config['range_resolution']} m")
    print(f"   • Velocity Resolution: {config['velocity_resolution']} m/s")
    
    # 2. Initialize Processing Components
    print("\n⚙️ Step 2: Initialize Processing Components")
    
    # FMCW Processor
    processor = rp.FMCWProcessor(config)
    print("   • FMCW Processor initialized")
    
    # CFAR Detector
    detector = rp.CFARDetector(
        guard_cells=2,
        training_cells=16,
        false_alarm_rate=1e-6,
        cfar_type='CA'
    )
    print("   • CFAR Detector initialized")
    
    # Multi-Target Tracker
    tracker = rp.MultiTargetTracker(
        max_association_distance=5.0,
        min_detections_for_confirmation=3,
        max_missed_detections=5
    )
    print("   • Multi-Target Tracker initialized")
    
    # 3. Generate Synthetic Data
    print("\n🎲 Step 3: Generate Synthetic Radar Data")
    adc_data = generate_synthetic_adc_data(config)
    print(f"   • Generated ADC data: {adc_data.shape}")
    print("   • Synthetic scenario: 3 targets at different ranges and velocities")
    
    # 4. Signal Processing
    print("\n🔄 Step 4: Signal Processing")
    start_time = time.time()
    
    # Range processing
    range_fft = processor.range_processing(adc_data)
    print("   • Range FFT completed")
    
    # Doppler processing
    range_doppler_map = processor.doppler_processing(range_fft)
    print("   • Doppler FFT completed")
    
    # CFAR processing
    cfar_processor = processor.get_cfar_processor()
    detections_binary = cfar_processor.process_2d(range_doppler_map)
    print("   • CFAR processing completed")
    
    processing_time = time.time() - start_time
    print(f"   • Total processing time: {processing_time*1000:.1f} ms")
    
    # 5. Detection
    print("\n🎯 Step 5: Target Detection")
    detections = detector.detect_2d(range_doppler_map)
    print(f"   • Found {len(detections)} detections")
    
    # Convert to physical units
    detections_physical = rp.detection.convert_detections_to_physical(
        detections,
        config['range_resolution'],
        config['velocity_resolution']
    )
    
    for i, det in enumerate(detections_physical):
        print(f"     - Detection {i+1}: Range={det.range_m:.1f}m, "
              f"Velocity={det.velocity_mps:.1f}m/s, SNR={det.snr_db:.1f}dB")
    
    # 6. Tracking
    print("\n📍 Step 6: Multi-Target Tracking")
    timestamp = 0.1  # 100ms timestamp
    tracks = tracker.update(detections_physical, timestamp)
    print(f"   • Active tracks: {len(tracks)}")
    
    for track in tracks:
        print(f"     - Track {track.id[-4:]}: State={track.state.value}, "
              f"Position=({track.position[0]:.1f}, {track.position[1]:.1f}), "
              f"Detections={track.detection_count}")
    
    # 7. Visualization
    print("\n📊 Step 7: Visualization")
    
    # Create comprehensive dashboard
    dashboard_fig = rp.create_radar_dashboard(
        range_doppler_map,
        detections_physical,
        tracks,
        config['range_resolution'],
        config['velocity_resolution'],
        timestamp
    )
    dashboard_fig.savefig('radar_dashboard.png', dpi=150, bbox_inches='tight')
    print("   • Dashboard saved as 'radar_dashboard.png'")
    
    # Range-Doppler map with detections
    rd_fig = rp.plot_detections_on_rd_map(
        range_doppler_map,
        detections_physical,
        config['range_resolution'],
        config['velocity_resolution'],
        title="Range-Doppler Map with CFAR Detections"
    )
    rd_fig.savefig('range_doppler_detections.png', dpi=150, bbox_inches='tight')
    print("   • Range-Doppler plot saved as 'range_doppler_detections.png'")
    
    # Cartesian plot
    cartesian_fig = rp.visualization.plot_cartesian_detections(
        detections_physical,
        tracks,
        title="Cartesian View - Detections and Tracks"
    )
    cartesian_fig.savefig('cartesian_view.png', dpi=150, bbox_inches='tight')
    print("   • Cartesian view saved as 'cartesian_view.png'")
    
    # Detection statistics
    stats_fig = rp.visualization.plot_detection_statistics(
        detections_physical,
        title="Detection Statistics Analysis"
    )
    stats_fig.savefig('detection_statistics.png', dpi=150, bbox_inches='tight')
    print("   • Statistics plot saved as 'detection_statistics.png'")
    
    # 8. Performance Analysis
    print("\n⚡ Step 8: Performance Analysis")
    
    # Calculate processing rates
    frame_size_mb = adc_data.nbytes / (1024 * 1024)
    processing_rate = frame_size_mb / processing_time
    
    print(f"   • Frame size: {frame_size_mb:.2f} MB")
    print(f"   • Processing rate: {processing_rate:.1f} MB/s")
    print(f"   • Frames per second: {1/processing_time:.1f} FPS")
    print(f"   • Memory usage: ~{frame_size_mb * 3:.1f} MB (input + intermediate)")
    
    # Detection performance
    detection_rate = len(detections) / (config['num_samples'] * config['num_chirps'])
    print(f"   • Detection rate: {detection_rate*100:.3f}% of range-Doppler cells")
    
    # 9. Summary
    print("\n✅ Processing Pipeline Complete!")
    print("=" * 60)
    print(f"📈 Results Summary:")
    print(f"   • Processed 1 radar frame ({config['num_samples']}x{config['num_chirps']})")
    print(f"   • Detected {len(detections)} targets")
    print(f"   • Tracking {len(tracks)} active tracks")
    print(f"   • Processing time: {processing_time*1000:.1f} ms")
    print(f"   • Generated 4 visualization plots")
    
    print(f"\n📁 Output Files:")
    print(f"   • radar_dashboard.png - Comprehensive overview")
    print(f"   • range_doppler_detections.png - Range-Doppler analysis")
    print(f"   • cartesian_view.png - Spatial visualization")
    print(f"   • detection_statistics.png - Statistical analysis")
    
    print(f"\n🎯 Next Steps:")
    print(f"   • Try with real radar data")
    print(f"   • Experiment with different CFAR parameters")
    print(f"   • Add more targets to test tracking performance")
    print(f"   • Explore multi-sensor fusion capabilities")
    
    # Keep plots open for interactive viewing
    print(f"\n👀 Close the plot windows to exit...")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🏁 Example complete.")
