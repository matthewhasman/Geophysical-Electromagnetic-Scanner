#!/usr/bin/env python3
"""
Single Tone FFT Analyzer

This script generates a single frequency tone using the Analog Discovery hardware
and displays the complete FFT spectrum of the signal captured on the oscilloscope.
This is useful for analyzing spectral characteristics, harmonic distortion, or
system frequency response.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import time
import sys
from WaveformInterface import AnalogDiscovery

def analyze_tone(frequency, amplitude=1.0, offset=0, buffer_size=2**16, 
                sampling_rate=1e6, coherent_sampling=True, window_type=None,
                extended_buffer=False):
    """
    Generate a single tone and analyze its frequency spectrum
    
    Parameters:
    - frequency: tone frequency in Hz
    - amplitude: signal amplitude in Volts (default: 1.0V)
    - offset: DC offset in Volts (default: 0V)
    - buffer_size: number of samples to capture (default: 65536)
    - sampling_rate: initial sampling rate in Hz (default: 1MHz)
    - coherent_sampling: whether to adjust sampling for coherent capture (default: True)
    - window_type: FFT window type (None uses rectangular for coherent, FlatTop for non-coherent)
    - extended_buffer: whether to use extended buffer mode to exceed hardware limitations
    
    Returns:
    - freq_data: frequency points array
    - mag_data: magnitude spectrum in dB
    - phase_data: phase spectrum in degrees
    - raw_data: time domain raw data
    """
    # Initialize the device
    device = AnalogDiscovery()
    
    # Get hardware limits
    hw_max_buffer = device.max_buffer_size
    print(f"Hardware buffer limits: {device.min_buffer_size} to {hw_max_buffer} samples")
    
    # Check if extended buffer is needed
    if buffer_size > hw_max_buffer and not extended_buffer:
        print(f"Warning: Requested buffer size {buffer_size} exceeds hardware maximum {hw_max_buffer}")
        print("Enabling extended buffer mode to accommodate the request")
        extended_buffer = True
    
    # Adjust for coherent sampling if requested
    actual_freq = frequency
    
    if coherent_sampling:
        # Calculate parameters for coherent sampling
        # For coherent sampling, we need: buffer_size * frequency / sampling_rate = integer
        # Choose a reasonable number of cycles to fit in the buffer
        if frequency < 1000:
            cycles = max(4, min(16, int(buffer_size / 20)))
        else:
            cycles = max(16, min(buffer_size // 100, 1000))
        
        # Calculate sampling rate for exact cycle fit
        adjusted_rate = buffer_size * frequency / cycles
        
        # Limit to reasonable bounds
        if adjusted_rate < 100:
            adjusted_rate = 100
        elif adjusted_rate > device.max_freq:
            adjusted_rate = device.max_freq
            print(f"Warning: Adjusted sampling rate capped to hardware maximum of {device.max_freq} Hz")
            
        # Calculate actual frequency for exact bin alignment
        actual_freq = cycles * adjusted_rate / buffer_size
        sampling_rate = adjusted_rate
        
        print(f"Coherent sampling parameters:")
        print(f"  Desired frequency: {frequency:.2f} Hz")
        print(f"  Actual frequency: {actual_freq:.2f} Hz")
        print(f"  Sampling rate: {sampling_rate/1e6:.3f} MHz")
        print(f"  Cycles in buffer: {cycles}")
        print(f"  Bin number: {cycles}")
        
        # Select optimal window (rectangular is best for coherent sampling)
        if window_type is None:
            window_type = device.constants.DwfWindowRectangular
    else:
        print(f"Using standard sampling at {sampling_rate/1e6:.3f} MHz")
        if window_type is None:
            window_type = device.constants.DwfWindowFlatTop
    
    # Print buffer mode
    if extended_buffer:
        print(f"Using extended buffer mode with {buffer_size} samples (exceeds hardware limit of {hw_max_buffer})")
        print(f"Will use multiple acquisitions to build the complete buffer")
    else:
        actual_buffer = min(buffer_size, hw_max_buffer)
        if actual_buffer != buffer_size:
            print(f"Warning: Requested buffer size {buffer_size} exceeds hardware maximum {hw_max_buffer}")
            print(f"Limiting to {actual_buffer} samples")
            buffer_size = actual_buffer
        print(f"Using standard buffer mode with {buffer_size} samples")
    
    try:
        # Configure scope
        if not extended_buffer:
            device.init_scope(
                sampling_frequency=sampling_rate,
                buffer_size=buffer_size,
                offset=0,
                amplitude_range=5
            )
        else:
            # For extended buffer, initially configure with maximum hardware buffer
            device.init_scope(
                sampling_frequency=sampling_rate,
                buffer_size=hw_max_buffer,
                offset=0,
                amplitude_range=5
            )
        
        # Configure function generator
        device.generate(
            channel=1,
            function=device.constants.funcSine,
            offset=offset,
            frequency=actual_freq, 
            amplitude=amplitude
        )
        
        # Wait for output to stabilize
        stabilization_time = max(0.1, 10 / actual_freq)  # At least 10 cycles
        print(f"Waiting {stabilization_time:.3f}s for signal to stabilize...")
        time.sleep(stabilization_time)
        
        # Capture and analyze the signal
        print("Capturing and analyzing signal...")
        freq_data, mag_data, phase_data, raw_data = device.fft_analyze(
            channel=1,
            window_type=window_type,
            wait_for_stable=True,
            requested_buffer_size=buffer_size if extended_buffer else None
        )
        
        print(f"FFT completed with {len(raw_data)} samples, {len(freq_data)} frequency bins")
        print(f"Frequency resolution: {freq_data[1]-freq_data[0]:.3f} Hz")
        
        # Also capture channel 2 if available
        try:
            freq_data2, mag_data2, phase_data2, raw_data2 = device.fft_analyze(
                channel=2,
                window_type=window_type,
                wait_for_stable=False,
                requested_buffer_size=buffer_size if extended_buffer else None
            )
            ch2_available = True
        except Exception as e:
            print(f"Warning: Could not capture from channel 2: {e}")
            ch2_available = False
        
        print("Analysis complete.")
        return freq_data, mag_data, phase_data, raw_data, ch2_available, \
               (freq_data2, mag_data2, phase_data2, raw_data2) if ch2_available else None
               
    finally:
        # Clean up
        print("Closing device...")
        device.close()

def plot_results(freq_data, mag_data, phase_data, raw_data, ch2_data=None, 
                expected_freq=None, title=None):
    """
    Plot the FFT results with detailed analysis
    
    Parameters:
    - freq_data: frequency points array
    - mag_data: magnitude spectrum in dB
    - phase_data: phase spectrum in degrees
    - raw_data: time domain raw data
    - ch2_data: optional tuple with channel 2 data (freq, mag, phase, raw)
    - expected_freq: the expected frequency to highlight
    - title: optional plot title
    """
    # Create a figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Time domain plot
    ax_time = fig.add_subplot(gs[0, :])
    time_points = np.arange(len(raw_data)) / (freq_data[-1] * 2)
    ax_time.plot(time_points * 1000, raw_data, label='Channel 1')
    ax_time.set_xlabel('Time (ms)')
    ax_time.set_ylabel('Amplitude (V)')
    ax_time.set_title('Time Domain Signal')
    ax_time.grid(True)
    
    if ch2_data:
        ax_time.plot(time_points * 1000, ch2_data[3], label='Channel 2', alpha=0.7)
        ax_time.legend()
        
    # Get time domain stats
    rms = np.sqrt(np.mean(np.square(raw_data)))
    vpp = np.max(raw_data) - np.min(raw_data)
    
    # Add signal stats to time domain plot
    ax_time.text(0.02, 0.95, f'RMS: {rms:.3f}V\nVpp: {vpp:.3f}V', 
                 transform=ax_time.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Full spectrum FFT plot
    ax_fft = fig.add_subplot(gs[1, :])
    ax_fft.semilogx(freq_data, mag_data, label='Channel 1')
    ax_fft.set_xlabel('Frequency (Hz)')
    ax_fft.set_ylabel('Magnitude (dB)')
    ax_fft.set_title(f'Frequency Spectrum (Resolution: {freq_data[1]-freq_data[0]:.3f} Hz)')
    ax_fft.grid(True, which="both")
    
    if ch2_data:
        ax_fft.semilogx(ch2_data[0], ch2_data[1], label='Channel 2', alpha=0.7)
        ax_fft.legend()
    
    # Highlight expected frequency if provided
    if expected_freq:
        # Find closest bin
        idx = np.argmin(np.abs(np.array(freq_data) - expected_freq))
        ax_fft.axvline(x=freq_data[idx], color='r', linestyle='--', alpha=0.5)
        
        # Add text annotation for the peak
        ax_fft.annotate(f'{freq_data[idx]:.1f}Hz, {mag_data[idx]:.1f}dB',
                       xy=(freq_data[idx], mag_data[idx]),
                       xytext=(0, 20), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->'))
    
    # Zoomed FFT plot
    ax_zoom = fig.add_subplot(gs[2, 0])
    
    # Determine zoom range centered around expected frequency
    center_idx = np.argmax(mag_data) if expected_freq is None else \
                np.argmin(np.abs(np.array(freq_data) - expected_freq))
    
    # For high resolution FFTs, use a narrower zoom window
    if len(freq_data) > 10000:
        zoom_range = max(20, int(len(freq_data) / 1000))
    else:
        zoom_range = max(20, int(len(freq_data) / 100))
        
    start_idx = max(0, center_idx - zoom_range)
    end_idx = min(len(freq_data) - 1, center_idx + zoom_range)
    
    ax_zoom.plot(freq_data[start_idx:end_idx], mag_data[start_idx:end_idx], label='Channel 1')
    ax_zoom.set_xlabel('Frequency (Hz)')
    ax_zoom.set_ylabel('Magnitude (dB)')
    ax_zoom.set_title('Zoomed Spectrum Around Peak')
    ax_zoom.grid(True)
    
    if ch2_data:
        ax_zoom.plot(ch2_data[0][start_idx:end_idx], ch2_data[1][start_idx:end_idx], 
                   label='Channel 2', alpha=0.7)
        ax_zoom.legend()
    
    # Phase plot
    ax_phase = fig.add_subplot(gs[2, 1])
    ax_phase.plot(freq_data[start_idx:end_idx], phase_data[start_idx:end_idx], label='Channel 1')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.set_ylabel('Phase (degrees)')
    ax_phase.set_title('Phase Around Peak')
    ax_phase.grid(True)
    
    if ch2_data:
        ax_phase.plot(ch2_data[0][start_idx:end_idx], ch2_data[2][start_idx:end_idx], 
                    label='Channel 2', alpha=0.7)
        ax_phase.legend()
    
    # Find harmonics
    if expected_freq:
        fundamental_idx = center_idx
        fundamental_mag = mag_data[fundamental_idx]
        
        # Look for harmonics
        harmonics = []
        for n in range(2, 10):  # Look for 2nd through 9th harmonics
            harmonic_freq = expected_freq * n
            if harmonic_freq >= freq_data[-1]:
                break
            
            # Find closest bin
            h_idx = np.argmin(np.abs(np.array(freq_data) - harmonic_freq))
            h_mag = mag_data[h_idx]
            
            # Check if there's a local peak
            local_peak = True
            if h_idx > 0 and h_idx < len(mag_data) - 1:
                if mag_data[h_idx-1] > h_mag or mag_data[h_idx+1] > h_mag:
                    local_peak = False
            
            if local_peak and h_mag > -80:  # Only consider significant peaks
                thd_db = h_mag - fundamental_mag
                harmonics.append((n, freq_data[h_idx], h_mag, thd_db))
                
                # Mark on the plot
                ax_fft.axvline(x=freq_data[h_idx], color='g', linestyle=':', alpha=0.5)
                
        # Calculate THD
        if harmonics:
            harmonic_powers = [10**(h[2]/10) for h in harmonics]
            fundamental_power = 10**(fundamental_mag/10)
            thd_percent = 100 * np.sqrt(sum(harmonic_powers)) / np.sqrt(fundamental_power)
            
            # Add THD to plot
            harmonics_text = f"Fundamental: {expected_freq:.1f}Hz, {fundamental_mag:.1f}dB\n"
            harmonics_text += f"THD: {thd_percent:.2f}%\n"
            for h in harmonics:
                harmonics_text += f"{h[0]}x: {h[1]:.1f}Hz, {h[2]:.1f}dB ({h[3]:+.1f}dB)\n"
                
            ax_fft.text(0.02, 0.05, harmonics_text, transform=ax_fft.transAxes,
                      bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
    
    # Add title and adjust layout
    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Single Tone FFT Analyzer')
    parser.add_argument('--frequency', '-f', type=float, default=1000,
                        help='Probe frequency in Hz (default: 1000)')
    parser.add_argument('--amplitude', '-a', type=float, default=1.0,
                        help='Signal amplitude in Volts (default: 1.0)')
    parser.add_argument('--offset', '-o', type=float, default=0.0,
                        help='DC offset in Volts (default: 0.0)')
    parser.add_argument('--buffer-size', '-b', type=int, default=2**16,
                        help='Capture buffer size (default: 65536)')
    parser.add_argument('--sampling-rate', '-sr', type=float, default=1e6,
                        help='Base sampling rate in Hz (default: 1000000)')
    parser.add_argument('--no-coherent', action='store_true',
                        help='Disable coherent sampling')
    parser.add_argument('--window', '-w', type=str, default=None,
                        choices=['rectangular', 'flattop', 'blackman', 'cosine', 'hanning', 'hamming'],
                        help='Window type (default: auto-select)')
    parser.add_argument('--extended-buffer', '-e', action='store_true',
                        help='Enable extended buffer mode for very large buffer sizes')
    parser.add_argument('--save', '-s', type=str, default=None,
                        help='Save figure to file')
    
    args = parser.parse_args()
    
    # Map window type string to constant
    window_map = {
        'rectangular': None,  # Will be mapped to DwfWindowRectangular in the function
        'flattop': None,      # Will be mapped to DwfWindowFlatTop in the function
        'blackman': 3,        # Assuming these are the correct constants
        'cosine': 6,          # These values may need adjustment based on SDK
        'hanning': 1,
        'hamming': 2
    }
    window_type = window_map.get(args.window) if args.window else None
    
    # Print configuration
    print(f"Analyzing single tone:")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Amplitude: {args.amplitude} V")
    print(f"  Offset: {args.offset} V")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Base sampling rate: {args.sampling_rate} Hz")
    print(f"  Coherent sampling: {'No' if args.no_coherent else 'Yes'}")
    print(f"  Extended buffer: {'Yes' if args.extended_buffer else 'No'}")
    print(f"  Window type: {args.window or 'Auto'}")
    
    try:
        # Run the analysis
        result = analyze_tone(
            frequency=args.frequency,
            amplitude=args.amplitude,
            offset=args.offset,
            buffer_size=args.buffer_size,
            sampling_rate=args.sampling_rate,
            coherent_sampling=not args.no_coherent,
            window_type=window_type,
            extended_buffer=args.extended_buffer
        )
        
        # Unpack results
        freq_data, mag_data, phase_data, raw_data, ch2_available, ch2_data = result
        
        # Generate title
        title = f"Single Tone Analysis: {args.frequency} Hz, {args.amplitude} V"
        
        # Plot results
        fig = plot_results(
            freq_data=freq_data,
            mag_data=mag_data,
            phase_data=phase_data,
            raw_data=raw_data,
            ch2_data=ch2_data,
            expected_freq=args.frequency,
            title=title
        )
        
        # Save if requested
        if args.save:
            fig.savefig(args.save, dpi=300)
            print(f"Figure saved to {args.save}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 