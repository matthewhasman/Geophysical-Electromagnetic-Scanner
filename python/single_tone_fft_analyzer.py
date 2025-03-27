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
from scipy.signal import find_peaks

def extract_and_average_cycles(signal, sampling_rate, frequency):
    """
    Extract individual cycles from a signal, align them, and compute the average cycle
    
    Parameters:
    - signal: time domain signal array
    - sampling_rate: rate at which signal was sampled in Hz
    - frequency: fundamental frequency of the signal in Hz
    
    Returns:
    - cycle_time: time points for a single cycle
    - average_cycle: average waveform of all extracted cycles
    - all_cycles: list of all extracted cycles
    - cycle_indices: indices where cycles begin
    """
    # Calculate expected samples per cycle
    samples_per_cycle = int(sampling_rate / frequency)
    
    if samples_per_cycle <= 3:
        raise ValueError(f"Frequency too high for reliable cycle extraction. Only {samples_per_cycle} samples per cycle.")
    
    # Try to find zero crossings or peaks for more reliable cycle detection
    try:
        # Find peaks for better cycle identification
        peaks, _ = find_peaks(signal, height=0.2*max(signal), distance=samples_per_cycle*0.8)
        
        if len(peaks) < 2:
            # If peaks detection fails, try zero crossings
            zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
            if len(zero_crossings) < 2:
                raise ValueError("Could not detect reliable cycles in signal")
            
            # Use rising zero crossings
            rising_zeros = [i for i in range(len(zero_crossings)-1) 
                         if signal[zero_crossings[i]+1] > signal[zero_crossings[i]]]
            cycle_indices = [zero_crossings[i] for i in rising_zeros]
        else:
            cycle_indices = peaks
    except Exception as e:
        print(f"Peak detection failed: {e}. Using simple cycle estimation.")
        # If both methods fail, just estimate cycles based on frequency
        num_cycles = int(len(signal) * frequency / sampling_rate)
        cycle_indices = [int(i * samples_per_cycle) for i in range(num_cycles)]
    
    # Ensure we have at least 2 complete cycles
    if len(cycle_indices) < 2:
        print("Warning: Not enough cycles detected for averaging")
        # Return empty results
        return np.array([]), np.array([]), [], []
    
    # Extract cycles
    all_cycles = []
    for i in range(len(cycle_indices) - 1):
        start_idx = cycle_indices[i]
        # Use the exact distance to the next cycle start as the cycle length
        cycle_length = cycle_indices[i+1] - start_idx
        
        # Ensure we don't go beyond the signal length
        if start_idx + cycle_length <= len(signal):
            cycle = signal[start_idx:start_idx + cycle_length]
            all_cycles.append(cycle)
    
    # Skip if no cycles were extracted
    if not all_cycles:
        return np.array([]), np.array([]), [], []
    
    # Prepare for averaging (resample cycles to the same length if needed)
    median_length = int(np.median([len(c) for c in all_cycles]))
    resampled_cycles = []
    
    for cycle in all_cycles:
        if len(cycle) != median_length:
            # Resample to median length
            indices = np.linspace(0, len(cycle)-1, median_length)
            resampled = np.interp(indices, np.arange(len(cycle)), cycle)
            resampled_cycles.append(resampled)
        else:
            resampled_cycles.append(cycle)
    
    # Compute average cycle
    average_cycle = np.mean(resampled_cycles, axis=0)
    
    # Generate time vector for a single cycle
    cycle_time = np.linspace(0, 1/frequency, len(average_cycle))
    
    return cycle_time, average_cycle, resampled_cycles, cycle_indices

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
               (freq_data2, mag_data2, phase_data2, raw_data2) if ch2_available else None, \
               actual_freq, sampling_rate
               
    finally:
        # Clean up
        print("Closing device...")
        device.close()

def plot_results(freq_data, mag_data, phase_data, raw_data, ch2_data=None, 
                expected_freq=None, title=None, actual_freq=None, sampling_rate=None):
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
    - actual_freq: actual frequency used for analysis
    - sampling_rate: sampling rate used for acquisition
    """
    # Create a figure with subplots - adding one more row for cycle average plot
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 2, figure=fig)
    
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
    
    # Add cycle average plot
    ax_cycle = fig.add_subplot(gs[1, :])
    
    # First check if we have all required information
    if actual_freq is not None and sampling_rate is not None:
        # Extract and average cycles for channel 1
        try:
            cycle_time1, avg_cycle1, all_cycles1, cycle_indices1 = extract_and_average_cycles(
                raw_data, sampling_rate, actual_freq)
            
            # If channel 2 is available, extract and average its cycles too
            if ch2_data and len(ch2_data) >= 4 and len(ch2_data[3]) > 0:
                cycle_time2, avg_cycle2, all_cycles2, cycle_indices2 = extract_and_average_cycles(
                    ch2_data[3], sampling_rate, actual_freq)
                ch2_cycles_available = len(avg_cycle2) > 0
            else:
                ch2_cycles_available = False
            
            # If channel 2 cycles are available, focus on those
            if ch2_cycles_available:
                # Calculate jitter metrics for channel 2
                cycle_period_samples = [cycle_indices2[i+1] - cycle_indices2[i] 
                                      for i in range(len(cycle_indices2)-1)]
                cycle_periods = [p / sampling_rate for p in cycle_period_samples]
                
                period_mean = np.mean(cycle_periods)
                period_std = np.std(cycle_periods)
                jitter_percent = 100 * period_std / period_mean if period_mean > 0 else 0
                
                # Plot channel 2 raw signal with cycle boundaries
                segment_length = min(5000, len(ch2_data[3]))  # Show a segment of the waveform
                segment_time = time_points[:segment_length] * 1000
                ax_cycle.plot(segment_time, ch2_data[3][:segment_length], 'purple', alpha=0.5, label='Ch2 Raw Signal')
                
                # Mark cycle boundaries on the raw signal
                for idx in cycle_indices2:
                    if idx < segment_length:
                        ax_cycle.axvline(x=time_points[idx] * 1000, color='r', linestyle='--', alpha=0.3)
                
                # Plot average cycle for channel 2
                ax_cycle_avg = ax_cycle.twinx()
                cycle_ms = cycle_time2 * 1000
                ax_cycle_avg.plot(cycle_ms, avg_cycle2, 'g-', linewidth=2, label='Ch2 Averaged Cycle')
                ax_cycle_avg.set_ylabel('Amplitude (V) - Averaged Cycle', color='g')
                
                # Add cycle statistics to the plot
                stats_text = (f"Channel 2 cycles: {len(all_cycles2)}\n"
                             f"Average period: {period_mean*1000:.3f} ms\n"
                             f"Period std dev: {period_std*1000:.3f} ms\n"
                             f"Jitter: {jitter_percent:.2f}%")
                
                ax_cycle.text(0.02, 0.95, stats_text, transform=ax_cycle.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8))
                
                # Set up legend and labels
                ax_cycle.set_xlabel('Time (ms)')
                ax_cycle.set_ylabel('Amplitude (V) - Raw Signal', color='purple')
                
                # Create a combined legend
                lines1, labels1 = ax_cycle.get_legend_handles_labels()
                lines2, labels2 = ax_cycle_avg.get_legend_handles_labels()
                ax_cycle.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
                
                ax_cycle.set_title('Channel 2 Signal Cycles: Raw and Averaged')
                ax_cycle.grid(True)
            elif len(avg_cycle1) > 0:
                # Fall back to channel 1 if channel 2 is not available or failed
                # Calculate jitter metrics for channel 1
                cycle_period_samples = [cycle_indices1[i+1] - cycle_indices1[i] 
                                      for i in range(len(cycle_indices1)-1)]
                cycle_periods = [p / sampling_rate for p in cycle_period_samples]
                
                period_mean = np.mean(cycle_periods)
                period_std = np.std(cycle_periods)
                jitter_percent = 100 * period_std / period_mean if period_mean > 0 else 0
                
                # Plot raw signal with cycle boundaries
                segment_length = min(5000, len(raw_data))  # Show a segment of the waveform
                segment_time = time_points[:segment_length] * 1000
                ax_cycle.plot(segment_time, raw_data[:segment_length], 'b-', alpha=0.5, label='Ch1 Raw Signal')
                
                # Mark cycle boundaries on the raw signal
                for idx in cycle_indices1:
                    if idx < segment_length:
                        ax_cycle.axvline(x=time_points[idx] * 1000, color='r', linestyle='--', alpha=0.3)
                
                # Plot average cycle
                ax_cycle_avg = ax_cycle.twinx()
                cycle_ms = cycle_time1 * 1000
                ax_cycle_avg.plot(cycle_ms, avg_cycle1, 'g-', linewidth=2, label='Ch1 Averaged Cycle')
                ax_cycle_avg.set_ylabel('Amplitude (V) - Averaged Cycle', color='g')
                
                # Add cycle statistics to the plot
                stats_text = (f"Channel 1 cycles: {len(all_cycles1)}\n"
                             f"Average period: {period_mean*1000:.3f} ms\n"
                             f"Period std dev: {period_std*1000:.3f} ms\n"
                             f"Jitter: {jitter_percent:.2f}%")
                
                ax_cycle.text(0.02, 0.95, stats_text, transform=ax_cycle.transAxes, 
                            bbox=dict(facecolor='white', alpha=0.8))
                
                # Set up legend and labels
                ax_cycle.set_xlabel('Time (ms)')
                ax_cycle.set_ylabel('Amplitude (V) - Raw Signal', color='b')
                
                # Create a combined legend
                lines1, labels1 = ax_cycle.get_legend_handles_labels()
                lines2, labels2 = ax_cycle_avg.get_legend_handles_labels()
                ax_cycle.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
                
                ax_cycle.set_title('Channel 1 Signal Cycles: Raw and Averaged')
                ax_cycle.grid(True)
            else:
                ax_cycle.text(0.5, 0.5, "Cycle averaging failed - not enough cycles detected", 
                             ha='center', va='center', transform=ax_cycle.transAxes)
                ax_cycle.set_title('Cycle Averaging (Failed)')
        
        except Exception as e:
            # If cycle extraction fails, display error message
            print(f"Cycle averaging error: {e}")
            ax_cycle.text(0.5, 0.5, f"Cycle averaging failed: {str(e)}", 
                         ha='center', va='center', transform=ax_cycle.transAxes)
            ax_cycle.set_title('Cycle Averaging (Failed)')
    else:
        ax_cycle.text(0.5, 0.5, "Cycle averaging requires frequency and sampling rate information", 
                     ha='center', va='center', transform=ax_cycle.transAxes)
        ax_cycle.set_title('Cycle Averaging (Not Available)')
    
    # Full spectrum FFT plot
    ax_fft = fig.add_subplot(gs[2, :])
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
    ax_zoom = fig.add_subplot(gs[3, 0])
    
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
    ax_phase = fig.add_subplot(gs[3, 1])
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
    parser.add_argument('--sampling-rate', '-sr', type=float, default=1e9,
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
        freq_data, mag_data, phase_data, raw_data, ch2_available, ch2_data, actual_freq, sampling_rate = result
        
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
            title=title,
            actual_freq=actual_freq,
            sampling_rate=sampling_rate
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