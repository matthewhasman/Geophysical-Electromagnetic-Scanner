from ctypes import *
import math
import time
import numpy as np
from scipy import signal
import scipy.io
import control
from control.matlab import ss, bode
import sys
from WaveformInterface import AnalogDiscovery
import matplotlib.pyplot as plt

def numpy_fft_analyze(signal_data, sampling_rate, window_type=None):
    """
    Perform FFT analysis on a signal using NumPy's FFT implementation
    
    Parameters:
    - signal_data: time domain signal array
    - sampling_rate: rate at which signal was sampled in Hz
    - window_type: window function to apply (None, 'hann', 'hamming', 'blackman', 'flattop')
    
    Returns:
    - frequencies: array of frequency points
    - magnitudes: array of magnitude values in dB
    - phases: array of phase values in degrees
    """
    # Get signal length
    N = len(signal_data)
    
    # Apply window if specified
    if window_type is not None:
        if window_type == 'hann':
            window = np.hanning(N)
        elif window_type == 'hamming':
            window = np.hamming(N)
        elif window_type == 'blackman':
            window = np.blackman(N)
        elif window_type == 'flattop':
            window = signal.windows.flattop(N)
        else:
            # Default to rectangular window (no window)
            window = np.ones(N)
        
        # Apply window to signal
        windowed_data = signal_data * window
    else:
        # No window (rectangular)
        windowed_data = signal_data
    
    # Perform FFT
    fft_result = np.fft.rfft(windowed_data)
    
    # Calculate frequency points
    frequencies = np.fft.rfftfreq(N, d=1.0/sampling_rate)
    
    # Calculate magnitude spectrum (normalized, in dB)
    magnitudes = 20 * np.log10(np.abs(fft_result) / (N/2))
    
    # Calculate phase spectrum (in degrees)
    phases = np.angle(fft_result, deg=True)
    
    # Mask phases where magnitude is very low (to reduce noise in phase plot)
    phase_mask = magnitudes > (np.max(magnitudes) - 60)
    masked_phases = phases.copy()
    masked_phases[~phase_mask] = 0
    
    return frequencies, magnitudes, masked_phases

def custom_frequency_response(ad2, input_channel, output_channel, start_freq, stop_freq, 
                             steps, amplitude=0.5, window_type='hann'):
    """
    Measure frequency response between two channels using NumPy's FFT
    
    Parameters:
    - ad2: AnalogDiscovery object
    - input_channel: input signal channel (1-2)
    - output_channel: output signal channel (1-2)
    - start_freq: starting frequency in Hz
    - stop_freq: ending frequency in Hz
    - steps: number of frequency points
    - amplitude: signal amplitude in Volts
    - window_type: window function for FFT ('hann', 'hamming', 'blackman', 'flattop', None)
    
    Returns:
    - frequencies: array of frequency points
    - magnitude: array of magnitude values in dB
    - phase: array of phase values in degrees
    """
    print(f"Performing custom frequency response analysis using NumPy FFT")
    print(f"Frequency range: {start_freq} Hz to {stop_freq} Hz with {steps} steps")
    
    # Generate logarithmically spaced frequency points
    desired_frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), num=steps)
    
    # Storage for measurements
    actual_frequencies = []
    magnitude_response = []
    phase_response = []
    
    # Configure AWG on channel 1
    ad2.dwf.FDwfAnalogOutNodeEnableSet(ad2.handle, c_int(0), ad2.constants.AnalogOutNodeCarrier, c_int(1))
    ad2.dwf.FDwfAnalogOutNodeFunctionSet(ad2.handle, c_int(0), ad2.constants.AnalogOutNodeCarrier, ad2.constants.funcSine)
    ad2.dwf.FDwfAnalogOutNodeAmplitudeSet(ad2.handle, c_int(0), ad2.constants.AnalogOutNodeCarrier, c_double(amplitude))
    
    # Process each frequency point
    for i, freq in enumerate(desired_frequencies):
        try:
            print(f"Testing frequency {i+1}/{steps}: {freq:.2f} Hz")
            
            # Set generator frequency
            ad2.dwf.FDwfAnalogOutNodeFrequencySet(ad2.handle, c_int(0), 
                                                 ad2.constants.AnalogOutNodeCarrier, 
                                                 c_double(freq))
            ad2.dwf.FDwfAnalogOutConfigure(ad2.handle, c_int(0), c_int(1))
            
            # Wait for output to stabilize (min 10 cycles)
            stabilization_time = max(0.01, 10 / freq)
            print(f"  Waiting {stabilization_time:.3f}s for signal to stabilize...")
            time.sleep(stabilization_time)
            
            # Capture data from both channels
            raw_data1, _ = ad2.record(input_channel)
            raw_data2, _ = ad2.record(output_channel)
            
            # Perform FFT analysis
            freq1, mag1, phase1 = numpy_fft_analyze(
                signal_data=raw_data1, 
                sampling_rate=ad2.sampling_frequency,
                window_type=window_type
            )
            
            freq2, mag2, phase2 = numpy_fft_analyze(
                signal_data=raw_data2, 
                sampling_rate=ad2.sampling_frequency,
                window_type=window_type
            )
            
            # Find index closest to the test frequency
            idx1 = np.argmin(np.abs(freq1 - freq))
            idx2 = np.argmin(np.abs(freq2 - freq))
            
            # Get magnitude and phase at the test frequency
            in_mag = 10**(mag1[idx1]/20)  # Convert from dB to linear
            in_phase = phase1[idx1]
            out_mag = 10**(mag2[idx2]/20)  # Convert from dB to linear
            out_phase = phase2[idx2]
            
            # Compute transfer function
            if in_mag <= 0:
                mag_db = float('-inf')
                phase_diff = 0
            else:
                mag_db = 20 * np.log10(out_mag / in_mag)
                phase_diff = out_phase - in_phase
                phase_diff = (phase_diff + 180) % 360 - 180  # Wrap to [-180, 180]
            
            # Store results
            actual_frequencies.append(freq)
            magnitude_response.append(mag_db)
            phase_response.append(phase_diff)
            
            print(f"  Result: {mag_db:.2f} dB, {phase_diff:.2f} degrees")
            
        except Exception as e:
            print(f"  Error measuring at {freq} Hz: {str(e)}")
            # Skip this frequency point
    
    return np.array(actual_frequencies), np.array(magnitude_response), np.array(phase_response)

def setup_transfer_functions(self):
    # Load primary response
    try:
        path = "../matlab/primary_curve_fit_python.mat"
        mat = scipy.io.loadmat(path)
        num = mat['num'].tolist()[0][0][0]
        den = mat['den'].tolist()[0][0][0]
        self.sys = control.TransferFunction(num, den)
        
        print("Transfer functions loaded successfully")
    except Exception as e:
        print(f"Error loading transfer functions: {e}")

# Create a theoretical transfer function for comparison
sys = control.TransferFunction([1], [1])

# Configure hardware
sample_rate = 1000000.0  # 1 MHz
buffer_size = 2**13      # 8192 points (reduced for faster acquisition)

print("Initializing Analog Discovery hardware...")
Ad2 = AnalogDiscovery()
Ad2.init_scope(sample_rate, buffer_size)

# Frequency range to analyze
start_freq = 1_000    # 50 kHz
end_freq = 20_000     # 200 kHz
steps = 31             # Number of frequency points

print("Starting custom frequency response analysis...")
# Call custom NumPy-based frequency response function
frequencies, magnitude, phase = custom_frequency_response(
    Ad2, 
    input_channel=1, 
    output_channel=2, 
    start_freq=start_freq, 
    stop_freq=end_freq, 
    steps=steps, 
    amplitude=1.0, 
    window_type='hann'
)

print("Analysis complete. Creating plots...")
# Create a figure with two subplots for magnitude and phase
plt.figure(figsize=(10, 8))

# Plot magnitude response
plt.subplot(2, 1, 1)
plt.semilogx(frequencies, magnitude)
plt.grid(True, which="both")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response - Magnitude (NumPy FFT)')

# Set y-axis limits
if len(magnitude) > 0:
    y_min = min(magnitude) - 5 if min(magnitude) > -100 else -60
    plt.ylim(y_min, max(magnitude) + 5)

# Plot phase response
plt.subplot(2, 1, 2)
plt.semilogx(frequencies, phase)
plt.grid(True, which="both")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.title('Frequency Response - Phase (NumPy FFT)')
plt.ylim(-180, 180)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('numpy_frequency_response.png', dpi=300)
print(f"Frequency response plot saved as 'numpy_frequency_response.png'")

# Show the plot
plt.show()

# Clean up
print("Closing hardware connection...")
Ad2.close()
print("Done!")


        