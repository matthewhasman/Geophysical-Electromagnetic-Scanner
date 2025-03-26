from ctypes import *
import math
import time
import numpy
import scipy
import scipy.io
import control
from control.matlab import ss, bode
import sys
import numpy as np
from WaveformInterface import AnalogDiscovery
import matplotlib.pyplot as plt

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

sys = control.TransferFunction([1], [1])

sample_rate = 1000000.0
buffer_size = 2**18

Ad2 = AnalogDiscovery()
Ad2.init_scope(sample_rate, buffer_size)

start_freq = 500
end_freq = 10_000

# Standard frequency response analysis (respects hardware limits)
frequencies, magnitude, phase = Ad2.frequency_response(1, 2, start_freq, end_freq, 51, 1.0, Ad2.constants.DwfWindowHann, extended_buffer=True, fixed_buffer_size=2**16)

# High-resolution analysis using extended buffer
# frequencies, magnitude, phase = Ad2.frequency_response(1, 2, start_freq, end_freq, 51, 1.0, Ad2.constants.DwfWindowHann, extended_buffer=True, fixed_buffer_size=2**20)


# Create a figure with two subplots for magnitude and phase
plt.figure(figsize=(10, 8))

# Plot magnitude response
plt.subplot(2, 1, 1)
plt.semilogx(frequencies, magnitude)
plt.grid(True, which="both")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Frequency Response - Magnitude')
# Set y-axis to show 0 dB at the top
y_min = min(magnitude) - 5  # Add some padding below the minimum value
plt.ylim(y_min, 0)

# Plot phase response
plt.subplot(2, 1, 2)
plt.semilogx(frequencies, phase)
plt.grid(True, which="both")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.title('Frequency Response - Phase')
# Set y-axis range to ±180 degrees
plt.ylim(-180, 180)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('frequency_response.png', dpi=300)
print(f"Frequency response plot saved as 'frequency_response.png'")

# Show the plot
plt.show()

Ad2.close()


        