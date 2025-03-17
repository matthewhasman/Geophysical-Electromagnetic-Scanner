from ctypes import *
from dwfconstants import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy
import control
import scipy
from control.matlab import ss, bode

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: " + str(version.value))

hdwf = c_int()
szerr = create_string_buffer(512)
print("Opening first device")
if dwf.FDwfDeviceOpen(-1, byref(hdwf)) != 1 or hdwf.value == hdwfNone.value:
    dwf.FDwfGetLastErrorMsg(szerr)
    print(szerr.value)
    print("Failed to open device")
    quit()

# Prevent temperature drift
dwf.FDwfParamSet(DwfParamOnClose, c_int(0))
dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))  # Manual configuration

# Configure AnalogOut for channel 0 (AWG)
dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(0.5))  # 1 Vpp
dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))  # Initial configuration

# Configure AnalogIn for channels 0 and 1
nSamples = 2**10  # Power of two for FFT efficiency
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(20000000.0))  # 20 MHz sample rate
dwf.FDwfAnalogInBufferSizeSet(hdwf, nSamples)
dwf.FDwfAnalogInChannelEnableSet(hdwf, 0, c_int(1))  # Enable channel 0 (C1)
dwf.FDwfAnalogInChannelRangeSet(hdwf, 0, c_double(20))  # 20V range
dwf.FDwfAnalogInChannelEnableSet(hdwf, 1, c_int(1))  # Enable channel 1 (C2)
dwf.FDwfAnalogInChannelRangeSet(hdwf, 1, c_double(2))  # 2V range

# Generate logarithmic frequency sweep
start_freq = 150.0  # Hz
stop_freq = 1e5     # Hz
steps = 101
frequencies = numpy.logspace(numpy.log10(start_freq), numpy.log10(stop_freq), num=steps)

# Prepare window for FFT
rgdWindow = (c_double * nSamples)()
vBeta = c_double(1.0)
vNEBW = c_double()
dwf.FDwfSpectrumWindow(byref(rgdWindow), c_int(nSamples), DwfWindowFlatTop, vBeta, byref(vNEBW))

h_mag = []
h_mag_lin = []
h_phase = []

##time.sleep(2)  # Wait for settings to stabilize
for freq in frequencies:
    print(f"Frequency: {freq/1e3:.1f} kHz")
    # Set AnalogOut frequency
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(freq))
    dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))
    
    # Start AnalogIn acquisition
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))
    
    # Wait for acquisition
    while True:
        sts = c_byte()
        if dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts)) != 1:
            dwf.FDwfGetLastErrorMsg(szerr)
            print(szerr.value)
            quit()
        if sts.value == DwfStateDone.value:
            break
        time.sleep(0.001)
    
    # Retrieve data
    rgdSamples1 = (c_double * nSamples)()
    rgdSamples2 = (c_double * nSamples)()
    dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples1, nSamples)
    dwf.FDwfAnalogInStatusData(hdwf, 1, rgdSamples2, nSamples)
    
    # Apply window and FFT for both channels
    def process_data(samples, frequency=freq):
        for i in range(nSamples):
            samples[i] = samples[i] * rgdWindow[i]
        nBins = nSamples // 2 + 1
        rgdBins = (c_double * nBins)()
        rgdPhase = (c_double * nBins)()
        dwf.FDwfSpectrumFFT(byref(samples), nSamples, byref(rgdBins), byref(rgdPhase), nBins)
        ## Get frequency index
        fIndex = int(freq / 20000000 * nSamples) + 1
        ##iPeak = numpy.argmax(rgdBins[1:nBins]) + 1 # Skip DC
        return rgdBins[fIndex], rgdPhase[fIndex]
    
    c1_mag, c1_phase = process_data(rgdSamples1)
    c2_mag, c2_phase = process_data(rgdSamples2)
    
    # Compute transfer function
    if c1_mag <= 0:
        h_db = -float('inf')
    else:
        c1_mag *= 10 ## Assuming 10x probe attenuation
        h_db = 20 * math.log10(c2_mag / c1_mag)
        h_linear = c2_mag / c1_mag
        ##h_db = 20 * math.log10(c2_mag / 20)

    
    phase_diff = (c2_phase - c1_phase) * 180 / math.pi
    phase_diff = (phase_diff + 180) % 360 - 180  # Wrap to [-180, 180]
    
    h_mag.append(h_db) 
    h_mag_lin.append(h_linear)
    h_phase.append(phase_diff)

dwf.FDwfDeviceCloseAll()
## Throw out first data point, always bad
frequencies = frequencies[1:]
h_mag = h_mag[1:]
h_mag_lin = h_mag_lin[1:]
h_phase = h_phase[1:]


## Get the transfer function for the primary response from .mat file 
## This is the transfer function of the primary response
path = "../matlab/primary_curve_fit_python.mat"
mat = scipy.io.loadmat(path)
num = mat['num'].tolist()[0][0][0]
den = mat['den'].tolist()[0][0][0]

sys = control.TransferFunction(num, den)
## Get primary response for frequencies
w = 2 * numpy.pi * frequencies
h_mag_prim, h_phase_prim, omega = bode(sys, w, plot=False)
## Convert to dB
h_mag_prim_lin = h_mag_prim
h_mag_prim = 20 * numpy.log10(h_mag_prim)
h_phase_prim = numpy.degrees(h_phase_prim)


## Convert values to complex numbers for plotting hs/hp
h_mag_lin = numpy.array(h_mag_lin)
h_phase = numpy.array(h_phase)

h_prim_complex = h_mag_prim_lin * numpy.exp(1j * numpy.radians(h_phase_prim))
h_complex = h_mag_lin * numpy.exp(1j * numpy.radians(h_phase))

hs_hp = h_complex / h_prim_complex

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.subplot(3, 1, 1)
# plt.semilogx(frequencies, h_mag)
# plt.semilogx(frequencies, h_mag_prim, '--')
# plt.ylabel('Magnitude (dB)')
# plt.grid(True, which="both", ls="--")
# plt.title('Transfer Function')

# plt.subplot(3, 1, 2)
# plt.semilogx(frequencies, h_phase)
# plt.semilogx(frequencies, h_phase_prim, '--')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Phase (degrees)')
# plt.grid(True, which="both", ls="--")

## plot hs/hp with both real and imaginary parts
# plt.subplot(3, 1, 3)
plt.figure(figsize=(10, 6))
plt.semilogx(frequencies, numpy.real(hs_hp))
plt.semilogx(frequencies, numpy.imag(hs_hp))
plt.ylabel('hs/hp')
plt.xlabel('Frequency (Hz)')
plt.grid(True, which="both", ls="--")



plt.tight_layout()
plt.show()