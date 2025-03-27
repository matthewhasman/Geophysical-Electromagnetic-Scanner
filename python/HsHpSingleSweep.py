"""
   DWF Python Example
   Author:  Digilent, Inc.
   Revision:  2018-07-19

   Requires:                       
       Python 2.7, 3
"""

from ctypes import *
import time
from dwfconstants import *
import sys
import matplotlib.pyplot as plt
import numpy
import control
from control.matlab import ss, bode
import sys
import scipy

def setup_transfer_functions():
    # Load primary response
    path = "../matlab/primary_curve_fit_python.mat"
    mat = scipy.io.loadmat(path)
    num = mat['num'].tolist()[0][0][0]
    den = mat['den'].tolist()[0][0][0]
    return control.TransferFunction(num, den)

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

# continue running after device close, prevent temperature drifts
dwf.FDwfParamSet(c_int(4), c_int(0)) # 4 = DwfParamOnClose, 0 = continue 1 = stop 2 = shutdown

#print(DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: "+str(version.value))

#open device
hdwf = c_int()
print("Opening first device...")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    print("failed to open device")
    quit()
# Define a range of frequencies to sweep through
frequencies = numpy.exp(numpy.linspace(numpy.log(500), numpy.log(4e4), 30))  # Example: 30 frequencies from 500 Hz to 40 kHz on an exponential scale

# Initialize lists to store results
peak_magnitudes1 = []
peak_magnitudes2 = []
peak_phases1 = []
peak_phases2 = []

for frequency in frequencies:
    channel = c_int(0)

    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequency))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(1.0))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf, channel, AnalogOutNodeCarrier, c_double(0.0))

    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))

    hzRate = frequency * 4
    cSamples = 8 * 1024
    rgdSamples1 = (c_double * cSamples)()
    rgdSamples2 = (c_double * cSamples)()
    sts = c_int()

    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(2.0))

    print(f"Configure analog in for frequency {frequency} Hz")
    dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(20))
    dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(0.01))

    dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(cSamples))
    dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1)
    dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5 * cSamples / hzRate))

    print("Starting acquisition...")
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

    while True:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if sts.value == DwfStateArmed.value:
            break
        time.sleep(0.1)

    dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))
    time.sleep(0.3)

    while True:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if sts.value == DwfStateDone.value:
            break
        time.sleep(0.1)

    dwf.FDwfAnalogInStatusData(hdwf, c_int(0), rgdSamples1, len(rgdSamples1))
    dwf.FDwfAnalogInStatusData(hdwf, c_int(1), rgdSamples2, len(rgdSamples2))

    # Calculate FFT for both signals
    fft1 = numpy.fft.rfft(rgdSamples1)
    fft2 = numpy.fft.rfft(rgdSamples2)

    # Calculate magnitude in dB
    magnitude1 = 20 * numpy.log10(numpy.abs(fft1) / cSamples)
    magnitude2 = 20 * numpy.log10(numpy.abs(fft2) / cSamples)

    # Calculate frequency axis
    freq = numpy.fft.rfftfreq(cSamples, 1 / hzRate)

    # Calculate phase in degrees
    phase1 = numpy.angle(fft1, deg=True)
    phase2 = numpy.angle(fft2, deg=True)

    # Find the index of the target frequency
    target_idx = numpy.argmax(numpy.abs(fft1))

    # Get magnitude and phase at target frequency
    peak_magnitudes1.append(magnitude1[target_idx])
    peak_magnitudes2.append(magnitude2[target_idx])
    peak_phases1.append(phase1[target_idx])
    peak_phases2.append(phase2[target_idx])


sys = setup_transfer_functions()
w = 2 * numpy.pi * numpy.array(frequencies)
h_mag_prim, h_phase_prim, omega = bode(sys, w, plot=False)
h_mag_prim_lin = h_mag_prim
h_mag_prim_db = 20 * numpy.log10(h_mag_prim)
h_phase_prim_deg = numpy.degrees(h_phase_prim)

h_prim_complex = h_mag_prim_lin * numpy.exp(1j * numpy.deg2rad(h_phase_prim_deg))

unwrapped_phase_diff = numpy.unwrap(numpy.array(peak_phases2) - numpy.array(peak_phases1), 180)
h_mag_sec_lin = numpy.pow(10, (numpy.array(peak_magnitudes2) - numpy.array(peak_magnitudes1))/20)
h_secondary_complex = h_mag_sec_lin * numpy.exp(1j * numpy.deg2rad(unwrapped_phase_diff))

hs_hp = h_secondary_complex / h_prim_complex

# Plot the frequency sweep results
plt.figure(figsize=(12, 8))

# Plot magnitude response
plt.subplot(3, 1, 1)
plt.semilogx(frequencies / 1000, peak_magnitudes1, label='Channel 1')
plt.semilogx(frequencies / 1000, numpy.array(peak_magnitudes2) - numpy.array(peak_magnitudes1), label='Channel 2')
plt.semilogx(frequencies / 1000, h_mag_prim_db, label='Expected Primary Signal')
plt.title('Frequency Sweep - Magnitude Response')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)

# Plot phase response
plt.subplot(3, 1, 2)
plt.semilogx(frequencies / 1000, unwrapped_phase_diff, label='Channel 2')
plt.semilogx(frequencies / 1000, h_phase_prim_deg, label='Expected Primary Signal')
plt.title('Frequency Sweep - Phase Response')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Phase (degrees)')
plt.legend()
plt.grid(True)

# Plot hs/hp response
plt.subplot(3, 1, 3)
plt.semilogx(frequencies / 1000, numpy.real(hs_hp), label='Real')
plt.semilogx(frequencies / 1000, numpy.imag(hs_hp), label='Complex')
plt.title('Frequency Sweep - hs/hp Response')
plt.xlabel('Frequency (kHz)')
plt.ylabel('hs/hp')
plt.legend()
plt.grid(True)




plt.tight_layout()
plt.show()

