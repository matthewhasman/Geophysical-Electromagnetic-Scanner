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
import pyqtgraph as pg

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

# Initialize lists to store results
peak_magnitudes1 = []
peak_magnitudes2 = []
peak_phases1 = []
peak_phases2 = []

channel = c_int(0)

frequency = 2e4
dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcSine)
dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequency))
dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(1.0))
dwf.FDwfAnalogOutNodeOffsetSet(hdwf, channel, AnalogOutNodeCarrier, c_double(0.0))
dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))

hzRate = frequency * 10
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

channel1Peaks = numpy.array([])
channel2Peaks = numpy.array([])
relativePhases = numpy.array([])
timestamps = numpy.array([])

# Initialize PyQt application and plots
app = pg.mkQApp("Real-time Plotting")
win = pg.GraphicsLayoutWidget(show=True, title="Real-time Data")
win.resize(1000, 600)
win.setWindowTitle('Real-time Data')

# Magnitude plot
magnitude_plot = win.addPlot(title="Magnitude")
magnitude_plot.addLegend()
magnitude_curve1 = magnitude_plot.plot(pen='r', name="Magnitude 1")
magnitude_curve2 = magnitude_plot.plot(pen='b', name="Magnitude 2")

# Phase plot
win.nextRow()
phase_plot = win.addPlot(title="Phase Difference")
phase_curve = phase_plot.plot(pen='g', name="Phase Difference")

# mag time plot
win.nextRow()
mag_time_plot = win.addPlot(title="Magnitude over time")
mag_time_curve = mag_time_plot.plot(pen='m', name="Running Secondary Magnitude")
start_time = time.time()

# time domain plot 
win.nextRow()
time_domain_plot = win.addPlot(title="Time Domain Signal")
time_domain_curve = time_domain_plot.plot(pen='m', name="Channel 2")



# Create a QWidget to hold the button
button_widget = pg.QtWidgets.QWidget()
button_layout = pg.QtWidgets.QVBoxLayout()
clear_button = pg.QtWidgets.QPushButton("Clear History")
clear_button.clicked.connect(lambda: clear_data())
button_layout.addWidget(clear_button)
button_widget.setLayout(button_layout)

# Add the button widget to the main PyQt window
win.nextRow()
win.addItem(pg.ViewBox())
win.scene().addWidget(button_widget)

def clear_data():
    global timestamps, relativePhases, peak_magnitudes1, peak_magnitudes2
    timestamps = numpy.array([])
    relativePhases = numpy.array([])
    peak_magnitudes1 = []
    peak_magnitudes2 = []
    print("History cleared.")


while True:
    
    # Configure analog input
    dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

    ## Turn on the output 
    dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))
    #time.sleep(0.3)

    ## Wait for data to finish 
    while True:
        dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
        if sts.value == DwfStateDone.value:
            break
        time.sleep(0.01)

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
    target_idx = numpy.argmax(magnitude1[10:-10])+10 
    peak_phase1 = phase1[target_idx]
    peak_phase2 = phase2[target_idx]

    # Get magnitude and phase at target frequency
    peak_magnitudes1 = numpy.append(peak_magnitudes1, magnitude1[target_idx])
    peak_magnitudes2 = numpy.append(peak_magnitudes2, magnitude2[target_idx])

    phase_diff = peak_phase2 - peak_phase1
    relativePhases = numpy.append(relativePhases, phase_diff)

    h_mag_sec_lin = numpy.pow(10, (numpy.array(peak_magnitudes2) - numpy.array(peak_magnitudes1))/20)
    h_secondary_complex = h_mag_sec_lin * numpy.exp(1j * numpy.deg2rad(phase_diff))

    timestamps = numpy.append(timestamps, time.time() - start_time)

    # Update plots
    magnitude_curve1.setData(freq, magnitude1)
    magnitude_curve2.setData(freq, magnitude2)
    phase_curve.setData(timestamps, numpy.unwrap(relativePhases, 180))
    mag_time_curve.setData(timestamps, peak_magnitudes2 - peak_magnitudes1)
    time_domain_curve.setData(numpy.array(rgdSamples2[0:100]))
    # Process events to update the plots
    app.processEvents()

