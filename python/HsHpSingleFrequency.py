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
import csv
import os
from datetime import datetime

def get_primary(freq):
    # Load primary response
    path = "../matlab/primary_curve_fit_python.mat"
    mat = scipy.io.loadmat(path)
    num = mat['num'].tolist()[0][0][0]
    den = mat['den'].tolist()[0][0][0]
    transfer_function =  control.TransferFunction(num, den)
    w = 2 * numpy.pi * numpy.array(freq)
    h_mag_prim, h_phase_prim, omega = bode(transfer_function, [w], plot=False)
    h_mag_prim_lin = h_mag_prim
    h_phase_prim_deg = numpy.degrees(h_phase_prim)
    h_prim_complex = h_mag_prim_lin * numpy.exp(1j * numpy.deg2rad(h_phase_prim_deg))
    return h_prim_complex


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

frequency = 1e3
h_prim_complex = get_primary(frequency)
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
hshpReals = numpy.array([])
hshpImag = numpy.array([])
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

# hs hp plot 
win.nextRow()
hshp_plot = win.addPlot(title="HsHp PLot")
magnitude_plot.addLegend()
hshp_curve_real = hshp_plot.plot(pen='r', name="Real")
hshp_curve_imag = hshp_plot.plot(pen='b', name="Imag")


# time domain plot 
win.nextRow()
time_domain_plot = win.addPlot(title="Time Domain Signal")
time_domain_curve = time_domain_plot.plot(pen='m', name="Channel 2")

# Create a QWidget to hold the buttons
button_widget = pg.QtWidgets.QWidget()
button_layout = pg.QtWidgets.QHBoxLayout()  # Changed to horizontal layout for buttons side by side

# Clear button
clear_button = pg.QtWidgets.QPushButton("Clear History")
clear_button.clicked.connect(lambda: clear_data())
button_layout.addWidget(clear_button)

# Export CSV button
export_button = pg.QtWidgets.QPushButton("Export CSV")
export_button.clicked.connect(lambda: export_data_csv())
button_layout.addWidget(export_button)

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

def export_data_csv():
    """Export the current data to a CSV file"""
    try:
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/peak_magnitude_data_{timestamp_str}.csv"
        
        # Calculate magnitude difference
        magnitude_difference = peak_magnitudes2 - peak_magnitudes1
        
        # Open CSV file for writing
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            csvwriter.writerow(['Timestamp (s)', 'Magnitude 1 (dB)', 'Magnitude 2 (dB)', 
                               'Magnitude Difference (dB)', 'Phase Difference (deg)'])
            
            # Write data
            for i in range(len(timestamps)):
                csvwriter.writerow([
                    timestamps[i],
                    peak_magnitudes1[i] if i < len(peak_magnitudes1) else '',
                    peak_magnitudes2[i] if i < len(peak_magnitudes2) else '',
                    magnitude_difference[i] if i < len(magnitude_difference) else '',
                    relativePhases[i] if i < len(relativePhases) else ''
                ])
        
        print(f"Data exported to {filename}")
        
        # Show a message box to inform the user
        msg_box = pg.QtWidgets.QMessageBox()
        msg_box.setWindowTitle("Export Successful")
        msg_box.setText(f"Data successfully exported to:\n{os.path.abspath(filename)}")
        msg_box.exec_()
        
    except Exception as e:
        print(f"Error exporting data: {e}")
        # Show error message
        error_box = pg.QtWidgets.QMessageBox()
        error_box.setWindowTitle("Export Error")
        error_box.setText(f"Error exporting data: {e}")
        error_box.exec_()

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

    h_mag_sec_lin = numpy.pow(10, (magnitude2[target_idx] - magnitude1[target_idx])/20)
    h_secondary_complex = h_mag_sec_lin * numpy.exp(1j * numpy.deg2rad(phase_diff))
    hshp = h_secondary_complex / h_prim_complex
    hshpReals = numpy.append(hshpReals, numpy.real(hshp))
    hshpImag = numpy.append(hshpImag, numpy.imag(hshp))

    timestamps = numpy.append(timestamps, time.time() - start_time)

    # Update plots
    magnitude_curve1.setData(freq, magnitude1)
    magnitude_curve2.setData(freq, magnitude2)
    phase_curve.setData(timestamps, numpy.unwrap(relativePhases, 180))
    mag_time_curve.setData(timestamps, peak_magnitudes2 - peak_magnitudes1)
    time_domain_curve.setData(numpy.array(rgdSamples2[0:100]))
    hshp_curve_real.setData(hshpReals)
    hshp_curve_imag.setData(hshpImag)
    
    # Process events to update the plots
    app.processEvents()