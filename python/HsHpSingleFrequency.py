from ctypes import *
import time
from dwfconstants import *
import sys
import matplotlib.pyplot as plt
import numpy as np
import control
from control.matlab import ss, bode
import sys
import scipy
import pyqtgraph as pg
import csv
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout


# Default frequency and frequency limits
default_frequency = 1e3
min_frequency = 3e2  # 300 Hz 
max_frequency = 7e4  # 70 kHz

# Global frequency variable
frequency = default_frequency

def get_primary(freq):
    # Load primary response
    path = "../matlab/primary_curve_fit_python.mat"
    mat = scipy.io.loadmat(path)
    num = mat['num'].tolist()[0][0][0]
    den = mat['den'].tolist()[0][0][0]
    transfer_function = control.TransferFunction(num, den)
    w = 2 * np.pi * np.array(freq)
    h_mag_prim, h_phase_prim, omega = bode(transfer_function, [w], plot=False)
    h_mag_prim_lin = h_mag_prim
    h_phase_prim_deg = np.degrees(h_phase_prim)
    h_prim_complex = h_mag_prim_lin * np.exp(1j * np.deg2rad(h_phase_prim_deg))
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

# Get primary response for initial frequency
h_prim_complex = get_primary(frequency)

# Setup initial analog out configuration
def configure_analog_out():
    global frequency, hdwf, channel, h_prim_complex
    # Update primary response for the new frequency
    h_prim_complex = get_primary(frequency)
    
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequency))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(2.0))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf, channel, AnalogOutNodeCarrier, c_double(0.0))
    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(2.0))

# Initial analog out configuration
configure_analog_out()


cSamples = 8 * 1024
FIXED_SAMPLE_RATE = 2e5
#hzRate = frequency * 10
hzRate = FIXED_SAMPLE_RATE
rgdSamples1 = (c_double * cSamples)()
rgdSamples2 = (c_double * cSamples)()
sts = c_int()

print(f"Configure analog in for frequency {frequency} Hz")
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(20))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(0.01))

dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(cSamples))
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1)
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5 * cSamples / hzRate))

channel1Peaks = np.array([])
channel2Peaks = np.array([])
relativePhases = np.array([])
hshpReals = np.array([])
hshpImag = np.array([])
timestamps = np.array([])
apparent_conductivity = np.array([])
# Initialize PyQt application and plots
app = pg.mkQApp("Real-time Plotting")
win = pg.GraphicsLayoutWidget(show=True, title="Real-time Data")
win.resize(1200, 800)  # Larger window size
win.setWindowTitle('Real-time Data Visualization')

# Set default theme for better appearance
pg.setConfigOption('background', (50, 50, 50))
pg.setConfigOption('foreground', 'w')

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
hshp_plot.addLegend()
hshp_curve_real = hshp_plot.plot(pen='r', name="Real")
hshp_curve_imag = hshp_plot.plot(pen='b', name="Imag")

# time domain plot 
win.nextRow()
apparent_conductivity_plot = win.addPlot(title="Apparent Conductivity (MS/m)")
apparent_cond_curve = apparent_conductivity_plot.plot(pen='r', name="conductivity")

# Helper function to create styled frames
def create_styled_frame():
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setFrameShadow(QFrame.Raised)
    frame.setLineWidth(2)
    
    # Set background color to match the plot background but slightly lighter
    palette = frame.palette()
    palette.setColor(QPalette.Window, QColor(60, 60, 60))
    frame.setPalette(palette)
    frame.setAutoFillBackground(True)
    
    return frame

# Create the frequency control widget
freq_control_frame = create_styled_frame()
freq_layout = QVBoxLayout(freq_control_frame)

# Create a header for the frequency section
header_label = QLabel("Frequency Control")
header_label.setFont(QFont("Arial", 12, QFont.Bold))
header_label.setStyleSheet("color: #FFFFFF;")
header_label.setAlignment(Qt.AlignCenter)
freq_layout.addWidget(header_label)

# Create slider container with value display
slider_container = QHBoxLayout()

# Create label for frequency display with better styling
freq_label = QLabel(f"Current: {frequency:.1f} Hz")
freq_label.setFont(QFont("Arial", 10))
freq_label.setStyleSheet("color: #FFFFFF; background-color: rgba(0, 0, 0, 40); padding: 3px; border-radius: 3px;")
freq_label.setMinimumWidth(150)
slider_container.addWidget(freq_label)

# Create slider for frequency control with better styling
freq_slider = pg.QtWidgets.QSlider(Qt.Horizontal)
freq_slider.setMinimum(0)
freq_slider.setMaximum(1000)

# Calculate initial slider position
slider_value = 1000 * (np.log10(default_frequency) - np.log10(min_frequency)) / (np.log10(max_frequency) - np.log10(min_frequency))
freq_slider.setValue(int(slider_value))

freq_slider.setTickInterval(100)
freq_slider.setTickPosition(pg.QtWidgets.QSlider.TicksBelow)
freq_slider.setStyleSheet("""
    QSlider::groove:horizontal {
        border: 1px solid #999999;
        height: 8px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #333333, stop:1 #999999);
        margin: 2px 0;
        border-radius: 4px;
    }
    QSlider::handle:horizontal {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #5DADE2, stop:1 #2980B9);
        border: 1px solid #5DADE2;
        width: 18px;
        margin: -5px 0;
        border-radius: 9px;
    }
    QSlider::handle:horizontal:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #3498DB, stop:1 #1ABC9C);
    }
""")
slider_container.addWidget(freq_slider)
freq_layout.addLayout(slider_container)

# Add min/max labels below the slider
range_container = QHBoxLayout()

min_label = QLabel(f"{min_frequency:.0f} Hz")
min_label.setStyleSheet("color: #AAAAAA;")
range_container.addWidget(min_label)

range_container.addStretch()

max_label = QLabel(f"{max_frequency:.0f} Hz")
max_label.setStyleSheet("color: #AAAAAA;")
range_container.addWidget(max_label)

freq_layout.addLayout(range_container)

# Create the button container
button_frame = create_styled_frame()
button_layout = QVBoxLayout(button_frame)

# Create a header for the buttons section
button_header = QLabel("Data Controls")
button_header.setFont(QFont("Arial", 12, QFont.Bold))
button_header.setStyleSheet("color: #FFFFFF;")
button_header.setAlignment(Qt.AlignCenter)
button_layout.addWidget(button_header)

# Button container
btn_container = QHBoxLayout()

# Clear button with styling
clear_button = pg.QtWidgets.QPushButton("Clear History")
clear_button.clicked.connect(lambda: clear_data())
clear_button.setMinimumHeight(30)
clear_button.setStyleSheet("""
    QPushButton {
        background-color: #E74C3C;
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #C0392B;
    }
    QPushButton:pressed {
        background-color: #A93226;
    }
""")
btn_container.addWidget(clear_button)

# Export CSV button with styling
export_button = pg.QtWidgets.QPushButton("Export CSV")
export_button.clicked.connect(lambda: export_data_csv())
export_button.setMinimumHeight(30)
export_button.setStyleSheet("""
    QPushButton {
        background-color: #27AE60;
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #229954;
    }
    QPushButton:pressed {
        background-color: #1E8449;
    }
""")
btn_container.addWidget(export_button)

button_layout.addLayout(btn_container)

# Update the frequency function
def update_frequency(value):
    global frequency, hdwf, channel, hzRate, h_prim_complex
    # Convert slider value to frequency using logarithmic scale
    frequency = 10 ** (value / 1000.0 * (np.log10(max_frequency) - np.log10(min_frequency)) + np.log10(min_frequency))
    
    # Update primary response for the new frequency
    h_prim_complex = get_primary(frequency)
    
    # Update frequency in device
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequency))
    
    # Update sample rate to be 10x the frequency
    #hzRate = frequency * 10
    hzRate = FIXED_SAMPLE_RATE
    dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
    dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5 * cSamples / hzRate))

    # Update label with nice formatting
    freq_label.setText(f"Current: {frequency:.1f} Hz")
    
    print(f"Frequency updated to {frequency:.1f} Hz, Sample rate: {hzRate:.1f} Hz")

# Connect the slider to the update function
freq_slider.valueChanged.connect(update_frequency)

# Create a dedicated control panel row at the bottom of the window
win.nextRow()

# Create proxies for both widgets
freq_proxy = pg.QtWidgets.QGraphicsProxyWidget()
freq_proxy.setWidget(freq_control_frame)

button_proxy = pg.QtWidgets.QGraphicsProxyWidget()
button_proxy.setWidget(button_frame)

# Add the widgets in a horizontal layout
control_layout = win.addLayout(row=6, col=0)
control_layout.addItem(freq_proxy, row=0, col=0)
control_layout.addItem(button_proxy, row=0, col=1)

# Clear data function
def clear_data():
    global timestamps, relativePhases, peak_magnitudes1, peak_magnitudes2, hshpReals, hshpImag, apparent_conductivity
    timestamps = np.array([])
    relativePhases = np.array([])
    peak_magnitudes1 = np.array([])
    peak_magnitudes2 = np.array([])
    hshpReals = np.array([])
    hshpImag = np.array([])
    apparent_conductivity = np.array([])
    print("History cleared.")

def export_data_csv():
    """Export the current data to a CSV file"""
    try:
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/peak_magnitude_data_{timestamp_str}.csv"
        
        # Calculate magnitude difference
        magnitude_difference = peak_magnitudes2 - peak_magnitudes1
        
        # Open CSV file for writing
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header
            csvwriter.writerow(['Timestamp (s)', 'Frequency (Hz)', 'Magnitude 1 (dB)', 'Magnitude 2 (dB)', 
                               'Magnitude Difference (dB)', 'Phase Difference (deg)', 
                               'HsHp Real', 'HsHp Imag'])
            
            # Write data
            for i in range(len(timestamps)):
                csvwriter.writerow([
                    timestamps[i],
                    frequency,  # Include current frequency
                    peak_magnitudes1[i] if i < len(peak_magnitudes1) else '',
                    peak_magnitudes2[i] if i < len(peak_magnitudes2) else '',
                    magnitude_difference[i] if i < len(magnitude_difference) else '',
                    relativePhases[i] if i < len(relativePhases) else '',
                    hshpReals[i] if i < len(hshpReals) else '',
                    hshpImag[i] if i < len(hshpImag) else ''
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
    fft1 = np.fft.rfft(rgdSamples1)
    fft2 = np.fft.rfft(rgdSamples2)

    # Calculate magnitude in dB
    magnitude1 = 20 * np.log10(np.abs(fft1) / cSamples)
    magnitude2 = 20 * np.log10(np.abs(fft2) / cSamples)

    # Calculate frequency axis
    freq = np.fft.rfftfreq(cSamples, 1 / hzRate)

    # Calculate phase in degrees
    phase1 = np.angle(fft1, deg=True)
    phase2 = np.angle(fft2, deg=True)

    # Find the index of the target frequency
    target_idx = np.argmax(magnitude1[10:-10])+10 
    peak_phase1 = phase1[target_idx]
    peak_phase2 = phase2[target_idx]

    # Get magnitude and phase at target frequency
    peak_magnitudes1 = np.append(peak_magnitudes1, magnitude1[target_idx])
    peak_magnitudes2 = np.append(peak_magnitudes2, magnitude2[target_idx])

    phase_diff = peak_phase2 - peak_phase1
    relativePhases = np.append(relativePhases, phase_diff)

    h_mag_sec_lin = np.pow(10, (magnitude2[target_idx] - magnitude1[target_idx])/20)
    h_secondary_complex = h_mag_sec_lin * np.exp(1j * np.deg2rad(phase_diff))
    hshp = h_secondary_complex / h_prim_complex
    hshpReals = np.append(hshpReals, np.real(hshp))
    hshpImag = np.append(hshpImag, np.imag(hshp))

    timestamps = np.append(timestamps, time.time() - start_time)

    mu0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * frequency
    r = 0.9
    apparent_conductivity =  hshpImag * 4 / (mu0 * omega * r**2)

    # Update plots
    magnitude_curve1.setData(freq, magnitude1)
    magnitude_curve2.setData(freq, magnitude2)
    phase_curve.setData(timestamps, np.unwrap(relativePhases, 180))
    mag_time_curve.setData(timestamps, peak_magnitudes2 - peak_magnitudes1)
    apparent_cond_curve.setData(timestamps, apparent_conductivity / 1e6)
    hshp_curve_real.setData(timestamps, hshpReals)
    hshp_curve_imag.setData(timestamps, hshpImag)
    
    # Process events to update the plots
    app.processEvents()