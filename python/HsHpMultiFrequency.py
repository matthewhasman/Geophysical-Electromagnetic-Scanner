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
import torch 
from training import FDEM1DInversionNet
import pickle

# Default frequency and frequency limits
min_frequency = 3e2  # 300 Hz 
max_frequency = 3e4  # 30 kHz

# Global frequency variable
frequencies = np.linspace(min_frequency, max_frequency, 8)

def get_primary(freq):
    # Load primary response
    path = "../matlab/primary_curve_fit_python.mat"
    mat = scipy.io.loadmat(path)
    num = mat['num'].tolist()[0][0][0]
    den = mat['den'].tolist()[0][0][0]
    transfer_function = control.TransferFunction(num, den)
    w = 2 * np.pi * freq
    h_mag_prim, h_phase_prim, omega = bode(transfer_function, w, plot=False)
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

channel = c_int(0)

# Get primary response for initial frequency
h_prim_complex = get_primary(frequencies)

# Setup initial analog out configuration
def configure_analog_out():
    global hdwf, channel
    
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequencies[0]))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(2.0))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf, channel, AnalogOutNodeCarrier, c_double(0.0))
    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(2.0))

def set_out_freq(frequency):
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(frequency))


# Initial analog out configuration
configure_analog_out()


cSamples = 8 * 1024
FIXED_SAMPLE_RATE = 2e5
hzRate = FIXED_SAMPLE_RATE
rgdSamples1 = (c_double * cSamples)()
rgdSamples2 = (c_double * cSamples)()
sts = c_int()

## Initialize NN

# Load the trained model - add this after initializing plots and before the main loop
print("Loading trained FDEM inversion model...")
try:
    checkpoint = torch.load('models/fdem_1d_model.pth')
    with open('models/fdem_1d_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Initialize the model
    trained_model = FDEM1DInversionNet(num_freqs=len(frequencies))
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    trained_model.eval()  # Set to evaluation mode

    
    print("Model loaded successfully!")
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    print("Continuing without neural network inference...")
    model_loaded = False

# Initialize lists to store results - no initial dummy data
peak_magnitudes = np.array([])
relativePhases = np.array([])
hshpReals = np.array([])
hshpImag = np.array([])
timestamps = np.array([])
data_count = 0  # Counter for number of data points collected

print(f"Configure analog in for frequency {frequencies[0]} Hz")
dwf.FDwfAnalogInFrequencySet(hdwf, c_double(hzRate))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(0), c_double(20))
dwf.FDwfAnalogInChannelRangeSet(hdwf, c_int(1), c_double(0.01))

dwf.FDwfAnalogInBufferSizeSet(hdwf, c_int(cSamples))
dwf.FDwfAnalogInTriggerSourceSet(hdwf, trigsrcAnalogOut1)
dwf.FDwfAnalogInTriggerPositionSet(hdwf, c_double(0.5 * cSamples / hzRate))

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
magnitude_plot.setLabel('left', 'Magnitude', units='dB')
magnitude_plot.setLabel('bottom', 'Frequency', units='Hz')

# Phase plot - converted to heatmap
win.nextRow()
phase_plot = win.addPlot(title="Phase Difference")
phase_img = pg.ImageItem()
phase_plot.addItem(phase_img)
# Create colorbar for the phase heatmap
phase_bar = pg.ColorBarItem(
    values=(0, 360),  # Assuming phase is in degrees
    colorMap=pg.colormap.get('viridis'),
    orientation='h'
)
phase_bar.setImageItem(phase_img)
phase_plot.setLabel('bottom', 'Time', units='s')
phase_plot.setLabel('left', 'Frequency Index')

# mag time plot - converted to heatmap
win.nextRow()
mag_time_plot = win.addPlot(title="Magnitude over time")
mag_time_img = pg.ImageItem()
mag_time_plot.addItem(mag_time_img)
# Create colorbar for magnitude heatmap
mag_bar = pg.ColorBarItem(
    colorMap=pg.colormap.get('plasma'),
    orientation='h'
)
mag_bar.setImageItem(mag_time_img)
mag_time_plot.setLabel('bottom', 'Time', units='s')
mag_time_plot.setLabel('left', 'Frequency Index')
start_time = time.time()

# hs hp plot - converted to heatmap
win.nextRow()
hshp_real_plot = win.addPlot(title="HsHp Reals")
hshp_real_img = pg.ImageItem()
hshp_real_plot.addItem(hshp_real_img)
# Create colorbar for HsHp heatmap
hshp_real_bar = pg.ColorBarItem(
    colorMap=pg.colormap.get('plasma'),  # Good for showing both real and imaginary data
    orientation='h'
)
hshp_real_bar.setImageItem(hshp_real_img)
hshp_real_plot.setLabel('bottom', 'Time', units='s')
hshp_real_plot.setLabel('left', 'Frequency Index')

# hs hp plot - converted to heatmap
win.nextRow()
hshp_imag_plot = win.addPlot(title="HsHp Imag")
hshp_imag_img = pg.ImageItem()
hshp_imag_plot.addItem(hshp_imag_img)
# Create colorbar for HsHp heatmap
hshp_imag_bar = pg.ColorBarItem(
    colorMap=pg.colormap.get('plasma'),  # Good for showing both real and imaginary data
    orientation='h'
)
hshp_imag_bar.setImageItem(hshp_imag_img)
hshp_imag_plot.setLabel('bottom', 'Time', units='s')
hshp_imag_plot.setLabel('left', 'Frequency Index')

# Add frequency ticks for the heatmaps
freq_ticks = [(i, f"{freq:.1e}Hz") for i, freq in enumerate(frequencies)]
phase_plot.getAxis('left').setTicks([freq_ticks])
mag_time_plot.getAxis('left').setTicks([freq_ticks])
hshp_real_plot.getAxis('left').setTicks([freq_ticks])
hshp_imag_plot.getAxis('left').setTicks([freq_ticks])

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

# Add a line plot for confidence values instead of a heatmap
win.nextRow()
confidence_plot = win.addPlot(title="Detection Confidence")
confidence_curve = confidence_plot.plot(pen={'color': (255, 165, 0), 'width': 3}, name="Confidence")
confidence_plot.setLabel('left', 'Confidence', units='')
confidence_plot.setLabel('bottom', 'Time', units='s')
confidence_plot.setYRange(0, 1)  # Set Y range from 0 to 1 for confidence

# Create a threshold line at 0.5
threshold_line = pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
confidence_plot.addItem(threshold_line)
threshold_text = pg.TextItem("Threshold (0.5)", color='r', anchor=(0, 1.1))
threshold_text.setPos(0, 0.5)
confidence_plot.addItem(threshold_text)

# Create array to store confidence values
confidence_values = np.array([])

# Create a dedicated control panel row at the bottom of the window
win.nextRow()

button_proxy = pg.QtWidgets.QGraphicsProxyWidget()
button_proxy.setWidget(button_frame)

# Add the widgets in a horizontal layout
control_layout = win.addLayout(row=6, col=0)
control_layout.addItem(button_proxy, row=0, col=1)

# Clear data function - properly resets all arrays
def clear_data():
    global timestamps, relativePhases, peak_magnitudes, hshpReals, hshpImag, data_count, confidence_values
    # Reset all arrays
    timestamps = np.array([])
    peak_magnitudes = np.array([])
    relativePhases = np.array([])
    hshpReals = np.array([])
    hshpImag = np.array([])
    confidence_values = np.array([])
    data_count = 0
    
    # Clear the image plots
    phase_img.clear()
    mag_time_img.clear()
    hshp_real_img.clear()
    hshp_imag_img.clear()
    confidence_curve.clear() # Clear confidence curve

    print("History cleared.")

def export_data_csv():
    """Export the current data to a CSV file with multiple frequencies"""
    try:
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        
        # Create filename with timestamp
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/peak_magnitude_data_{timestamp_str}.csv"
        
        # Open CSV file for writing
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            # Write header row with all frequencies
            header = ['Timestamp (s)']
            
            # Add columns for each frequency
            for freq in frequencies:
                header.extend([
                    f'Magnitude ({freq:.1e} Hz)',
                    f'Phase ({freq:.1e} Hz)',
                    f'HsHp Real ({freq:.1e} Hz)',
                    f'HsHp Imag ({freq:.1e} Hz)'
                ])
            csvwriter.writerow(header)
            
            # Write data rows
            for i in range(len(timestamps)):
                row_data = [timestamps[i]]
                
                # Add data for each frequency if available
                if i < len(peak_magnitudes):
                    for j in range(len(frequencies)):
                        row_data.extend([
                            peak_magnitudes[i][j] if j < len(peak_magnitudes[i]) else '',
                            relativePhases[i][j] if j < len(relativePhases[i]) else '',
                            hshpReals[i][j] if j < len(hshpReals[i]) else '',
                            hshpImag[i][j] if j < len(hshpImag[i]) else ''
                        ])
                
                csvwriter.writerow(row_data)
        
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

# Function to update heatmap scaling and time axis
def update_heatmap_axes():
    if len(timestamps) > 0:
        # Create time ticks based on actual timestamps
        time_ticks = [(i, f"{timestamps[i]:.1f}s") for i in range(0, len(timestamps), max(1, len(timestamps)//5))]
        
        # Update axes ticks
        phase_plot.getAxis('bottom').setTicks([time_ticks])
        mag_time_plot.getAxis('bottom').setTicks([time_ticks])
        hshp_real_plot.getAxis('bottom').setTicks([time_ticks])
        hshp_imag_plot.getAxis('bottom').setTicks([time_ticks])

while True:
    phase_diff_1d = []
    magnitude_1d = []
    for freq_idx, frequency in enumerate(frequencies):
            
        # Configure analog input
        dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))

        set_out_freq(frequency)
        ## Turn on the output 
        dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))

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
        phase_diff_1d.append(phase2[target_idx] - phase1[target_idx])
        magnitude_1d.append(magnitude2[target_idx] - magnitude1[target_idx])


    h_mag_sec_lin = np.pow(10, np.array(magnitude_1d)/20)
    h_secondary_complex = h_mag_sec_lin * np.exp(1j * np.deg2rad(phase_diff_1d))
    hshp = h_secondary_complex / h_prim_complex

    # Capture the current time
    current_time = time.time() - start_time
    timestamps = np.append(timestamps, current_time)
    
    # Store the current data for all frequencies
    if len(peak_magnitudes) == 0:
        # First data point
        peak_magnitudes = np.array([magnitude_1d])
        relativePhases = np.array([phase_diff_1d])
        hshpReals = np.array([np.real(hshp)])
        hshpImag = np.array([np.imag(hshp)])
    else:
        # Add new data
        peak_magnitudes = np.vstack((peak_magnitudes, magnitude_1d))
        relativePhases = np.vstack((relativePhases, phase_diff_1d))
        hshpReals = np.vstack((hshpReals, np.real(hshp)))
        hshpImag = np.vstack((hshpImag, np.imag(hshp)))

    data_count += 1

    # Run neural network inference if model is loaded
    if model_loaded:
        try:
            # Prepare input data for the model (convert complex values to real and imaginary parts)
            real_part = np.real(h_secondary_complex)
            imag_part = np.imag(h_secondary_complex)
            
            # Concatenate real and imaginary parts
            model_input = np.concatenate([real_part, imag_part])
            
            # Normalize with the scaler if available
            if scaler is not None:
                model_input = scaler.transform(model_input.reshape(1, -1)).reshape(-1)
            
            # Convert to tensor
            model_input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                presence_prob, location, size = trained_model(model_input_tensor)
            
            # Convert to numpy and get the confidence value
            confidence = presence_prob.item()
            
            # Update confidence values
            if len(confidence_values) == 0:
                confidence_values = np.array([confidence])
            else:
                confidence_values = np.append(confidence_values, confidence)
                
            # # Print the results
            # x, y, z = location.squeeze().numpy()
            # length, radius = size.squeeze().numpy()
            # print(f"Detection confidence: {confidence:.4f}, Location: ({x:.2f}, {y:.2f}, {z:.2f}), "
            #       f"Size: length={length:.2f}m, radius={radius:.2f}m")
                  
        except Exception as e:
            print(f"Error during inference: {e}")
            # Add a dummy confidence value if inference fails
            if len(confidence_values) == 0:
                confidence_values = np.array([0.0])
            else:
                confidence_values = np.append(confidence_values, 0.0)
    else:
        # Add a dummy confidence value if model isn't loaded
        if len(confidence_values) == 0:
            confidence_values = np.array([0.0])
        else:
            confidence_values = np.append(confidence_values, 0.0)

    # Update plots when we have data
    if data_count > 0:
        # Update spectrograms with proper scaling
        magnitude_curve1.setData(freq, magnitude1)
        magnitude_curve2.setData(freq, magnitude2)
        relativePhases = np.unwrap(relativePhases, 180, axis=0)
        # Update heatmaps
        phase_img.setImage(relativePhases)
        mag_time_img.setImage(peak_magnitudes)
        mean_minus_reals = hshpReals - np.mean(hshpReals, axis=0)
        mean_minus_imags = hshpImag - np.mean(hshpImag, axis=0)
        hshp_real_img.setImage(hshpReals - np.mean(hshpReals, axis=0))
        hshp_imag_img.setImage(hshpImag - np.mean(hshpImag, axis=0))
        confidence_curve.setData(timestamps, confidence_values)

        # Update heatmap axes and scaling
        update_heatmap_axes()
        
        # Auto-scale the colorbars based on data
        phase_bar.setLevels((np.min(relativePhases), np.max(relativePhases)))
        mag_bar.setLevels((np.min(peak_magnitudes), np.max(peak_magnitudes)))
        hshp_real_bar.setLevels((np.min(mean_minus_reals ), np.max(mean_minus_reals)))
        hshp_imag_bar.setLevels((np.min(mean_minus_imags), np.max(mean_minus_imags)))

    # Process events to update the plots
    app.processEvents()