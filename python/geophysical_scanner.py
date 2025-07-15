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
import socket
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QApplication
import torch 
from training import FDEM1DInversionNet
import pickle

# Import the sensor fusion service
from sensor_fusion_service import SensorFusionService, SensorFusionState

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

class GeophysicalScanner:
    def __init__(self, sensor_port=5001, debug=False):
        self.sensor_port = sensor_port
        self.debug = debug
        
        # Initialize DWF device
        self.init_dwf_device()
        
        # Initialize sensor fusion service
        self.sensor_service = SensorFusionService(port=sensor_port, debug=debug)
        
        # Get primary response for frequencies
        self.h_prim_complex = get_primary(frequencies)
        
        # Initialize EM measurement variables
        self.cSamples = 8 * 1024
        self.FIXED_SAMPLE_RATE = 2e5
        self.hzRate = self.FIXED_SAMPLE_RATE
        self.rgdSamples1 = (c_double * self.cSamples)()
        self.rgdSamples2 = (c_double * self.cSamples)()
        self.sts = c_int()
        self.channel = c_int(0)
        
        # Configure analog out
        self.configure_analog_out()
        
        # Configure analog in
        self.configure_analog_in()
        
        # Load neural network model
        self.load_model()
        
        # Data storage for EM measurements
        self.peak_magnitudes = np.array([])
        self.relativePhases = np.array([])
        self.hshpReals = np.array([])
        self.hshpImag = np.array([])
        self.timestamps = np.array([])
        self.confidence_values = np.array([])
        self.data_count = 0
        self.start_time = time.time()
        
        # Data storage for position tracking
        self.position_history = np.array([]).reshape(0, 3)
        self.velocity_history = np.array([]).reshape(0, 3)
        self.euler_history = np.array([]).reshape(0, 3)
        self.position_timestamps = np.array([])
        self.speed_history = np.array([])
        
        # Setup GUI
        self.setup_gui()
        self.setup_timer()
        
    def init_dwf_device(self):
        """Initialize DWF device"""
        if sys.platform.startswith("win"):
            self.dwf = cdll.dwf
        elif sys.platform.startswith("darwin"):
            self.dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        else:
            self.dwf = cdll.LoadLibrary("libdwf.so")

        # Continue running after device close, prevent temperature drifts
        self.dwf.FDwfParamSet(c_int(4), c_int(0))

        # Print DWF version
        version = create_string_buffer(16)
        self.dwf.FDwfGetVersion(version)
        print("DWF Version: "+str(version.value))

        # Open device
        self.hdwf = c_int()
        print("Opening first device...")
        self.dwf.FDwfDeviceOpen(c_int(-1), byref(self.hdwf))

        if self.hdwf.value == hdwfNone.value:
            print("failed to open device")
            quit()
            
    def configure_analog_out(self):
        """Configure analog output for frequency sweep"""
        self.dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, self.channel, AnalogOutNodeCarrier, c_int(1))
        self.dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, self.channel, AnalogOutNodeCarrier, funcSine)
        self.dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, self.channel, AnalogOutNodeCarrier, c_double(frequencies[0]))
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, self.channel, AnalogOutNodeCarrier, c_double(2.0))
        self.dwf.FDwfAnalogOutNodeOffsetSet(self.hdwf, self.channel, AnalogOutNodeCarrier, c_double(0.0))
        self.dwf.FDwfAnalogOutRepeatSet(self.hdwf, self.channel, c_int(1))
        self.dwf.FDwfAnalogOutRunSet(self.hdwf, self.channel, c_double(2.0))

    def configure_analog_in(self):
        """Configure analog input"""
        print(f"Configure analog in for frequency {frequencies[0]} Hz")
        self.dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.hzRate))
        self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(0), c_double(20))
        self.dwf.FDwfAnalogInChannelRangeSet(self.hdwf, c_int(1), c_double(0.01))
        self.dwf.FDwfAnalogInBufferSizeSet(self.hdwf, c_int(self.cSamples))
        self.dwf.FDwfAnalogInTriggerSourceSet(self.hdwf, trigsrcAnalogOut1)
        self.dwf.FDwfAnalogInTriggerPositionSet(self.hdwf, c_double(0.5 * self.cSamples / self.hzRate))

    def set_out_freq(self, frequency):
        """Set output frequency"""
        self.dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, self.channel, AnalogOutNodeCarrier, c_double(frequency))

    def load_model(self):
        """Load the trained neural network model"""
        print("Loading trained FDEM inversion model...")
        try:
            checkpoint = torch.load('models/fdem_1d_model.pth')
            with open('models/fdem_1d_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Initialize the model
            self.trained_model = FDEM1DInversionNet(num_freqs=len(frequencies))
            self.trained_model.load_state_dict(checkpoint['model_state_dict'])
            self.trained_model.eval()
            
            print("Model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Continuing without neural network inference...")
            self.model_loaded = False

    def get_local_ip(self):
        """Get the local IP address."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def setup_gui(self):
        """Initialize the GUI components."""
        # Initialize PyQt application
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        # Main window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Geophysical Scanner")
        self.win.resize(1600, 1200)
        self.win.setWindowTitle('Geophysical Scanner - EM + Position Tracking')
        
        # Set dark theme
        pg.setConfigOption('background', (30, 30, 30))
        pg.setConfigOption('foreground', 'w')
        
        # Row 1: EM Magnitude and Phase plots
        self.magnitude_plot = self.win.addPlot(title="EM Magnitude")
        self.magnitude_plot.addLegend()
        self.magnitude_curve1 = self.magnitude_plot.plot(pen='r', name="Channel 1")
        self.magnitude_curve2 = self.magnitude_plot.plot(pen='b', name="Channel 2")
        self.magnitude_plot.setLabel('left', 'Magnitude', units='dB')
        self.magnitude_plot.setLabel('bottom', 'Frequency', units='Hz')
        
        self.phase_plot = self.win.addPlot(title="Phase Difference")
        self.phase_img = pg.ImageItem()
        self.phase_plot.addItem(self.phase_img)
        self.phase_bar = pg.ColorBarItem(
            values=(0, 360),
            colorMap=pg.colormap.get('viridis'),
            orientation='h'
        )
        self.phase_bar.setImageItem(self.phase_img)
        self.phase_plot.setLabel('bottom', 'Time', units='s')
        self.phase_plot.setLabel('left', 'Frequency Index')
        
        # Row 2: EM Time plots and Position tracking
        self.win.nextRow()
        
        self.mag_time_plot = self.win.addPlot(title="EM Magnitude over Time")
        self.mag_time_img = pg.ImageItem()
        self.mag_time_plot.addItem(self.mag_time_img)
        self.mag_bar = pg.ColorBarItem(
            colorMap=pg.colormap.get('plasma'),
            orientation='h'
        )
        self.mag_bar.setImageItem(self.mag_time_img)
        self.mag_time_plot.setLabel('bottom', 'Time', units='s')
        self.mag_time_plot.setLabel('left', 'Frequency Index')
        
        # XY Position Plot
        self.xy_plot = self.win.addPlot(title="XY Position Track")
        self.xy_plot.setLabel('left', 'Y Position', units='m')
        self.xy_plot.setLabel('bottom', 'X Position', units='m')
        self.xy_plot.setAspectLocked(True)
        self.xy_plot.showGrid(True, True, alpha=0.3)
        
        # Position trail
        self.position_trail = self.xy_plot.plot(pen=pg.mkPen(color=(100, 149, 237), width=2), 
                                               name="Position Trail")
        self.current_position = self.xy_plot.plot(pen=None, 
                                                 symbol='o', 
                                                 symbolBrush=(255, 0, 0), 
                                                 symbolSize=10, 
                                                 name="Current Position")
        self.start_position = self.xy_plot.plot(pen=None, 
                                               symbol='s', 
                                               symbolBrush=(0, 255, 0), 
                                               symbolSize=12, 
                                               name="Start Position")
        self.xy_plot.addLegend()
        
        # Row 3: HsHp and Speed plots
        self.win.nextRow()
        
        self.hshp_real_plot = self.win.addPlot(title="HsHp Real")
        self.hshp_real_img = pg.ImageItem()
        self.hshp_real_plot.addItem(self.hshp_real_img)
        self.hshp_real_bar = pg.ColorBarItem(
            colorMap=pg.colormap.get('plasma'),
            orientation='h'
        )
        self.hshp_real_bar.setImageItem(self.hshp_real_img)
        self.hshp_real_plot.setLabel('bottom', 'Time', units='s')
        self.hshp_real_plot.setLabel('left', 'Frequency Index')
        
        # Speed over time
        self.speed_plot = self.win.addPlot(title="Speed over Time")
        self.speed_plot.setLabel('left', 'Speed', units='m/s')
        self.speed_plot.setLabel('bottom', 'Time', units='s')
        self.speed_curve = self.speed_plot.plot(pen=pg.mkPen(color=(255, 165, 0), width=2))
        self.speed_plot.showGrid(True, True, alpha=0.3)
        
        # Row 4: HsHp Imaginary and Confidence
        self.win.nextRow()
        
        self.hshp_imag_plot = self.win.addPlot(title="HsHp Imaginary")
        self.hshp_imag_img = pg.ImageItem()
        self.hshp_imag_plot.addItem(self.hshp_imag_img)
        self.hshp_imag_bar = pg.ColorBarItem(
            colorMap=pg.colormap.get('plasma'),
            orientation='h'
        )
        self.hshp_imag_bar.setImageItem(self.hshp_imag_img)
        self.hshp_imag_plot.setLabel('bottom', 'Time', units='s')
        self.hshp_imag_plot.setLabel('left', 'Frequency Index')
        
        # Detection confidence
        self.confidence_plot = self.win.addPlot(title="Detection Confidence")
        self.confidence_curve = self.confidence_plot.plot(pen={'color': (255, 165, 0), 'width': 3})
        self.confidence_plot.setLabel('left', 'Confidence', units='')
        self.confidence_plot.setLabel('bottom', 'Time', units='s')
        self.confidence_plot.setYRange(0, 1)
        
        # Threshold line
        threshold_line = pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen('r', width=1, style=Qt.DashLine))
        self.confidence_plot.addItem(threshold_line)
        
        # Row 5: Control panel
        self.win.nextRow()
        self.setup_control_panel()
        
        # Add frequency ticks for heatmaps
        freq_ticks = [(i, f"{freq:.1e}Hz") for i, freq in enumerate(frequencies)]
        self.phase_plot.getAxis('left').setTicks([freq_ticks])
        self.mag_time_plot.getAxis('left').setTicks([freq_ticks])
        self.hshp_real_plot.getAxis('left').setTicks([freq_ticks])
        self.hshp_imag_plot.getAxis('left').setTicks([freq_ticks])
        
        # Status text
        self.status_text = pg.TextItem("Starting geophysical scanner...", 
                                     color=(255, 255, 255), 
                                     anchor=(0, 0))
        self.xy_plot.addItem(self.status_text)
        self.status_text.setPos(0.02, 0.95)

    def setup_control_panel(self):
        """Setup the control panel with buttons and statistics."""
        button_frame = self.create_styled_frame()
        button_layout = QVBoxLayout(button_frame)
        
        # Header
        header = QLabel("Geophysical Scanner Controls")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        header.setStyleSheet("color: #FFFFFF;")
        header.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(header)
        
        # Statistics display
        self.stats_label = QLabel("No data yet")
        self.stats_label.setStyleSheet("color: #CCCCCC; font-family: monospace;")
        self.stats_label.setAlignment(Qt.AlignLeft)
        button_layout.addWidget(self.stats_label)
        
        # Button container
        btn_container = QHBoxLayout()
        
        # Clear button
        clear_button = pg.QtWidgets.QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_data)
        clear_button.setMinimumHeight(30)
        clear_button.setStyleSheet(self.get_button_style("#E74C3C", "#C0392B", "#A93226"))
        btn_container.addWidget(clear_button)
        
        # Export CSV button
        export_button = pg.QtWidgets.QPushButton("Export Data")
        export_button.clicked.connect(self.export_data_csv)
        export_button.setMinimumHeight(30)
        export_button.setStyleSheet(self.get_button_style("#27AE60", "#229954", "#1E8449"))
        btn_container.addWidget(export_button)
        
        button_layout.addLayout(btn_container)
        
        # Add to window
        button_proxy = pg.QtWidgets.QGraphicsProxyWidget()
        button_proxy.setWidget(button_frame)
        control_layout = self.win.addLayout(row=4, col=0, colspan=2)
        control_layout.addItem(button_proxy, row=0, col=0)

    def create_styled_frame(self):
        """Create a styled frame for the control panel."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setLineWidth(2)
        
        palette = frame.palette()
        palette.setColor(QPalette.Window, QColor(50, 50, 50))
        frame.setPalette(palette)
        frame.setAutoFillBackground(True)
        
        return frame
        
    def get_button_style(self, normal_color, hover_color, pressed_color):
        """Get button style CSS."""
        return f"""
            QPushButton {{
                background-color: {normal_color};
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """

    def setup_timer(self):
        """Setup timer for periodic updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sensor_plots)
        self.timer.start(50)  # Update every 50ms

    def start(self):
        """Start both EM scanning and position tracking services."""
        self.sensor_service.start()
        local_ip = self.get_local_ip()
        print(f"Geophysical scanner started")
        print(f"Position tracking on port {self.sensor_port}")
        print(f"Connect from your phone to: {local_ip}:{self.sensor_port}")

    def perform_em_measurement(self):
        """Perform one complete electromagnetic measurement sweep."""
        phase_diff_1d = []
        magnitude_1d = []
        
        for freq_idx, frequency in enumerate(frequencies):
            # Configure analog input
            self.dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1))
            
            self.set_out_freq(frequency)
            # Turn on the output 
            self.dwf.FDwfAnalogOutConfigure(self.hdwf, self.channel, c_int(1))
            
            # Wait for data to finish 
            while True:
                self.dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(self.sts))
                if self.sts.value == DwfStateDone.value:
                    break
                time.sleep(0.01)
            
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(0), self.rgdSamples1, len(self.rgdSamples1))
            self.dwf.FDwfAnalogInStatusData(self.hdwf, c_int(1), self.rgdSamples2, len(self.rgdSamples2))
            
            # Calculate FFT for both signals
            fft1 = np.fft.rfft(self.rgdSamples1)
            fft2 = np.fft.rfft(self.rgdSamples2)
            
            # Calculate magnitude in dB
            magnitude1 = 20 * np.log10(np.abs(fft1) / self.cSamples)
            magnitude2 = 20 * np.log10(np.abs(fft2) / self.cSamples)
            
            # Calculate frequency axis
            freq = np.fft.rfftfreq(self.cSamples, 1 / self.hzRate)
            
            # Calculate phase in degrees
            phase1 = np.angle(fft1, deg=True)
            phase2 = np.angle(fft2, deg=True)
            
            # Find the index of the target frequency
            target_idx = np.argmax(magnitude1[10:-10])+10 
            phase_diff_1d.append(phase2[target_idx] - phase1[target_idx])
            magnitude_1d.append(magnitude2[target_idx] - magnitude1[target_idx])
        
        # Store the latest frequency response for plotting
        self.freq = freq
        self.magnitude1 = magnitude1
        self.magnitude2 = magnitude2
        
        # Calculate secondary field response
        h_mag_sec_lin = np.pow(10, np.array(magnitude_1d)/20)
        h_secondary_complex = h_mag_sec_lin * np.exp(1j * np.deg2rad(phase_diff_1d))
        hshp = h_secondary_complex / self.h_prim_complex
        
        # Store EM data
        current_time = time.time() - self.start_time
        self.timestamps = np.append(self.timestamps, current_time)
        
        if len(self.peak_magnitudes) == 0:
            self.peak_magnitudes = np.array([magnitude_1d])
            self.relativePhases = np.array([phase_diff_1d])
            self.hshpReals = np.array([np.real(hshp)])
            self.hshpImag = np.array([np.imag(hshp)])
        else:
            self.peak_magnitudes = np.vstack((self.peak_magnitudes, magnitude_1d))
            self.relativePhases = np.vstack((self.relativePhases, phase_diff_1d))
            self.hshpReals = np.vstack((self.hshpReals, np.real(hshp)))
            self.hshpImag = np.vstack((self.hshpImag, np.imag(hshp)))
        
        self.data_count += 1
        
        # Run neural network inference
        self.run_inference(h_secondary_complex)
        
        return current_time

    def run_inference(self, h_secondary_complex):
        """Run neural network inference on EM data."""
        if self.model_loaded:
            try:
                real_part = np.real(h_secondary_complex)
                imag_part = np.imag(h_secondary_complex)
                model_input = np.concatenate([real_part, imag_part])
                
                if self.scaler is not None:
                    model_input = self.scaler.transform(model_input.reshape(1, -1)).reshape(-1)
                
                model_input_tensor = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0)
                
                with torch.no_grad():
                    presence_prob, location, size = self.trained_model(model_input_tensor)
                
                confidence = presence_prob.item()
                
                if len(self.confidence_values) == 0:
                    self.confidence_values = np.array([confidence])
                else:
                    self.confidence_values = np.append(self.confidence_values, confidence)
                    
            except Exception as e:
                print(f"Error during inference: {e}")
                confidence = 0.0
                if len(self.confidence_values) == 0:
                    self.confidence_values = np.array([confidence])
                else:
                    self.confidence_values = np.append(self.confidence_values, confidence)
        else:
            confidence = 0.0
            if len(self.confidence_values) == 0:
                self.confidence_values = np.array([confidence])
            else:
                self.confidence_values = np.append(self.confidence_values, confidence)

    def update_sensor_plots(self):
        """Update position tracking plots with latest sensor data."""
        # Get current state from sensor fusion
        state = self.sensor_service.get_current_state()
        
        if state is not None:
            # Add new position data
            self.position_history = np.vstack([self.position_history, state.position])
            self.velocity_history = np.vstack([self.velocity_history, state.velocity])
            self.euler_history = np.vstack([self.euler_history, state.euler])
            self.position_timestamps = np.append(self.position_timestamps, state.timestamp)
            
            # Calculate speed
            speed = np.linalg.norm(state.velocity)
            self.speed_history = np.append(self.speed_history, speed)
            
            # Update position plots
            if len(self.position_history) > 0:
                x_positions = self.position_history[:, 0]
                y_positions = self.position_history[:, 1]
                
                self.position_trail.setData(x_positions, y_positions)
                self.current_position.setData([x_positions[-1]], [y_positions[-1]])
                
                if len(x_positions) > 0:
                    self.start_position.setData([x_positions[0]], [y_positions[0]])
            
            # Update speed plot
            if len(self.position_timestamps) > 1:
                relative_times = self.position_timestamps - self.position_timestamps[0]
                self.speed_curve.setData(relative_times, self.speed_history)
            
            # Update status
            self.update_status(state)

    def update_status(self, state):
        """Update status text and statistics."""
        status_lines = [
            f"EM Samples: {self.data_count}",
            f"Position: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}) m",
            f"Speed: {np.linalg.norm(state.velocity):.2f} m/s",
            f"Heading: {np.degrees(state.euler[2]):.1f}°"
        ]
        
        if state.gps_reference:
            status_lines.append(f"GPS Ref: ({state.gps_reference[0]:.6f}, {state.gps_reference[1]:.6f})")
        
        if state.last_gps_accuracy:
            status_lines.append(f"GPS Acc: {state.last_gps_accuracy[0]:.1f}m")
            
        self.status_text.setText("\\n".join(status_lines))
        
        # Statistics panel
        if len(self.position_history) > 1 and self.data_count > 0:
            total_distance = self.calculate_total_distance()
            max_speed = np.max(self.speed_history) if len(self.speed_history) > 0 else 0
            avg_speed = np.mean(self.speed_history) if len(self.speed_history) > 0 else 0
            duration = self.position_timestamps[-1] - self.position_timestamps[0] if len(self.position_timestamps) > 1 else 0
            latest_confidence = self.confidence_values[-1] if len(self.confidence_values) > 0 else 0
            
            stats_text = f"""Scanner Statistics:
EM Duration: {self.timestamps[-1] if len(self.timestamps) > 0 else 0:.1f} seconds
Position Duration: {duration:.1f} seconds
EM Measurements: {self.data_count}
Position Samples: {len(self.position_history)}
Total Distance: {total_distance:.2f} meters
Max Speed: {max_speed:.2f} m/s
Latest Confidence: {latest_confidence:.3f}
Current Position: ({state.position[0]:.2f}, {state.position[1]:.2f}) m"""
            
            self.stats_label.setText(stats_text)

    def calculate_total_distance(self):
        """Calculate total distance traveled."""
        if len(self.position_history) < 2:
            return 0.0
        distances = np.linalg.norm(np.diff(self.position_history, axis=0), axis=1)
        return np.sum(distances)

    def update_em_plots(self):
        """Update EM plots with latest data."""
        if self.data_count > 0:
            # Update spectrograms
            self.magnitude_curve1.setData(self.freq, self.magnitude1)
            self.magnitude_curve2.setData(self.freq, self.magnitude2)
            
            # Unwrap phases and update heatmaps
            self.relativePhases = np.unwrap(self.relativePhases, 180, axis=0)
            self.phase_img.setImage(self.relativePhases)
            self.mag_time_img.setImage(self.peak_magnitudes)
            
            # Update HsHp plots (mean-subtracted)
            mean_minus_reals = self.hshpReals - np.mean(self.hshpReals, axis=0)
            mean_minus_imags = self.hshpImag - np.mean(self.hshpImag, axis=0)
            self.hshp_real_img.setImage(mean_minus_reals)
            self.hshp_imag_img.setImage(mean_minus_imags)
            
            # Update confidence plot
            self.confidence_curve.setData(self.timestamps, self.confidence_values)
            
            # Update heatmap axes
            self.update_heatmap_axes()
            
            # Auto-scale colorbars
            self.phase_bar.setLevels((np.min(self.relativePhases), np.max(self.relativePhases)))
            self.mag_bar.setLevels((np.min(self.peak_magnitudes), np.max(self.peak_magnitudes)))
            self.hshp_real_bar.setLevels((np.min(mean_minus_reals), np.max(mean_minus_reals)))
            self.hshp_imag_bar.setLevels((np.min(mean_minus_imags), np.max(mean_minus_imags)))

    def update_heatmap_axes(self):
        """Update heatmap time axes."""
        if len(self.timestamps) > 0:
            time_ticks = [(i, f"{self.timestamps[i]:.1f}s") for i in range(0, len(self.timestamps), max(1, len(self.timestamps)//5))]
            self.phase_plot.getAxis('bottom').setTicks([time_ticks])
            self.mag_time_plot.getAxis('bottom').setTicks([time_ticks])
            self.hshp_real_plot.getAxis('bottom').setTicks([time_ticks])
            self.hshp_imag_plot.getAxis('bottom').setTicks([time_ticks])

    def clear_data(self):
        """Clear all stored data and reset plots."""
        # Clear EM data
        self.timestamps = np.array([])
        self.peak_magnitudes = np.array([])
        self.relativePhases = np.array([])
        self.hshpReals = np.array([])
        self.hshpImag = np.array([])
        self.confidence_values = np.array([])
        self.data_count = 0
        
        # Clear position data
        self.position_history = np.array([]).reshape(0, 3)
        self.velocity_history = np.array([]).reshape(0, 3)
        self.euler_history = np.array([]).reshape(0, 3)
        self.position_timestamps = np.array([])
        self.speed_history = np.array([])
        
        # Clear all plots
        self.phase_img.clear()
        self.mag_time_img.clear()
        self.hshp_real_img.clear()
        self.hshp_imag_img.clear()
        self.confidence_curve.clear()
        self.position_trail.clear()
        self.current_position.clear()
        self.start_position.clear()
        self.speed_curve.clear()
        
        self.status_text.setText("Data cleared. Waiting for new measurements...")
        self.stats_label.setText("No data yet")
        
        print("All data cleared.")

    def export_data_csv(self):
        """Export both EM and position data to CSV files."""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export EM data
            em_filename = f"data/em_data_{timestamp_str}.csv"
            with open(em_filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                # EM data header
                header = ['Timestamp (s)']
                for freq in frequencies:
                    header.extend([
                        f'Magnitude ({freq:.1e} Hz)',
                        f'Phase ({freq:.1e} Hz)',
                        f'HsHp Real ({freq:.1e} Hz)',
                        f'HsHp Imag ({freq:.1e} Hz)'
                    ])
                header.append('Confidence')
                csvwriter.writerow(header)
                
                # EM data rows
                for i in range(len(self.timestamps)):
                    row_data = [self.timestamps[i]]
                    
                    if i < len(self.peak_magnitudes):
                        for j in range(len(frequencies)):
                            row_data.extend([
                                self.peak_magnitudes[i][j] if j < len(self.peak_magnitudes[i]) else '',
                                self.relativePhases[i][j] if j < len(self.relativePhases[i]) else '',
                                self.hshpReals[i][j] if j < len(self.hshpReals[i]) else '',
                                self.hshpImag[i][j] if j < len(self.hshpImag[i]) else ''
                            ])
                    
                    # Add confidence
                    if i < len(self.confidence_values):
                        row_data.append(self.confidence_values[i])
                    else:
                        row_data.append('')
                    
                    csvwriter.writerow(row_data)
            
            # Export position data
            pos_filename = f"data/position_data_{timestamp_str}.csv"
            if len(self.position_timestamps) > 0:
                with open(pos_filename, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    
                    # Position data header
                    header = ['Timestamp', 'Relative_Time_s', 'X_m', 'Y_m', 'Z_m', 
                             'VX_ms', 'VY_ms', 'VZ_ms', 'Speed_ms',
                             'Pitch_deg', 'Roll_deg', 'Yaw_deg']
                    csvwriter.writerow(header)
                    
                    # Position data rows
                    relative_times = self.position_timestamps - self.position_timestamps[0]
                    for i in range(len(self.position_timestamps)):
                        row = [
                            self.position_timestamps[i],
                            relative_times[i],
                            self.position_history[i, 0],
                            self.position_history[i, 1], 
                            self.position_history[i, 2],
                            self.velocity_history[i, 0],
                            self.velocity_history[i, 1],
                            self.velocity_history[i, 2],
                            self.speed_history[i],
                            np.degrees(self.euler_history[i, 0]),  # pitch
                            np.degrees(self.euler_history[i, 1]),  # roll
                            np.degrees(self.euler_history[i, 2])   # yaw
                        ]
                        csvwriter.writerow(row)
            
            print(f"EM data exported to {em_filename}")
            if len(self.position_timestamps) > 0:
                print(f"Position data exported to {pos_filename}")
            
            # Show success message
            msg_box = pg.QtWidgets.QMessageBox()
            msg_box.setWindowTitle("Export Successful")
            msg_text = f"Data exported to:\\n{os.path.abspath(em_filename)}"
            if len(self.position_timestamps) > 0:
                msg_text += f"\\n{os.path.abspath(pos_filename)}"
            msg_box.setText(msg_text)
            msg_box.exec_()
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            error_box = pg.QtWidgets.QMessageBox()
            error_box.setWindowTitle("Export Error")
            error_box.setText(f"Error exporting data: {e}")
            error_box.exec_()

    def run_measurement_loop(self):
        """Main measurement loop that performs EM sweeps and updates position."""
        while True:
            try:
                # Perform EM measurement
                measurement_time = self.perform_em_measurement()
                
                # Update EM plots
                self.update_em_plots()
                
                # Process Qt events
                self.app.processEvents()
                
            except KeyboardInterrupt:
                print("\\nMeasurement loop interrupted")
                break
            except Exception as e:
                print(f"Error in measurement loop: {e}")
                time.sleep(0.1)

    def run(self):
        """Run the complete geophysical scanner."""
        self.start()
        
        try:
            # Start measurement loop in a separate thread or use timer
            self.measurement_timer = QTimer()
            self.measurement_timer.timeout.connect(lambda: self.perform_em_measurement() or self.update_em_plots())
            self.measurement_timer.start(100)  # EM measurements every 100ms
            
            # Run Qt event loop
            if hasattr(self.app, 'exec_'):
                self.app.exec_()
            else:
                self.app.exec()
        except KeyboardInterrupt:
            print("\\nShutting down...")
        finally:
            self.sensor_service.stop()

# Main execution
if __name__ == "__main__":
    scanner = GeophysicalScanner(sensor_port=5001, debug=True)
    scanner.run()