from ctypes import *
from dwfconstants import *
import math
import time
import numpy as np
import scipy
import control
from control.matlab import ss, bode
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from collections import deque

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

class SingleFrequencyAnalyzerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Single Frequency Signal Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Parameters
        self.frequency = 10000.0  # Default 10 kHz
        self.history_length = 100  # Number of points to keep in history
        self.ema_enabled = True  # Enable EMA filtering by default
        self.ema_alpha = 0.2  # Default EMA alpha value (smoothing factor)
        
        # EMA filter state
        self.ema_magnitude = None
        self.ema_phase = None
        self.ema_hs_hp_real = None
        self.ema_hs_hp_imag = None
        
        # Time history data
        self.timestamps = deque(maxlen=self.history_length)
        self.magnitude_history = deque(maxlen=self.history_length)
        self.phase_history = deque(maxlen=self.history_length)
        self.hs_hp_real_history = deque(maxlen=self.history_length)
        self.hs_hp_imag_history = deque(maxlen=self.history_length)
        
        # Filtered history data
        self.magnitude_filtered_history = deque(maxlen=self.history_length)
        self.phase_filtered_history = deque(maxlen=self.history_length)
        self.hs_hp_real_filtered_history = deque(maxlen=self.history_length)
        self.hs_hp_imag_filtered_history = deque(maxlen=self.history_length)
        
        self.start_time = time.time()
        
        # System Transfer Functions
        self.setup_transfer_functions()
        
        # UI Setup
        self.setup_ui()
        
        # Hardware
        self.hdwf = None
        self.nSamples = None
        self.rgdWindow = None
        self.vNEBW = c_double()
        
        # Start
        self.initialize_hardware()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_measurement)
        self.timer.start(50)  # Update every 50ms (20 Hz)
    
    def setup_ui(self):
        # Main widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Frequency selection
        freq_layout = QtWidgets.QVBoxLayout()
        freq_label = QtWidgets.QLabel("Frequency (Hz):")
        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(100, 1000000)
        self.freq_input.setValue(self.frequency)
        self.freq_input.setSingleStep(100)
        self.freq_input.valueChanged.connect(self.update_parameters)
        
        self.freq_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.freq_slider.setRange(0, 1000)  # Logarithmic mapping
        self.freq_slider.setValue(self.log_map_to_slider(self.frequency))
        self.freq_slider.valueChanged.connect(self.slider_changed)
        
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        freq_layout.addWidget(self.freq_slider)
        
        # History length
        history_layout = QtWidgets.QVBoxLayout()
        history_label = QtWidgets.QLabel("History Length (samples):")
        self.history_input = QtWidgets.QSpinBox()
        self.history_input.setRange(10, 1000)
        self.history_input.setValue(self.history_length)
        self.history_input.valueChanged.connect(self.update_history_length)
        history_layout.addWidget(history_label)
        history_layout.addWidget(self.history_input)
        
        # EMA Filter Controls
        ema_layout = QtWidgets.QVBoxLayout()
        ema_label = QtWidgets.QLabel("EMA Filter:")
        ema_controls = QtWidgets.QHBoxLayout()
        
        self.ema_checkbox = QtWidgets.QCheckBox("Enable")
        self.ema_checkbox.setChecked(self.ema_enabled)
        self.ema_checkbox.stateChanged.connect(self.toggle_ema)
        
        alpha_label = QtWidgets.QLabel("Alpha:")
        self.alpha_input = QtWidgets.QDoubleSpinBox()
        self.alpha_input.setRange(0.01, 1.0)
        self.alpha_input.setValue(self.ema_alpha)
        self.alpha_input.setSingleStep(0.05)
        self.alpha_input.setDecimals(2)
        self.alpha_input.valueChanged.connect(self.update_ema_alpha)
        
        # Add a label explaining alpha
        alpha_help = QtWidgets.QLabel("(Lower = smoother)")
        alpha_help.setStyleSheet("color: gray; font-size: 8pt;")
        
        ema_controls.addWidget(self.ema_checkbox)
        ema_controls.addWidget(alpha_label)
        ema_controls.addWidget(self.alpha_input)
        
        ema_layout.addWidget(ema_label)
        ema_layout.addLayout(ema_controls)
        ema_layout.addWidget(alpha_help)
        
        # Current values display
        metrics_layout = QtWidgets.QVBoxLayout()
        self.current_mag_label = QtWidgets.QLabel("Magnitude: 0 dB")
        self.current_phase_label = QtWidgets.QLabel("Phase: 0°")
        self.current_hs_hp_label = QtWidgets.QLabel("hs/hp: 0 + 0j")
        metrics_layout.addWidget(self.current_mag_label)
        metrics_layout.addWidget(self.current_phase_label)
        metrics_layout.addWidget(self.current_hs_hp_label)
        
        # Add all controls to panel
        control_panel.addLayout(freq_layout)
        control_panel.addLayout(history_layout)
        control_panel.addLayout(ema_layout)
        control_panel.addLayout(metrics_layout)
        control_panel.addStretch(1)
        
        # Reset button
        reset_button = QtWidgets.QPushButton("Reset History")
        reset_button.clicked.connect(self.reset_history)
        control_panel.addWidget(reset_button)
        
        # Add control panel to main layout
        main_layout.addLayout(control_panel)
        
        # Create plots
        plots_layout = QtWidgets.QVBoxLayout()
        
        # Magnitude plot
        self.magnitude_plot = pg.PlotWidget(title="Magnitude Over Time")
        self.magnitude_plot.setLabel('left', 'Magnitude', units='dB')
        self.magnitude_plot.setLabel('bottom', 'Time', units='s')
        self.magnitude_plot.showGrid(x=True, y=True)
        self.magnitude_curve = self.magnitude_plot.plot(pen='b', name='Raw')
        self.magnitude_filtered_curve = self.magnitude_plot.plot(pen=pg.mkPen('c', width=2), name='Filtered')
        self.magnitude_plot.addLegend()
        
        # Phase plot
        self.phase_plot = pg.PlotWidget(title="Phase Over Time")
        self.phase_plot.setLabel('left', 'Phase', units='degrees')
        self.phase_plot.setLabel('bottom', 'Time', units='s')
        self.phase_plot.showGrid(x=True, y=True)
        self.phase_curve = self.phase_plot.plot(pen='g', name='Raw')
        self.phase_filtered_curve = self.phase_plot.plot(pen=pg.mkPen('y', width=2), name='Filtered')
        self.phase_plot.addLegend()
        
        # hs/hp plot
        self.hs_hp_plot = pg.PlotWidget(title="hs/hp Over Time")
        self.hs_hp_plot.setLabel('left', 'Value')
        self.hs_hp_plot.setLabel('bottom', 'Time', units='s')
        self.hs_hp_plot.showGrid(x=True, y=True)
        self.hs_hp_real_curve = self.hs_hp_plot.plot(pen='b', name='Real (Raw)')
        self.hs_hp_imag_curve = self.hs_hp_plot.plot(pen='r', name='Imaginary (Raw)')
        self.hs_hp_real_filtered_curve = self.hs_hp_plot.plot(pen=pg.mkPen('c', width=2), name='Real (Filtered)')
        self.hs_hp_imag_filtered_curve = self.hs_hp_plot.plot(pen=pg.mkPen('m', width=2), name='Imaginary (Filtered)')
        self.hs_hp_plot.addLegend()
        
        # Add plots to layout
        plots_layout.addWidget(self.magnitude_plot)
        plots_layout.addWidget(self.phase_plot)
        plots_layout.addWidget(self.hs_hp_plot)
        
        # Add plots to main layout
        main_layout.addLayout(plots_layout)
    
    def log_map_to_slider(self, freq):
        """Map frequency to logarithmic slider position"""
        min_freq = 100
        max_freq = 1000000
        slider_val = (np.log10(freq) - np.log10(min_freq)) / (np.log10(max_freq) - np.log10(min_freq)) * 1000
        return int(slider_val)
    
    def slider_to_log_map(self, slider_val):
        """Map slider position to frequency (logarithmic)"""
        min_freq = 100
        max_freq = 1000000
        freq = 10 ** (slider_val / 1000 * (np.log10(max_freq) - np.log10(min_freq)) + np.log10(min_freq))
        return freq
    
    def slider_changed(self):
        """Update frequency when slider is moved"""
        freq = self.slider_to_log_map(self.freq_slider.value())
        self.freq_input.setValue(freq)
    
    def setup_transfer_functions(self):
        # Load primary response
        try:
            path = "../matlab/primary_curve_fit_python2.mat"
            mat = scipy.io.loadmat(path)
            num = mat['num'].tolist()[0]
            den = mat['den'].tolist()[0]
            self.sys = control.TransferFunction(num, den)
            
            print("Transfer functions loaded successfully")
        except Exception as e:
            print(f"Error loading transfer functions: {e}")
            self.sys = control.TransferFunction([1], [1])
    
    def initialize_hardware(self, max_retries=5, retry_delay=1):
        """Try to initialize hardware with retries"""
        for attempt in range(max_retries):
            version = create_string_buffer(16)
            dwf.FDwfGetVersion(version)
            print(f"Attempt {attempt + 1}/{max_retries}: DWF Version: {str(version.value)}")

            self.hdwf = c_int()
            szerr = create_string_buffer(512)
            print("Opening first device")
            if dwf.FDwfDeviceOpen(-1, byref(self.hdwf)) == 1 and self.hdwf.value != hdwfNone.value:
                print("Successfully connected to device")
                
                # Device configuration
                dwf.FDwfParamSet(DwfParamOnClose, c_int(0))
                dwf.FDwfDeviceAutoConfigureSet(self.hdwf, c_int(0))

                # Configure AWG
                dwf.FDwfAnalogOutNodeEnableSet(self.hdwf, c_int(0), AnalogOutNodeCarrier, c_int(1))
                dwf.FDwfAnalogOutNodeFunctionSet(self.hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
                dwf.FDwfAnalogOutNodeAmplitudeSet(self.hdwf, c_int(0), AnalogOutNodeCarrier, c_double(0.5))
                dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1))

                # Configure Scope
                self.nSamples = 2**16
                dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(20000000.0))
                dwf.FDwfAnalogInBufferSizeSet(self.hdwf, self.nSamples)
                dwf.FDwfAnalogInChannelEnableSet(self.hdwf, 0, c_int(1))
                dwf.FDwfAnalogInChannelRangeSet(self.hdwf, 0, c_double(2))
                dwf.FDwfAnalogInChannelEnableSet(self.hdwf, 1, c_int(1))
                dwf.FDwfAnalogInChannelRangeSet(self.hdwf, 1, c_double(2))
                
                # Setup FFT window
                self.rgdWindow = (c_double * self.nSamples)()
                vBeta = c_double(1.0)
                dwf.FDwfSpectrumWindow(byref(self.rgdWindow), c_int(self.nSamples), DwfWindowFlatTop, vBeta, byref(self.vNEBW))
                
                # Set initial frequency
                self.update_parameters()
                
                return True
            
            dwf.FDwfGetLastErrorMsg(szerr)
            print(f"Attempt {attempt + 1} failed: {szerr.value}")
            
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return False
    
    def update_parameters(self):
        """Update frequency from UI input"""
        self.frequency = self.freq_input.value()
        
        # Update frequency on the device
        if self.hdwf is not None:
            dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_int(0), AnalogOutNodeCarrier, c_double(self.frequency))
            dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1))
    
    def update_history_length(self):
        """Update the history length"""
        new_length = self.history_input.value()
        
        # Save current data
        mag_list = list(self.magnitude_history)
        phase_list = list(self.phase_history)
        hs_hp_real_list = list(self.hs_hp_real_history)
        hs_hp_imag_list = list(self.hs_hp_imag_history)
        time_list = list(self.timestamps)
        
        # Create new deques with new length
        self.magnitude_history = deque(maxlen=new_length)
        self.phase_history = deque(maxlen=new_length)
        self.hs_hp_real_history = deque(maxlen=new_length)
        self.hs_hp_imag_history = deque(maxlen=new_length)
        self.timestamps = deque(maxlen=new_length)
        
        # Copy data back (will be truncated if new length is smaller)
        self.magnitude_history.extend(mag_list[-new_length:] if len(mag_list) > new_length else mag_list)
        self.phase_history.extend(phase_list[-new_length:] if len(phase_list) > new_length else phase_list)
        self.hs_hp_real_history.extend(hs_hp_real_list[-new_length:] if len(hs_hp_real_list) > new_length else hs_hp_real_list)
        self.hs_hp_imag_history.extend(hs_hp_imag_list[-new_length:] if len(hs_hp_imag_list) > new_length else hs_hp_imag_list)
        self.timestamps.extend(time_list[-new_length:] if len(time_list) > new_length else time_list)
        
        self.history_length = new_length
    
    def toggle_ema(self, state):
        """Toggle EMA filtering on/off"""
        self.ema_enabled = (state == QtCore.Qt.Checked)
        
        # Reset EMA state if disabled and re-enabled
        if self.ema_enabled:
            self.ema_magnitude = None
            self.ema_phase = None
            self.ema_hs_hp_real = None
            self.ema_hs_hp_imag = None
    
    def update_ema_alpha(self, value):
        """Update EMA alpha value"""
        self.ema_alpha = value
        
        # Reset EMA state when alpha changes
        self.ema_magnitude = None
        self.ema_phase = None
        self.ema_hs_hp_real = None
        self.ema_hs_hp_imag = None
    
    def apply_ema(self, new_value, current_ema):
        """Apply EMA filter to a value"""
        if current_ema is None:
            return new_value
        return self.ema_alpha * new_value + (1 - self.ema_alpha) * current_ema
    
    def reset_history(self):
        """Clear all history data"""
        self.magnitude_history.clear()
        self.phase_history.clear()
        self.hs_hp_real_history.clear()
        self.hs_hp_imag_history.clear()
        self.magnitude_filtered_history.clear()
        self.phase_filtered_history.clear()
        self.hs_hp_real_filtered_history.clear()
        self.hs_hp_imag_filtered_history.clear()
        self.timestamps.clear()
        
        # Reset EMA state
        self.ema_magnitude = None
        self.ema_phase = None
        self.ema_hs_hp_real = None
        self.ema_hs_hp_imag = None
        
        self.start_time = time.time()
    
    def update_measurement(self):
        """Perform a single measurement and update the display"""
        if self.hdwf is None:
            return
        
        # Acquire data
        dwf.FDwfAnalogInConfigure(self.hdwf, c_int(1), c_int(1))
        
        # Wait for acquisition
        while True:
            sts = c_byte()
            dwf.FDwfAnalogInStatus(self.hdwf, c_int(1), byref(sts))
            if sts.value == DwfStateDone.value:
                break
            QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
        # Get data
        rgdSamples1 = (c_double * self.nSamples)()
        rgdSamples2 = (c_double * self.nSamples)()
        dwf.FDwfAnalogInStatusData(self.hdwf, 0, rgdSamples1, self.nSamples)
        dwf.FDwfAnalogInStatusData(self.hdwf, 1, rgdSamples2, self.nSamples)
        
        # Process data
        def process_data(samples):
            for i in range(self.nSamples):
                samples[i] = samples[i] * self.rgdWindow[i]
            nBins = self.nSamples // 2 + 1
            rgdBins = (c_double * nBins)()
            rgdPhase = (c_double * nBins)()
            dwf.FDwfSpectrumFFT(byref(samples), self.nSamples, byref(rgdBins), byref(rgdPhase), nBins)
            fIndex = int(self.frequency / 20000000 * self.nSamples) + 1
            return rgdBins[fIndex], rgdPhase[fIndex]
        
        c1_mag, c1_phase = process_data(rgdSamples1)
        c2_mag, c2_phase = process_data(rgdSamples2)
        
        # Store results if valid
        if c1_mag > 0:
            c1_mag *= 1  # 10x probe attenuation
            h_db = 20 * math.log10(c2_mag * 20)
            h_linear = c2_mag * 20
            phase_diff = (c2_phase - c1_phase) * 180 / math.pi
            phase_diff = (phase_diff + 180) % 360 - 180
            
            # Get primary response at current frequency
            w = 2 * np.pi * self.frequency
            h_mag_prim, h_phase_prim, _ = bode(self.sys, [w], plot=False)
            h_mag_prim_lin = h_mag_prim[0]
            h_phase_prim_deg = np.degrees(h_phase_prim[0])
            
            # Calculate hs/hp
            h_prim_complex = h_mag_prim_lin * np.exp(1j * np.radians(h_phase_prim_deg))
            h_complex = h_linear * np.exp(1j * np.radians(phase_diff))
            hs_hp = h_complex / h_prim_complex
            
            # Apply EMA filtering if enabled
            if self.ema_enabled:
                # Apply EMA filter
                self.ema_magnitude = self.apply_ema(h_db, self.ema_magnitude)
                self.ema_phase = self.apply_ema(phase_diff, self.ema_phase)
                self.ema_hs_hp_real = self.apply_ema(np.real(hs_hp), self.ema_hs_hp_real)
                self.ema_hs_hp_imag = self.apply_ema(np.imag(hs_hp), self.ema_hs_hp_imag)
                
                # Store filtered values for display
                filtered_mag = self.ema_magnitude
                filtered_phase = self.ema_phase
                filtered_real = self.ema_hs_hp_real
                filtered_imag = self.ema_hs_hp_imag
            else:
                # Use raw values if filtering is disabled
                filtered_mag = h_db
                filtered_phase = phase_diff
                filtered_real = np.real(hs_hp)
                filtered_imag = np.imag(hs_hp)
            
            # Store in history
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # Store raw values
            self.magnitude_history.append(h_db)
            self.phase_history.append(phase_diff)
            self.hs_hp_real_history.append(np.real(hs_hp))
            self.hs_hp_imag_history.append(np.imag(hs_hp))
            
            # Store filtered values
            self.magnitude_filtered_history.append(filtered_mag)
            self.phase_filtered_history.append(filtered_phase)
            self.hs_hp_real_filtered_history.append(filtered_real)
            self.hs_hp_imag_filtered_history.append(filtered_imag)
            
            # Update metrics display - use filtered values if EMA is enabled
            display_mag = filtered_mag if self.ema_enabled else h_db
            display_phase = filtered_phase if self.ema_enabled else phase_diff
            display_real = filtered_real if self.ema_enabled else np.real(hs_hp)
            display_imag = filtered_imag if self.ema_enabled else np.imag(hs_hp)
            
            self.current_mag_label.setText(f"Magnitude: {display_mag:.2f} dB")
            self.current_phase_label.setText(f"Phase: {display_phase:.2f}°")
            self.current_hs_hp_label.setText(f"hs/hp: {display_real:.3f} + {display_imag:.3f}j")
            
            # Update plots
            self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data"""
        if len(self.timestamps) < 2:
            return
        
        timestamps = list(self.timestamps)
        
        # Update magnitude plot
        self.magnitude_curve.setData(timestamps, list(self.magnitude_history))
        self.magnitude_filtered_curve.setData(timestamps, list(self.magnitude_filtered_history))
        
        # Update phase plot
        self.phase_curve.setData(timestamps, list(self.phase_history))
        self.phase_filtered_curve.setData(timestamps, list(self.phase_filtered_history))
        
        # Update hs/hp plot
        self.hs_hp_real_curve.setData(timestamps, list(self.hs_hp_real_history))
        self.hs_hp_imag_curve.setData(timestamps, list(self.hs_hp_imag_history))
        self.hs_hp_real_filtered_curve.setData(timestamps, list(self.hs_hp_real_filtered_history))
        self.hs_hp_imag_filtered_curve.setData(timestamps, list(self.hs_hp_imag_filtered_history))
    
    def closeEvent(self, event):
        """Clean up hardware when closing the application"""
        if self.hdwf is not None:
            dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(0))
            dwf.FDwfDeviceCloseAll()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set dark theme
    dark_palette = QtGui.QPalette()
    dark_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
    dark_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    dark_palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    dark_palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    dark_palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    dark_palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(dark_palette)
    
    # Set pyqtgraph configuration
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    analyzer = SingleFrequencyAnalyzerApp()
    analyzer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()