from ctypes import *
import math
import time
import numpy
import scipy
import scipy.io
import control
from control.matlab import ss, bode
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import numpy as np
from WaveformInterface import AnalogDiscovery

class SignalAnalyzerApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Transfer Function Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Parameters
        self.start_freq = 1000.0
        self.stop_freq = 100000.0
        self.steps = 16
        self.frequencies = numpy.logspace(numpy.log10(self.start_freq), numpy.log10(self.stop_freq), num=int(self.steps))
        
        self.sample_rate = 1000000.0
        self.buffer_size = 2**16

        # System Transfer Functions
        self.setup_transfer_functions()
        
        # UI Setup
        self.setup_ui()
        
        # Hardware
        self.device = None
        
        # Data
        self.current_freq_idx = 0
        self.h_mag = []
        self.h_mag_lin = []
        self.h_phase = []
        self.frequencies_plot = []
        
        # Start
        self.initialize_hardware()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_measurement)
        self.timer.start(10)  # Update every 10ms (100 Hz)
    
    def setup_ui(self):
        # Main widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Start frequency
        start_freq_layout = QtWidgets.QVBoxLayout()
        start_freq_label = QtWidgets.QLabel("Start Frequency (Hz):")
        self.start_freq_input = QtWidgets.QDoubleSpinBox()
        self.start_freq_input.setRange(100, 10000)
        self.start_freq_input.setValue(self.start_freq)
        self.start_freq_input.valueChanged.connect(self.update_parameters)
        start_freq_layout.addWidget(start_freq_label)
        start_freq_layout.addWidget(self.start_freq_input)
        
        # Stop frequency
        stop_freq_layout = QtWidgets.QVBoxLayout()
        stop_freq_label = QtWidgets.QLabel("Stop Frequency (Hz):")
        self.stop_freq_input = QtWidgets.QDoubleSpinBox()
        self.stop_freq_input.setRange(1000, 1000000)
        self.stop_freq_input.setValue(self.stop_freq)
        self.stop_freq_input.valueChanged.connect(self.update_parameters)
        stop_freq_layout.addWidget(stop_freq_label)
        stop_freq_layout.addWidget(self.stop_freq_input)
        
        # Steps
        steps_layout = QtWidgets.QVBoxLayout()
        steps_label = QtWidgets.QLabel("Number of Steps:")
        self.steps_input = QtWidgets.QSpinBox()
        self.steps_input.setRange(4, 1000)
        self.steps_input.setValue(self.steps)
        self.steps_input.valueChanged.connect(self.update_parameters)
        steps_layout.addWidget(steps_label)
        steps_layout.addWidget(self.steps_input)
        
        # Current frequency and magnitude display
        metrics_layout = QtWidgets.QVBoxLayout()
        self.current_freq_label = QtWidgets.QLabel("Current Frequency: 0 kHz")
        self.current_mag_label = QtWidgets.QLabel("Latest Magnitude: 0 dB")
        metrics_layout.addWidget(self.current_freq_label)
        metrics_layout.addWidget(self.current_mag_label)
        
        # Add all controls to panel
        control_panel.addLayout(start_freq_layout)
        control_panel.addLayout(stop_freq_layout)
        control_panel.addLayout(steps_layout)
        control_panel.addLayout(metrics_layout)
        control_panel.addStretch(1)
        
        # Add control panel to main layout
        main_layout.addLayout(control_panel)
        
        # Create plots
        plots_layout = QtWidgets.QVBoxLayout()
        
        # Magnitude plot
        self.magnitude_plot = pg.PlotWidget(title="Transfer Function - Magnitude")
        self.magnitude_plot.setLabel('left', 'Magnitude', units='dB')
        self.magnitude_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.magnitude_plot.setLogMode(x=True, y=False)
        self.magnitude_plot.showGrid(x=True, y=True)
        self.measured_mag_curve = self.magnitude_plot.plot(pen='b', name='Measured')
        self.primary_mag_curve = self.magnitude_plot.plot(pen='r', name='Primary')
        self.magnitude_plot.addLegend()
        
        # Phase plot
        self.phase_plot = pg.PlotWidget(title="Transfer Function - Phase")
        self.phase_plot.setLabel('left', 'Phase', units='degrees')
        self.phase_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.phase_plot.setLogMode(x=True, y=False)
        self.phase_plot.showGrid(x=True, y=True)
        self.measured_phase_curve = self.phase_plot.plot(pen='b', name='Measured')
        self.primary_phase_curve = self.phase_plot.plot(pen='r', name='Primary')
        self.phase_plot.addLegend()
        
        # hs/hp plot
        self.hs_hp_plot = pg.PlotWidget(title="hs/hp")
        self.hs_hp_plot.setLabel('left', 'hs/hp')
        self.hs_hp_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.hs_hp_plot.setLogMode(x=True, y=False)
        self.hs_hp_plot.showGrid(x=True, y=True)
        self.hs_hp_real_curve = self.hs_hp_plot.plot(pen='b', name='Real')
        self.hs_hp_imag_curve = self.hs_hp_plot.plot(pen='r', name='Imaginary')
        self.hs_hp_plot.addLegend()
        
        # Add plots to layout
        plots_layout.addWidget(self.magnitude_plot)
        plots_layout.addWidget(self.phase_plot)
        plots_layout.addWidget(self.hs_hp_plot)
        
        # Add plots to main layout
        main_layout.addLayout(plots_layout)
    
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
            self.sys = control.TransferFunction([1], [1])
    
    def initialize_hardware(self, max_retries=5, retry_delay=1):
        """Initialize hardware using AnalogDiscovery class"""
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries}: Initializing AnalogDiscovery")
                
                # Create and initialize device
                self.device = AnalogDiscovery()
                self.device.init()
                
                # Configure scope
                self.device.init_scope(
                    sampling_frequency=self.sample_rate,
                    buffer_size=self.buffer_size,
                    offset=0,
                    amplitude_range=2
                )
                
                # Configure AWG - start with a sine wave at the first frequency
                self.device.generate(
                    channel=1,
                    function=self.device.constants.funcSine,
                    offset=0,
                    frequency=self.frequencies[0],
                    amplitude=0.5
                )
                
                print("Successfully connected to device")
                return True
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                
                # Clean up if device was partially initialized
                if self.device is not None:
                    try:
                        self.device.close()
                    except:
                        pass
                    self.device = None
                
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        
        print("Failed to initialize hardware after multiple attempts")
        return False
    
    def update_parameters(self):
        """Update sweep parameters from UI inputs"""
        self.start_freq = self.start_freq_input.value()
        self.stop_freq = self.stop_freq_input.value()
        self.steps = self.steps_input.value()
        self.frequencies = numpy.logspace(numpy.log10(self.start_freq), numpy.log10(self.stop_freq), num=int(self.steps))
        
        # Reset measurements
        self.current_freq_idx = 0
        self.h_mag = []
        self.h_mag_lin = []
        self.h_phase = []
        self.frequencies_plot = []
    
    def update_measurement(self):
        """Perform a single frequency measurement and update the display"""
        if self.device is None or self.current_freq_idx >= len(self.frequencies):
            self.current_freq_idx = 0
            self.h_mag = []
            self.h_mag_lin = []
            self.h_phase = []
            self.frequencies_plot = []
        
        # Get current frequency
        freq = self.frequencies[self.current_freq_idx]
        print(f"\rFrequency: {freq/1e3:.1f} kHz", end='')
        
        try:
            # Set frequency and generate signal
            self.device.generate(
                channel=1,
                function=self.device.constants.funcSine,
                offset=0,
                frequency=freq,
                amplitude=1.0
            )
            
            # Allow signal to stabilize
            time.sleep(0.01)
            
            # Perform FFT analysis on both channels
            freq1, mag1, phase1, raw1 = self.device.fft_analyze(channel=1, wait_for_stable=True)
            freq2, mag2, phase2, raw2 = self.device.fft_analyze(channel=2, wait_for_stable=True)
            
            # Find the index closest to our test frequency
            freq_idx1 = min(range(len(freq1)), key=lambda i: abs(freq1[i] - freq))
            freq_idx2 = min(range(len(freq2)), key=lambda i: abs(freq2[i] - freq))
            
            # Get magnitude and phase at the test frequency
            c1_mag = 10**(mag1[freq_idx1]/20)  # Convert from dB to linear
            c1_phase = phase1[freq_idx1]
            c2_mag = 10**(mag2[freq_idx2]/20)  # Convert from dB to linear
            c2_phase = phase2[freq_idx2]
            
            # Apply probe attenuation if needed (10x probe)
            c1_mag *= 1  # Adjust if using attenuating probes
            
            # Calculate transfer function
            if c1_mag > 0:
                h_linear = c2_mag * 20
                h_db = 20 * math.log10(h_linear)
                phase_diff = c2_phase - c1_phase
                phase_diff = ((phase_diff) % 360) - 360 # Wrap to [-180, 180]
                
                self.frequencies_plot.append(freq)
                self.h_mag.append(h_db)
                self.h_mag_lin.append(h_linear)
                self.h_phase.append(phase_diff)
                
                # Update metrics display
                self.current_freq_label.setText(f"Current Frequency: {freq/1e3:.1f} kHz")
                self.current_mag_label.setText(f"Latest Magnitude: {h_db:.2f} dB")
            
        except Exception as e:
            print(f"\nError during measurement at {freq} Hz: {e}")
        
        # Process UI events to keep the interface responsive
        QtWidgets.QApplication.processEvents()
        
        # Move to next frequency
        self.current_freq_idx += 1
        
        # If we've completed a sweep, update the plots
        if self.current_freq_idx >= len(self.frequencies):
            self.update_plots()
    
    def update_plots(self):
        """Update all plots with current data"""
        if len(self.frequencies_plot) < 2:
            return
        
        # Calculate primary response
        w = 2 * numpy.pi * numpy.array(self.frequencies_plot)
        h_mag_prim, h_phase_prim, omega = bode(self.sys, w, plot=False)
        h_mag_prim_lin = h_mag_prim
        h_mag_prim_db = 20 * numpy.log10(h_mag_prim)
        h_phase_prim_deg = numpy.degrees(h_phase_prim)
        
        # Calculate hs/hp
        h_mag_lin_array = numpy.array(self.h_mag_lin)
        h_phase_array = numpy.array(self.h_phase)
        h_prim_complex = h_mag_prim_lin * numpy.exp(1j * numpy.radians(h_phase_prim_deg))
        h_complex = h_mag_lin_array * numpy.exp(1j * numpy.radians(h_phase_array))
        hs_hp = h_complex / h_prim_complex
        
        # Update magnitude plot
        self.measured_mag_curve.setData(self.frequencies_plot, self.h_mag)
        self.primary_mag_curve.setData(self.frequencies_plot, h_mag_prim_db)
        
        # Update phase plot
        self.measured_phase_curve.setData(self.frequencies_plot, self.h_phase)
        self.primary_phase_curve.setData(self.frequencies_plot, h_phase_prim_deg)
        
        # Update hs/hp plot
        self.hs_hp_real_curve.setData(self.frequencies_plot, numpy.real(hs_hp))
        self.hs_hp_imag_curve.setData(self.frequencies_plot, numpy.imag(hs_hp))
    
    def closeEvent(self, event):
        """Clean up hardware when closing the application"""
        if self.device is not None:
            try:
                self.device.close()  # This will clean up scope and generator
                self.device = None
            except Exception as e:
                print(f"Error closing device: {e}")
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
    
    analyzer = SignalAnalyzerApp()
    analyzer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()