from ctypes import *
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# Platform-specific DWF library loading
if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

# Constants from dwfconstants.py
hdwfNone = c_int(0)
DwfParamOnClose = c_int(2)
AnalogOutNodeCarrier = c_int(0)
funcSine = c_int(1)
DwfStateDone = c_byte(2)
DwfWindowFlatTop = c_int(4)

class FrequencyTraceAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Full Frequency Trace Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        # Parameters
        self.sample_rate = 1000000.0  # 10 MHz
        self.nSamples = 2**16
        
        # UI Setup
        self.setup_ui()
        
        # Hardware
        self.hdwf = None
        self.rgdWindow = None
        self.vNEBW = c_double()
        
        # FFT bins
        self.freq_bins = np.linspace(0, self.sample_rate/2, self.nSamples//2 + 1)
        self.ch1_fft_mag = np.zeros(self.nSamples//2 + 1)
        self.ch2_fft_mag = np.zeros(self.nSamples//2 + 1)
        self.ch1_fft_phase = np.zeros(self.nSamples//2 + 1)
        self.ch2_fft_phase = np.zeros(self.nSamples//2 + 1)
        
        # Start
        if self.initialize_hardware():
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_measurement)
            self.timer.start(100)  # Update every 100ms (10 Hz)
        else:
            self.show_error_message("Failed to initialize hardware")
    
    def setup_ui(self):
        # Main widget
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Frequency control
        freq_layout = QtWidgets.QHBoxLayout()
        freq_label = QtWidgets.QLabel("Stimulus Frequency (Hz):")
        self.freq_input = QtWidgets.QDoubleSpinBox()
        self.freq_input.setRange(100, 1000000)
        self.freq_input.setValue(10000)  # Default 10kHz
        self.freq_input.setSingleStep(1000)
        self.freq_input.valueChanged.connect(self.update_stimulus_frequency)
        freq_layout.addWidget(freq_label)
        freq_layout.addWidget(self.freq_input)
        
        # Add all controls to panel
        control_panel.addLayout(freq_layout)
        control_panel.addStretch(1)
        
        # Add control panel to main layout
        main_layout.addLayout(control_panel)
        
        # Create plots
        plots_layout = QtWidgets.QVBoxLayout()
        
        # Channel 1 FFT plot
        self.ch1_fft_plot = pg.PlotWidget(title="Channel 1 - Frequency Domain")
        self.ch1_fft_plot.setLabel('left', 'Magnitude', units='dB')
        self.ch1_fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.ch1_fft_plot.setLogMode(x=True, y=False)
        self.ch1_fft_plot.showGrid(x=True, y=True)
        self.ch1_fft_curve = self.ch1_fft_plot.plot(pen='y', name='Channel 1')
        
        # Channel 2 FFT plot
        self.ch2_fft_plot = pg.PlotWidget(title="Channel 2 - Frequency Domain")
        self.ch2_fft_plot.setLabel('left', 'Magnitude', units='dB')
        self.ch2_fft_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.ch2_fft_plot.setLogMode(x=True, y=False)
        self.ch2_fft_plot.showGrid(x=True, y=True)
        self.ch2_fft_curve = self.ch2_fft_plot.plot(pen='c', name='Channel 2')
        
        # Transfer Function plot
        self.tf_plot = pg.PlotWidget(title="Transfer Function - Magnitude")
        self.tf_plot.setLabel('left', 'Magnitude', units='dB')
        self.tf_plot.setLabel('bottom', 'Frequency', units='Hz')
        self.tf_plot.setLogMode(x=True, y=False)
        self.tf_plot.showGrid(x=True, y=True)
        self.tf_curve = self.tf_plot.plot(pen='g', name='Transfer Function')
        
        # Add plots to layout
        plots_layout.addWidget(self.ch1_fft_plot)
        plots_layout.addWidget(self.ch2_fft_plot)
        plots_layout.addWidget(self.tf_plot)
        
        # Add plots to main layout
        main_layout.addLayout(plots_layout)
    
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
                dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_int(0), AnalogOutNodeCarrier, c_double(10000))
                dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1))

                # Configure Scope
                dwf.FDwfAnalogInFrequencySet(self.hdwf, c_double(self.sample_rate))
                dwf.FDwfAnalogInBufferSizeSet(self.hdwf, self.nSamples)
                dwf.FDwfAnalogInChannelEnableSet(self.hdwf, 0, c_int(1))
                dwf.FDwfAnalogInChannelRangeSet(self.hdwf, 0, c_double(2))
                dwf.FDwfAnalogInChannelEnableSet(self.hdwf, 1, c_int(1))
                dwf.FDwfAnalogInChannelRangeSet(self.hdwf, 1, c_double(2))
                
                # Setup FFT window
                self.rgdWindow = (c_double * self.nSamples)()
                vBeta = c_double(1.0)
                dwf.FDwfSpectrumWindow(byref(self.rgdWindow), c_int(self.nSamples), DwfWindowFlatTop, vBeta, byref(self.vNEBW))
                
                return True
            
            dwf.FDwfGetLastErrorMsg(szerr)
            print(f"Attempt {attempt + 1} failed: {szerr.value}")
            
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        return False
    
    def update_stimulus_frequency(self):
        """Update the AWG frequency"""
        freq = self.freq_input.value()
        if self.hdwf is not None:
            dwf.FDwfAnalogOutNodeFrequencySet(self.hdwf, c_int(0), AnalogOutNodeCarrier, c_double(freq))
            dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(1))
            print(f"Stimulus frequency set to {freq} Hz")
    
    def update_measurement(self):
        """Perform a full frequency trace measurement and update the display"""
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
            time.sleep(0.001)
            QtWidgets.QApplication.processEvents()  # Keep UI responsive
        
        # Get data
        rgdSamples1 = (c_double * self.nSamples)()
        rgdSamples2 = (c_double * self.nSamples)()
        dwf.FDwfAnalogInStatusData(self.hdwf, 0, rgdSamples1, self.nSamples)
        dwf.FDwfAnalogInStatusData(self.hdwf, 1, rgdSamples2, self.nSamples)
        
        # Process data
        def process_data(samples):
            # Apply window function
            windowed_samples = (c_double * self.nSamples)()
            for i in range(self.nSamples):
                windowed_samples[i] = samples[i] * self.rgdWindow[i]
            
            # Calculate FFT
            nBins = self.nSamples // 2 + 1
            rgdBins = (c_double * nBins)()
            rgdPhase = (c_double * nBins)()
            dwf.FDwfSpectrumFFT(byref(windowed_samples), self.nSamples, byref(rgdBins), byref(rgdPhase), nBins)
            
            # Convert to numpy arrays
            bins = np.array([rgdBins[i] for i in range(nBins)])
            phase = np.array([rgdPhase[i] for i in range(nBins)])
            
            return bins, phase
        
        # Get FFT for both channels
        self.ch1_fft_mag, self.ch1_fft_phase = process_data(rgdSamples1)
        self.ch2_fft_mag, self.ch2_fft_phase = process_data(rgdSamples2)
        
        # Convert magnitude to dB, avoiding log(0)
        epsilon = 1e-10
        ch1_fft_db = 20 * np.log10(self.ch1_fft_mag + epsilon)
        ch2_fft_db = 20 * np.log10(self.ch2_fft_mag + epsilon)
        
        # Calculate transfer function (CH2/CH1)
        tf_mag = self.ch2_fft_mag / (self.ch1_fft_mag + epsilon)
        tf_db = 20 * np.log10(tf_mag + epsilon)
        
        # Remove DC and very high frequencies for better visualization
        # Only show from ~100Hz to ~sample_rate/2.5
        start_idx = max(1, int(100 / (self.sample_rate/2) * len(self.freq_bins)))
        end_idx = int(len(self.freq_bins) * 0.8)  # Show 80% of spectrum
        
        # Update plots
        self.ch1_fft_curve.setData(self.freq_bins[start_idx:end_idx], ch1_fft_db[start_idx:end_idx])
        self.ch2_fft_curve.setData(self.freq_bins[start_idx:end_idx], ch2_fft_db[start_idx:end_idx])
        self.tf_curve.setData(self.freq_bins[start_idx:end_idx], tf_db[start_idx:end_idx])
    
    def show_error_message(self, message):
        """Display an error message box"""
        msg_box = QtWidgets.QMessageBox()
        msg_box.setIcon(QtWidgets.QMessageBox.Critical)
        msg_box.setText(message)
        msg_box.setWindowTitle("Error")
        msg_box.exec_()
    
    def closeEvent(self, event):
        """Clean up hardware when closing the application"""
        if self.hdwf is not None:
            dwf.FDwfAnalogOutConfigure(self.hdwf, c_int(0), c_int(0))
            dwf.FDwfDeviceCloseAll()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set dark theme for PyQtGraph
    pg.setConfigOption('background', 'k')
    pg.setConfigOption('foreground', 'w')
    
    analyzer = FrequencyTraceAnalyzer()
    analyzer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()