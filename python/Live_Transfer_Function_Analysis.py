import math
import time
import numpy as np
import scipy
import control
from control.matlab import ss, bode
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from WaveformInterface import AnalogDiscovery

class TransferFunctionAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize the Analog Discovery device
        self.device = AnalogDiscovery()
        self.device.init()
        
        # Configure the device
        self.setupDevice()
        
        # Set up the UI
        self.setupUI()
        
        # Initialize data structures
        self.frequencies = []
        self.h_mag = []
        self.h_phase = []
        self.h_mag_prim = []
        self.h_phase_prim = []
        self.hs_hp_real = []
        self.hs_hp_imag = []
        
        # Start the measurement timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateMeasurement)
        self.timer.start(100)  # Update every 100ms
        
    def setupDevice(self):
        # Initialize the scope with appropriate settings
        self.device.init_scope(sampling_frequency=20e6, buffer_size=2**10, amplitude_range=5)
        
        # Enable both channels
        self.device.dwf.FDwfAnalogInChannelEnableSet(self.device.handle, 0, 1)  # Enable channel 0 (C1)
        self.device.dwf.FDwfAnalogInChannelRangeSet(self.device.handle, 0, 20)  # 20V range
        self.device.dwf.FDwfAnalogInChannelEnableSet(self.device.handle, 1, 1)  # Enable channel 1 (C2)
        self.device.dwf.FDwfAnalogInChannelRangeSet(self.device.handle, 1, 2)   # 2V range
        
    def setupUI(self):
        # Set up the main window
        self.setWindowTitle("Transfer Function Analyzer")
        self.setGeometry(100, 100, 1000, 800)
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Create control panel
        control_panel = QtWidgets.QHBoxLayout()
        
        # Frequency range controls
        freq_group = QtWidgets.QGroupBox("Frequency Range")
        freq_layout = QtWidgets.QFormLayout(freq_group)
        
        self.start_freq_input = QtWidgets.QLineEdit("150")
        self.stop_freq_input = QtWidgets.QLineEdit("100000")
        self.steps_input = QtWidgets.QLineEdit("101")
        
        freq_layout.addRow("Start (Hz):", self.start_freq_input)
        freq_layout.addRow("Stop (Hz):", self.stop_freq_input)
        freq_layout.addRow("Steps:", self.steps_input)
        
        # Amplitude control
        self.amplitude_input = QtWidgets.QLineEdit("0.5")
        freq_layout.addRow("Amplitude (V):", self.amplitude_input)
        
        # Add buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.save_button = QtWidgets.QPushButton("Save Data")
        
        self.start_button.clicked.connect(self.startMeasurement)
        self.stop_button.clicked.connect(self.stopMeasurement)
        self.save_button.clicked.connect(self.saveData)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.save_button)
        
        freq_layout.addRow(button_layout)
        control_panel.addWidget(freq_group)
        
        # Add primary response controls
        prim_group = QtWidgets.QGroupBox("Primary Response")
        prim_layout = QtWidgets.QFormLayout(prim_group)
        
        self.prim_path_input = QtWidgets.QLineEdit("../matlab/primary_curve_fit_python.mat")
        self.load_prim_button = QtWidgets.QPushButton("Load Primary")
        self.load_prim_button.clicked.connect(self.loadPrimaryResponse)
        
        prim_layout.addRow("Path:", self.prim_path_input)
        prim_layout.addWidget(self.load_prim_button)
        
        control_panel.addWidget(prim_group)
        layout.addLayout(control_panel)
        
        # Create plot widgets
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget1.setTitle("Magnitude Response")
        self.plot_widget1.setLabel('left', "Magnitude (dB)")
        self.plot_widget1.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget1.setLogMode(x=True, y=False)
        
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget2.setTitle("Phase Response")
        self.plot_widget2.setLabel('left', "Phase (degrees)")
        self.plot_widget2.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget2.setLogMode(x=True, y=False)
        
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget3.setTitle("hs/hp")
        self.plot_widget3.setLabel('left', "Value")
        self.plot_widget3.setLabel('bottom', "Frequency (Hz)")
        self.plot_widget3.setLogMode(x=True, y=False)
        
        # Add plots to layout
        layout.addWidget(self.plot_widget1)
        layout.addWidget(self.plot_widget2)
        layout.addWidget(self.plot_widget3)
        
        # Create plot data items
        self.mag_plot = self.plot_widget1.plot(pen='r', name="Measured")
        self.mag_prim_plot = self.plot_widget1.plot(pen=pg.mkPen('b', style=QtCore.Qt.DashLine), name="Primary")
        
        self.phase_plot = self.plot_widget2.plot(pen='r', name="Measured")
        self.phase_prim_plot = self.plot_widget2.plot(pen=pg.mkPen('b', style=QtCore.Qt.DashLine), name="Primary")
        
        self.hs_hp_real_plot = self.plot_widget3.plot(pen='g', name="Real")
        self.hs_hp_imag_plot = self.plot_widget3.plot(pen='m', name="Imaginary")
        
        # Add legends
        self.plot_widget1.addLegend()
        self.plot_widget2.addLegend()
        self.plot_widget3.addLegend()
        
    def startMeasurement(self):
        # Get parameters from UI
        start_freq = float(self.start_freq_input.text())
        stop_freq = float(self.stop_freq_input.text())
        steps = int(self.steps_input.text())
        amplitude = float(self.amplitude_input.text())
        
        # Start the measurement
        self.is_measuring = True
        self.frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), num=steps)
        self.current_index = 0
        self.h_mag = []
        self.h_phase = []
        self.h_mag_lin = []
        
        # Update the UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
    def stopMeasurement(self):
        self.is_measuring = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def updateMeasurement(self):
        if not hasattr(self, 'is_measuring') or not self.is_measuring:
            return
            
        if self.current_index >= len(self.frequencies):
            self.is_measuring = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.calculateHsHp()
            return
            
        # Get current frequency
        freq = self.frequencies[self.current_index]
        
        # Set generator frequency
        amplitude = float(self.amplitude_input.text())
        self.device.generate(channel=1, function=self.device.constants.funcSine, 
                            offset=0, frequency=freq, amplitude=amplitude)
        
        # Wait for signal to stabilize
        time.sleep(0.05)
        
        # Measure the response
        try:
            # Use the frequency_response method for a single frequency point
            _, mag, phase = self.device.frequency_response(
                input_channel=1, output_channel=2,
                start_freq=freq, stop_freq=freq, steps=1,
                amplitude=amplitude, probe_attenuation=10.0
            )
            
            # Store the results
            self.h_mag.append(mag[0])
            self.h_phase.append(phase[0])
            self.h_mag_lin.append(10**(mag[0]/20))  # Convert dB to linear
            
            # Update the plots
            self.updatePlots()
            
            # Move to next frequency
            self.current_index += 1
            
        except Exception as e:
            print(f"Error during measurement: {e}")
            self.is_measuring = False
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def loadPrimaryResponse(self):
        try:
            path = self.prim_path_input.text()
            mat = scipy.io.loadmat(path)
            num = mat['num'].tolist()[0][0][0]
            den = mat['den'].tolist()[0][0][0]
            
            sys = control.TransferFunction(num, den)
            
            # Calculate primary response for current frequencies
            if len(self.frequencies) > 0:
                w = 2 * np.pi * self.frequencies[:self.current_index]
                self.h_mag_prim, self.h_phase_prim, _ = bode(sys, w, plot=False)
                
                # Convert to dB
                self.h_mag_prim_lin = self.h_mag_prim
                self.h_mag_prim = 20 * np.log10(self.h_mag_prim)
                self.h_phase_prim = np.degrees(self.h_phase_prim)
                
                # Update plots
                self.updatePlots()
                
                # Calculate hs/hp if we have both measurements
                if len(self.h_mag) > 0:
                    self.calculateHsHp()
                    
        except Exception as e:
            print(f"Error loading primary response: {e}")
    
    def calculateHsHp(self):
        if len(self.h_mag_prim) == 0 or len(self.h_mag) == 0:
            return
            
        # Make sure arrays are the same length
        min_len = min(len(self.h_mag), len(self.h_mag_prim))
        
        # Convert values to complex numbers
        h_mag_lin = np.array(self.h_mag_lin[:min_len])
        h_phase = np.array(self.h_phase[:min_len])
        h_prim_lin = self.h_mag_prim_lin[:min_len]
        h_prim_phase = self.h_phase_prim[:min_len]
        
        # Calculate complex transfer functions
        h_complex = h_mag_lin * np.exp(1j * np.radians(h_phase))
        h_prim_complex = h_prim_lin * np.exp(1j * np.radians(h_prim_phase))
        
        # Calculate hs/hp
        hs_hp = h_complex / h_prim_complex
        
        # Store real and imaginary parts
        self.hs_hp_real = np.real(hs_hp)
        self.hs_hp_imag = np.imag(hs_hp)
        
        # Update plots
        self.updatePlots()
    
    def updatePlots(self):
        # Get the frequencies measured so far
        freqs = self.frequencies[:self.current_index]
        
        # Update magnitude plot
        self.mag_plot.setData(freqs, self.h_mag)
        if len(self.h_mag_prim) > 0:
            self.mag_prim_plot.setData(freqs[:len(self.h_mag_prim)], self.h_mag_prim)
        
        # Update phase plot
        self.phase_plot.setData(freqs, self.h_phase)
        if len(self.h_phase_prim) > 0:
            self.phase_prim_plot.setData(freqs[:len(self.h_phase_prim)], self.h_phase_prim)
        
        # Update hs/hp plot
        if len(self.hs_hp_real) > 0:
            self.hs_hp_real_plot.setData(freqs[:len(self.hs_hp_real)], self.hs_hp_real)
            self.hs_hp_imag_plot.setData(freqs[:len(self.hs_hp_imag)], self.hs_hp_imag)
    
    def saveData(self):
        try:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
            if filename:
                with open(filename, 'w') as f:
                    f.write("Frequency,Magnitude,Phase")
                    if len(self.h_mag_prim) > 0:
                        f.write(",PrimaryMagnitude,PrimaryPhase")
                    if len(self.hs_hp_real) > 0:
                        f.write(",HsHpReal,HsHpImag")
                    f.write("\n")
                    
                    for i in range(self.current_index):
                        f.write(f"{self.frequencies[i]},{self.h_mag[i]},{self.h_phase[i]}")
                        
                        if len(self.h_mag_prim) > 0 and i < len(self.h_mag_prim):
                            f.write(f",{self.h_mag_prim[i]},{self.h_phase_prim[i]}")
                        else:
                            f.write(",,")
                            
                        if len(self.hs_hp_real) > 0 and i < len(self.hs_hp_real):
                            f.write(f",{self.hs_hp_real[i]},{self.hs_hp_imag[i]}")
                        
                        f.write("\n")
                    
                print(f"Data saved to {filename}")
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def closeEvent(self, event):
        # Clean up when the application is closed
        if hasattr(self, 'device'):
            self.device.close()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = TransferFunctionAnalyzer()
    window.show()
    sys.exit(app.exec_())