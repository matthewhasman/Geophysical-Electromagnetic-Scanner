from ctypes import *
import ctypes                     # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators
import numpy as np
import time
import math

class AnalogDiscovery:

    def init(self):
        # load the dynamic library, get constants path (the path is OS specific)
        if platform.startswith("win"):
            # on Windows
            self.dwf = ctypes.cdll.dwf
            constants_path = "C:" + sep + "Program Files (x86)" + sep + "Digilent" + sep + "WaveFormsSDK" + sep + "samples" + sep + "py"
        elif platform.startswith("darwin"):
            # on macOS
            lib_path = sep + "Library" + sep + "Frameworks" + sep + "dwf.framework" + sep + "dwf"
            self.dwf = ctypes.cdll.LoadLibrary(lib_path)
            constants_path = sep + "Applications" + sep + "WaveForms.app" + sep + "Contents" + sep + "Resources" + sep + "SDK" + sep + "samples" + sep + "py"
        else:
            # on Linux
            self.dwf = ctypes.cdll.LoadLibrary("libdwf.so")
            constants_path = sep + "usr" + sep + "share" + sep + "digilent" + sep + "waveforms" + sep + "samples" + sep + "py"
        
        # import constants
        path.append(constants_path)
        import dwfconstants as constants
        self.constants = constants
        self.handle = self.open()
        
        # Prevent temperature drift
        self.dwf.FDwfParamSet(constants.DwfParamOnClose, c_int(0))
        self.dwf.FDwfDeviceAutoConfigureSet(self.handle, c_int(0))  # Manual configuration


    def open(self):
        """
            open the first available device
        """
        # this is the device handle - it will be used by all functions to "address" the connected device
        device_handle = ctypes.c_int()
        # connect to the first available device
        self.dwf.FDwfDeviceOpen(ctypes.c_int(-1), ctypes.byref(device_handle))
        return device_handle
    
    def init_scope(self, sampling_frequency=20e06, buffer_size=8192, offset=0, amplitude_range=5):
        """
            initialize the oscilloscope
            parameters: 
                        - sampling frequency in Hz, default is 20MHz
                        - buffer size, default is 8192
                        - offset voltage in Volts, default is 0V
                        - amplitude range in Volts, default is ±5V
        """
        # enable all channels
        self.dwf.FDwfAnalogInChannelEnableSet(self.handle, ctypes.c_int(0), ctypes.c_bool(True))
    
        # set offset voltage (in Volts)
        self.dwf.FDwfAnalogInChannelOffsetSet(self.handle, ctypes.c_int(0), ctypes.c_double(offset))
    
        # set range (maximum signal amplitude in Volts)
        self.dwf.FDwfAnalogInChannelRangeSet(self.handle, ctypes.c_int(0), ctypes.c_double(amplitude_range))
    
        # set the buffer size (data point in a recording)
        self.dwf.FDwfAnalogInBufferSizeSet(self.handle, ctypes.c_int(buffer_size))
    
        # set the acquisition frequency (in Hz)
        self.dwf.FDwfAnalogInFrequencySet(self.handle, ctypes.c_double(sampling_frequency))
    
        # disable averaging (for more info check the documentation)
        self.dwf.FDwfAnalogInChannelFilterSet(self.handle, ctypes.c_int(-1), self.constants.filterDecimate)
        self.buffer_size = buffer_size
        self.sampling_frequency = sampling_frequency
        return
    
    def record(self, channel):
        """
            record an analog signal
            parameters: - device data
                        - the selected oscilloscope channel (1-2, or 1-4)
            returns:    - buffer - a list with the recorded voltages
                        - time - a list with the time moments for each voltage in seconds (with the same index as "buffer")
        """
        # set up the instrument
        self.dwf.FDwfAnalogInConfigure(self.handle, ctypes.c_bool(False), ctypes.c_bool(True))
    
        # read data to an internal buffer
        while True:
            status = ctypes.c_byte()    # variable to store buffer status
            self.dwf.FDwfAnalogInStatus(self.handle, ctypes.c_bool(True), ctypes.byref(status))
    
            # check internal buffer status
            if status.value == self.constants.DwfStateDone.value:
                    # exit loop when ready
                    break
    
        # copy buffer
        buffer = (ctypes.c_double * self.buffer_size)()   # create an empty buffer
        self.dwf.FDwfAnalogInStatusData(self.handle, ctypes.c_int(channel - 1), buffer, ctypes.c_int(self.buffer_size))
    
        # calculate aquisition time
        time = range(0, self.buffer_size)
        time = [moment / self.sampling_frequency for moment in time]
    
        # convert into list
        buffer = [float(element) for element in buffer]
        return buffer, time
    
    

    def generate(self, channel, function, offset, frequency=1e03, amplitude=1, symmetry=50, wait=0, run_time=0, repeat=0, data=[]):
        """
            generate an analog signal
            parameters: - the selected wavegen channel (1-2)
                        - function - possible: custom, sine, square, triangle, noise, ds, pulse, trapezium, sine_power, ramp_up, ramp_down
                        - offset voltage in Volts
                        - frequency in Hz, default is 1KHz
                        - amplitude in Volts, default is 1V
                        - signal symmetry in percentage, default is 50%
                        - wait time in seconds, default is 0s
                        - run time in seconds, default is infinite (0)
                        - repeat count, default is infinite (0)
                        - data - list of voltages, used only if function=custom, default is empty
        """
        # enable channel
        channel_idx = ctypes.c_int(channel - 1)
        self.dwf.FDwfAnalogOutNodeEnableSet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, ctypes.c_bool(True))
     
        # set function type
        self.dwf.FDwfAnalogOutNodeFunctionSet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, function)
     
        # load data if the function type is custom
        if function == self.constants.funcCustom:
            data_length = len(data)
            buffer = (ctypes.c_double * data_length)()
            for index in range(0, len(buffer)):
                buffer[index] = ctypes.c_double(data[index])
            self.dwf.FDwfAnalogOutNodeDataSet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, buffer, ctypes.c_int(data_length))
     
        # set frequency
        self.dwf.FDwfAnalogOutNodeFrequencySet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, ctypes.c_double(frequency))
     
        # set amplitude or DC voltage
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, ctypes.c_double(amplitude))
     
        # set offset
        self.dwf.FDwfAnalogOutNodeOffsetSet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, ctypes.c_double(offset))
     
        # set symmetry
        self.dwf.FDwfAnalogOutNodeSymmetrySet(self.handle, channel_idx, self.constants.AnalogOutNodeCarrier, ctypes.c_double(symmetry))
     
        # set running time limit
        self.dwf.FDwfAnalogOutRunSet(self.handle, channel_idx, ctypes.c_double(run_time))
     
        # set wait time before start
        self.dwf.FDwfAnalogOutWaitSet(self.handle, channel_idx, ctypes.c_double(wait))
     
        # set number of repeating cycles
        self.dwf.FDwfAnalogOutRepeatSet(self.handle, channel_idx, ctypes.c_int(repeat))
     
        # start
        self.dwf.FDwfAnalogOutConfigure(self.handle, channel_idx, ctypes.c_bool(True))
        return

    def close_generator(self, channel=0):
        """
            reset a wavegen channel, or all channels (channel=0)
        """
        channel_idx = ctypes.c_int(channel - 1)
        self.dwf.FDwfAnalogOutReset(self.handle, channel_idx)
        return

    def fft_analyze(self, channel, window_type=None, wait_for_stable=True):
        """
            Perform FFT analysis on a signal
            parameters: - the selected oscilloscope channel (1-2, or 1-4)
                        - window_type: None, FlatTop, Hamming, Hann, Blackman, etc.
                        - wait_for_stable: whether to wait for signal to stabilize
            returns:    - frequencies - array of frequency points
                        - magnitudes - array of magnitude values in dB
                        - phases - array of phase values in degrees
                        - raw_data - the raw time domain data
        """
        # Make sure buffer size is power of 2 for efficient FFT
        if not math.log2(self.buffer_size).is_integer():
            print("Warning: Buffer size is not a power of 2, which is suboptimal for FFT")
        
        # Start acquisition
        self.dwf.FDwfAnalogInConfigure(self.handle, ctypes.c_int(0), ctypes.c_int(1))
        
        if wait_for_stable:
            # Wait for signal to stabilize
            time.sleep(0.1)
        
        # Start acquisition again
        self.dwf.FDwfAnalogInConfigure(self.handle, ctypes.c_int(1), ctypes.c_int(1))
        
        # Wait for acquisition to complete
        while True:
            status = ctypes.c_byte()
            self.dwf.FDwfAnalogInStatus(self.handle, ctypes.c_int(1), ctypes.byref(status))
            if status.value == self.constants.DwfStateDone.value:
                break
            time.sleep(0.001)
        
        # Get the raw data
        raw_data = (ctypes.c_double * self.buffer_size)()
        self.dwf.FDwfAnalogInStatusData(self.handle, ctypes.c_int(channel - 1), raw_data, ctypes.c_int(self.buffer_size))
        
        # Apply window if specified
        window = (ctypes.c_double * self.buffer_size)()
        noise_equivalent_bw = ctypes.c_double()
        
        if window_type is None:
            window_type = self.constants.DwfWindowFlatTop
            
        self.dwf.FDwfSpectrumWindow(ctypes.byref(window), ctypes.c_int(self.buffer_size), 
                                   window_type, ctypes.c_double(1.0), 
                                   ctypes.byref(noise_equivalent_bw))
        
        # Apply window to data
        windowed_data = (ctypes.c_double * self.buffer_size)()
        for i in range(self.buffer_size):
            windowed_data[i] = raw_data[i] * window[i]
        
        # Perform FFT
        n_bins = self.buffer_size // 2 + 1
        magnitudes = (ctypes.c_double * n_bins)()
        phases = (ctypes.c_double * n_bins)()
        
        self.dwf.FDwfSpectrumFFT(ctypes.byref(windowed_data), ctypes.c_int(self.buffer_size), 
                                ctypes.byref(magnitudes), ctypes.byref(phases), ctypes.c_int(n_bins))
        
        # Convert magnitudes to dB
        sqrt2 = math.sqrt(2)
        mag_db = [20.0 * math.log10(magnitudes[i] / sqrt2) if magnitudes[i] > 0 else -200.0 for i in range(n_bins)]
        
        # Convert phases to degrees and handle low magnitude points
        phase_deg = []
        for i in range(n_bins):
            if mag_db[i] < -60:  # Mask phase at low magnitude
                phase_deg.append(0)
            else:
                phase_val = phases[i] * 180.0 / math.pi  # Convert to degrees
                if phase_val < 0:
                    phase_val = 180.0 + phase_val
                phase_deg.append(phase_val)
        
        # Calculate frequency points
        hzRate = ctypes.c_double()
        self.dwf.FDwfAnalogInFrequencyGet(self.handle, ctypes.byref(hzRate))
        hzTop = hzRate.value / 2
        
        frequencies = [hzTop * i / (n_bins - 1) for i in range(n_bins)]
        
        # Convert raw data to list
        raw_data_list = [float(raw_data[i]) for i in range(self.buffer_size)]
        
        return frequencies, mag_db, phase_deg, raw_data_list

    def frequency_response(self, input_channel, output_channel, start_freq, stop_freq, steps, 
                          amplitude=0.5, window_type=None, probe_attenuation=1.0):
        """
            Measure frequency response between two channels
            parameters: - input_channel: the channel connected to the input signal (1-2, or 1-4)
                        - output_channel: the channel connected to the output signal (1-2, or 1-4)
                        - start_freq: starting frequency in Hz
                        - stop_freq: ending frequency in Hz
                        - steps: number of frequency points
                        - amplitude: signal amplitude in Volts
                        - window_type: window function for FFT
                        - probe_attenuation: attenuation factor for input probe (e.g., 10 for 10x probe)
            returns:    - frequencies - array of frequency points
                        - magnitude - array of magnitude values in dB
                        - phase - array of phase values in degrees
        """
        # Generate logarithmic frequency sweep
        frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), num=steps)
        
        magnitude = []
        phase = []
        
        # Configure AWG on channel 1
        self.dwf.FDwfAnalogOutNodeEnableSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, c_int(1))
        self.dwf.FDwfAnalogOutNodeFunctionSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, self.constants.funcSine)
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, c_double(amplitude))
        
        # Prepare window for FFT
        window = (c_double * self.buffer_size)()
        vBeta = c_double(1.0)
        vNEBW = c_double()
        
        if window_type is None:
            window_type = self.constants.DwfWindowFlatTop
            
        self.dwf.FDwfSpectrumWindow(ctypes.byref(window), ctypes.c_int(self.buffer_size), 
                                   window_type, vBeta, ctypes.byref(vNEBW))
        
        for freq in frequencies:
            # Set AWG frequency
            self.dwf.FDwfAnalogOutNodeFrequencySet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, c_double(freq))
            self.dwf.FDwfAnalogOutConfigure(self.handle, c_int(0), c_int(1))
            
            # Start acquisition
            self.dwf.FDwfAnalogInConfigure(self.handle, c_int(1), c_int(1))
            
            # Wait for acquisition to complete
            while True:
                sts = c_byte()
                self.dwf.FDwfAnalogInStatus(self.handle, c_int(1), ctypes.byref(sts))
                if sts.value == self.constants.DwfStateDone.value:
                    break
                time.sleep(0.001)
            
            # Retrieve data for both channels
            samples_in = (c_double * self.buffer_size)()
            samples_out = (c_double * self.buffer_size)()
            self.dwf.FDwfAnalogInStatusData(self.handle, c_int(input_channel - 1), samples_in, self.buffer_size)
            self.dwf.FDwfAnalogInStatusData(self.handle, c_int(output_channel - 1), samples_out, self.buffer_size)
            
            # Apply window and perform FFT
            def process_data(samples):
                # Apply window
                for i in range(self.buffer_size):
                    samples[i] = samples[i] * window[i]
                
                # Perform FFT
                n_bins = self.buffer_size // 2 + 1
                bins = (c_double * n_bins)()
                phases = (c_double * n_bins)()
                self.dwf.FDwfSpectrumFFT(ctypes.byref(samples), self.buffer_size, ctypes.byref(bins), ctypes.byref(phases), n_bins)
                
                # Get frequency index
                freq_index = int(freq / self.sampling_frequency * self.buffer_size) + 1
                return bins[freq_index], phases[freq_index]
            
            # Process both channels
            in_mag, in_phase = process_data(samples_in)
            out_mag, out_phase = process_data(samples_out)
            
            # Apply probe attenuation if needed
            in_mag *= probe_attenuation
            
            # Compute transfer function
            if in_mag <= 0:
                mag_db = float('-inf')
                phase_diff = 0
            else:
                mag_db = 20 * math.log10(out_mag / in_mag)
                phase_diff = (out_phase - in_phase) * 180 / math.pi
                phase_diff = (phase_diff + 180) % 360 - 180  # Wrap to [-180, 180]
            
            magnitude.append(mag_db)
            phase.append(phase_diff)
        
        return frequencies.tolist(), magnitude, phase

    def close_scope(self):
        """
            Reset and close the oscilloscope
        """
        # Reset the analog in (oscilloscope)
        self.dwf.FDwfAnalogInReset(self.handle)
        return

    def close(self):
        """
            Close the device and clean up all resources
        """
        # First close the scope
        self.close_scope()
        
        # Then close the generator (all channels)
        self.close_generator()
        
        # Finally close the device
        self.dwf.FDwfDeviceClose(self.handle)
        return
