from ctypes import *
import ctypes                     # import the C compatible data types
from sys import platform, path    # this is needed to check the OS type and get the PATH
from os import sep                # OS specific file path separators
import numpy as np
import time
import math

class AnalogDiscovery:

    def __init__(self):
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

        # Get frequency range information
        self.min_freq = c_double(0)
        self.max_freq = c_double(0)
        self.dwf.FDwfAnalogInFrequencyInfo(self.handle, ctypes.byref(self.min_freq), ctypes.byref(self.max_freq))
        self.min_freq = self.min_freq.value
        self.max_freq = self.max_freq.value
        print(f"Frequency Range: {self.min_freq} Hz to {self.max_freq} Hz")

        # Get buffer size information
        self.min_buffer_size = ctypes.c_int(0)
        self.max_buffer_size = ctypes.c_int(0)
        self.dwf.FDwfAnalogInBufferSizeInfo(self.handle, ctypes.byref(self.min_buffer_size), ctypes.byref(self.max_buffer_size))
        self.min_buffer_size = self.min_buffer_size.value
        self.max_buffer_size = self.max_buffer_size.value
        print(f"Buffer Size Range: {self.min_buffer_size} to {self.max_buffer_size} samples")

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

    def fft_analyze(self, channel, window_type=None, wait_for_stable=True, requested_buffer_size=None):
        """
            Perform FFT analysis on a signal
            parameters: - the selected oscilloscope channel (1-2, or 1-4)
                        - window_type: None, FlatTop, Hamming, Hann, Blackman, etc.
                        - wait_for_stable: whether to wait for signal to stabilize
                        - requested_buffer_size: optional custom buffer size (can exceed hardware max)
            returns:    - frequencies - array of frequency points
                        - magnitudes - array of magnitude values in dB
                        - phases - array of phase values in degrees
                        - raw_data - the raw time domain data
        """
        # If a custom buffer size is requested that exceeds hardware max, use multiple acquisitions
        is_extended_buffer = False
        original_buffer_size = self.buffer_size
        hw_buffer_size = self.buffer_size
        
        if requested_buffer_size is not None and requested_buffer_size > self.max_buffer_size:
            is_extended_buffer = True
            requested_buffer_size = int(requested_buffer_size)
            # Ensure requested size is power of 2 for efficient FFT
            if not math.log2(requested_buffer_size).is_integer():
                next_power = math.ceil(math.log2(requested_buffer_size))
                requested_buffer_size = 2 ** next_power
                print(f"Adjusting requested buffer size to power of 2: {requested_buffer_size}")
            
            # Calculate how many acquisitions we need
            num_acquisitions = math.ceil(requested_buffer_size / self.max_buffer_size)
            hw_buffer_size = min(self.max_buffer_size, original_buffer_size)
            
            print(f"Using extended buffer mode: {requested_buffer_size} samples")
            print(f"Will concatenate {num_acquisitions} acquisitions of {hw_buffer_size} samples each")
            
            # Configure scope with maximum possible buffer size for efficiency
            if hw_buffer_size != self.buffer_size:
                print(f"Temporarily setting hardware buffer to {hw_buffer_size}")
                self.init_scope(
                    sampling_frequency=self.sampling_frequency,
                    buffer_size=hw_buffer_size,
                    offset=0,
                    amplitude_range=5
                )
        elif requested_buffer_size is not None:
            # If requested buffer size is within hardware limits, use it directly
            if requested_buffer_size != self.buffer_size:
                self.init_scope(
                    sampling_frequency=self.sampling_frequency, 
                    buffer_size=requested_buffer_size,
                    offset=0,
                    amplitude_range=5
                )
        
        # Make sure buffer size is power of 2 for efficient FFT
        if not math.log2(self.buffer_size).is_integer():
            print("Warning: Buffer size is not a power of 2, which is suboptimal for FFT")
        
        # If using extended buffer, collect multiple acquisitions
        if is_extended_buffer:
            # Preallocate the extended buffer
            combined_raw_data = np.zeros(requested_buffer_size)
            samples_collected = 0
            
            # Perform multiple acquisitions and combine them
            while samples_collected < requested_buffer_size:
                # Start acquisition
                self.dwf.FDwfAnalogInConfigure(self.handle, ctypes.c_int(0), ctypes.c_int(1))
                
                if wait_for_stable and samples_collected == 0:
                    # Wait for signal to stabilize (only for the first acquisition)
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
                
                # Get the raw data from this acquisition
                acq_buffer = (ctypes.c_double * self.buffer_size)()
                self.dwf.FDwfAnalogInStatusData(self.handle, ctypes.c_int(channel - 1), acq_buffer, ctypes.c_int(self.buffer_size))
                
                # Convert to numpy array and add to the combined buffer
                samples_to_copy = min(self.buffer_size, requested_buffer_size - samples_collected)
                combined_raw_data[samples_collected:samples_collected + samples_to_copy] = [float(acq_buffer[i]) for i in range(samples_to_copy)]
                samples_collected += samples_to_copy
                
                # Small delay between acquisitions
                time.sleep(0.01)
                
                print(f"Collected {samples_collected}/{requested_buffer_size} samples...")
            
            # Convert the numpy array to a ctypes array for FFT processing
            raw_data = (ctypes.c_double * requested_buffer_size)()
            for i in range(requested_buffer_size):
                raw_data[i] = ctypes.c_double(combined_raw_data[i])
                
            # Update buffer size for FFT calculation
            current_buffer_size = requested_buffer_size
            
        else:
            # Standard single acquisition mode
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
            current_buffer_size = self.buffer_size
        
        # Apply window if specified
        window = (ctypes.c_double * current_buffer_size)()
        noise_equivalent_bw = ctypes.c_double()
        
        if window_type is None:
            window_type = self.constants.DwfWindowFlatTop
            
        self.dwf.FDwfSpectrumWindow(ctypes.byref(window), ctypes.c_int(current_buffer_size), 
                                   window_type, ctypes.c_double(1.0), 
                                   ctypes.byref(noise_equivalent_bw))
        
        # Apply window to data
        windowed_data = (ctypes.c_double * current_buffer_size)()
        for i in range(current_buffer_size):
            windowed_data[i] = raw_data[i] * window[i]
        
        # Perform FFT
        n_bins = current_buffer_size // 2 + 1
        magnitudes = (ctypes.c_double * n_bins)()
        phases = (ctypes.c_double * n_bins)()
        
        self.dwf.FDwfSpectrumFFT(ctypes.byref(windowed_data), ctypes.c_int(current_buffer_size), 
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
        raw_data_list = [float(raw_data[i]) for i in range(current_buffer_size)]
        
        # If we changed the buffer size, restore the original
        if (is_extended_buffer or requested_buffer_size is not None) and original_buffer_size != self.buffer_size:
            print(f"Restoring original buffer size: {original_buffer_size}")
            self.init_scope(
                sampling_frequency=self.sampling_frequency,
                buffer_size=original_buffer_size,
                offset=0,
                amplitude_range=5
            )
        
        return frequencies, mag_db, phase_deg, raw_data_list

    def frequency_response(self, input_channel, output_channel, start_freq, stop_freq, steps, 
                          amplitude=0.5, window_type=None, probe_attenuation=1.0, auto_adjust_sampling=True,
                          min_samples_per_period=20, max_sampling_rate=100e6, extended_buffer=True,
                          fixed_buffer_size=None):
        """
            Measure frequency response between two channels with coherent sampling
            parameters: - input_channel: the channel connected to the input signal (1-2, or 1-4)
                        - output_channel: the channel connected to the output signal (1-2, or 1-4)
                        - start_freq: starting frequency in Hz
                        - stop_freq: ending frequency in Hz
                        - steps: number of frequency points
                        - amplitude: signal amplitude in Volts
                        - window_type: window function for FFT
                        - probe_attenuation: attenuation factor for input probe (e.g., 10 for 10x probe)
                        - auto_adjust_sampling: automatically adjust sampling rate based on frequency
                        - min_samples_per_period: minimum number of samples per signal period
                        - max_sampling_rate: maximum sampling rate in Hz
                        - extended_buffer: use extended buffer mode to exceed hardware limitations
                        - fixed_buffer_size: optional custom buffer size to use for all frequencies
            returns:    - frequencies - array of frequency points
                        - magnitude - array of magnitude values in dB
                        - phase - array of phase values in degrees
        """
        # Determine buffer size to use
        if fixed_buffer_size is not None:
            # Use specified buffer size
            if fixed_buffer_size > self.max_buffer_size and not extended_buffer:
                print(f"Warning: Requested buffer size {fixed_buffer_size} exceeds hardware maximum {self.max_buffer_size}")
                print("Using extended buffer mode to accommodate the request")
                extended_buffer = True
            FIXED_BUFFER_SIZE = fixed_buffer_size
        else:
            # Default buffer size if none specified
            FIXED_BUFFER_SIZE = 2**13
            
            # If extended buffer is enabled, ensure we don't exceed reasonable limits
            if extended_buffer and FIXED_BUFFER_SIZE > 2**24:  # Cap at 16M points for extended buffer
                FIXED_BUFFER_SIZE = 2**24
                print(f"Capping extended buffer size at {FIXED_BUFFER_SIZE} samples")
        
        # Log buffer mode
        if extended_buffer:
            print(f"Using extended buffer mode with {FIXED_BUFFER_SIZE} samples")
            # If buffer exceeds hardware limits, we'll handle it in fft_analyze
        else:
            print(f"Using standard buffer mode with max {min(FIXED_BUFFER_SIZE, self.max_buffer_size)} samples")
            FIXED_BUFFER_SIZE = min(FIXED_BUFFER_SIZE, self.max_buffer_size)
        
        # Generate desired logarithmic frequency points
        desired_frequencies = np.logspace(np.log10(start_freq), np.log10(stop_freq), num=steps)
        
        # Storage for actual frequencies and measurements
        actual_frequencies = []
        magnitude = []
        phase = []
        
        # Configure AWG on channel 1
        self.dwf.FDwfAnalogOutNodeEnableSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, c_int(1))
        self.dwf.FDwfAnalogOutNodeFunctionSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, self.constants.funcSine)
        self.dwf.FDwfAnalogOutNodeAmplitudeSet(self.handle, c_int(0), self.constants.AnalogOutNodeCarrier, c_double(amplitude))
        
        # Original sampling rate
        original_sampling_rate = self.sampling_frequency
        original_buffer_size = self.buffer_size
        
        # Make sure buffer size is set to our fixed value (if not using extended buffer)
        if not extended_buffer and self.buffer_size != FIXED_BUFFER_SIZE:
            print(f"Setting buffer size to fixed value of {FIXED_BUFFER_SIZE}")
            self.init_scope(
                sampling_frequency=self.sampling_frequency,
                buffer_size=FIXED_BUFFER_SIZE,
                offset=0,
                amplitude_range=2
            )
        
        # Process each desired frequency point
        for target_freq in desired_frequencies:
            try:
                # If auto-adjusting, calculate sampling rate for this frequency to ensure coherent sampling
                if auto_adjust_sampling:
                    # Minimum sampling rate based on desired samples per period
                    min_rate = target_freq * min_samples_per_period
                    
                    # For coherent sampling with a fixed buffer size, we need to find a sampling rate
                    # such that an integer number of cycles fits exactly in the buffer
                    
                    # Choose a reasonable number of cycles (more for higher frequencies)
                    if target_freq < 1000:
                        cycles = max(1, min(8, int(FIXED_BUFFER_SIZE / min_samples_per_period)))
                    elif target_freq < 10000:
                        cycles = max(8, min(32, int(FIXED_BUFFER_SIZE / min_samples_per_period)))
                    else:
                        cycles = max(32, min(128, int(FIXED_BUFFER_SIZE / min_samples_per_period)))
                    
                    # Calculate sampling rate: buffer_size * frequency / cycles
                    # This ensures an exact number of cycles in the buffer
                    sampling_rate = FIXED_BUFFER_SIZE * target_freq / cycles
                    
                    # Make sure sampling rate is within bounds
                    if sampling_rate < 1000:
                        # If we can't get coherent sampling at a reasonable rate, 
                        # we might need to adjust cycles
                        sampling_rate = 1000
                        cycles = max(1, int(FIXED_BUFFER_SIZE * target_freq / sampling_rate))
                    
                    if sampling_rate > max_sampling_rate:
                        sampling_rate = max_sampling_rate
                        cycles = max(1, int(FIXED_BUFFER_SIZE * target_freq / sampling_rate))
                    
                    # Calculate the actual frequency to achieve coherent sampling
                    actual_freq = cycles * sampling_rate / FIXED_BUFFER_SIZE
                    
                    # Print adjustment information
                    bin_number = int(cycles * FIXED_BUFFER_SIZE / FIXED_BUFFER_SIZE)  # Should be equal to cycles
                    print(f"Desired: {target_freq:.2f} Hz, Adjusted: {actual_freq:.2f} Hz")
                    print(f"  Cycles: {cycles}, Bin: {bin_number}, Sampling rate: {sampling_rate/1e6:.3f} MHz")
                    
                    # Reconfigure scope sampling rate if needed (only if not using extended buffer)
                    actual_hw_buffer = min(FIXED_BUFFER_SIZE, self.max_buffer_size) if not extended_buffer else self.buffer_size
                    if not extended_buffer and abs(sampling_rate - self.sampling_frequency) > 0.01 * self.sampling_frequency:
                        print(f"Reconfiguring scope sampling rate")
                        self.init_scope(
                            sampling_frequency=sampling_rate,
                            buffer_size=actual_hw_buffer,
                            offset=0,
                            amplitude_range=2
                        )
                    
                    # Use the actual frequency that allows coherent sampling
                    measurement_freq = actual_freq
                else:
                    # If not auto-adjusting, use the target frequency directly
                    measurement_freq = target_freq
                
                # Prepare window for FFT
                if window_type is None:
                    # For coherent sampling, rectangular window (no window) can give best amplitude accuracy
                    if auto_adjust_sampling:
                        # Use rectangular window for coherent sampling
                        window_type = self.constants.DwfWindowRectangular
                    else:
                        # Use FlatTop as default for non-coherent sampling
                        window_type = self.constants.DwfWindowFlatTop
                
                # Set AWG to the exact measurement frequency
                self.dwf.FDwfAnalogOutNodeFrequencySet(self.handle, c_int(0), 
                                                      self.constants.AnalogOutNodeCarrier, 
                                                      c_double(measurement_freq))
                self.dwf.FDwfAnalogOutConfigure(self.handle, c_int(0), c_int(1))
                
                # Wait for output to stabilize
                # For coherent sampling, waiting an exact number of periods can help
                stabilization_periods = 10  # Wait for at least 10 periods
                stabilization_time = max(0.01, stabilization_periods / measurement_freq)
                print(f"Waiting {stabilization_time:.3f}s for signal to stabilize...")
                time.sleep(stabilization_time)
                
                # Perform FFT analysis on both channels using the updated fft_analyze method
                # This handles extended buffers automatically if needed
                freq1, mag1, phase1, raw1 = self.fft_analyze(
                    channel=input_channel, 
                    window_type=window_type,
                    wait_for_stable=True,
                    requested_buffer_size=FIXED_BUFFER_SIZE if extended_buffer else None
                )
                
                freq2, mag2, phase2, raw2 = self.fft_analyze(
                    channel=output_channel, 
                    window_type=window_type,
                    wait_for_stable=False,
                    requested_buffer_size=FIXED_BUFFER_SIZE if extended_buffer else None
                )
                
                # Find the bin closest to our measurement frequency
                bin_size = self.sampling_frequency / len(raw1) * 2  # Calculate actual bin size
                exact_bin = int(round(measurement_freq / bin_size))
                
                # Sanity check - should be in range
                if exact_bin >= len(freq1):
                    exact_bin = len(freq1) - 1
                elif exact_bin < 0:
                    exact_bin = 0
                
                # Get magnitude and phase at the measurement frequency
                in_mag = 10**(mag1[exact_bin]/20)  # Convert from dB to linear
                in_phase = phase1[exact_bin]
                out_mag = 10**(mag2[exact_bin]/20)  # Convert from dB to linear
                out_phase = phase2[exact_bin]
                
                # Apply probe attenuation if needed
                in_mag *= probe_attenuation
                
                # Compute transfer function
                if in_mag <= 0:
                    mag_db = float('-inf')
                    phase_diff = 0
                else:
                    mag_db = 20 * math.log10(out_mag / in_mag)
                    phase_diff = out_phase - in_phase
                    phase_diff = (phase_diff + 180) % 360 - 180  # Wrap to [-180, 180]
                
                # Store results
                actual_frequencies.append(measurement_freq)
                magnitude.append(mag_db)
                phase.append(phase_diff)
                
            except Exception as e:
                print(f"Error measuring at {target_freq} Hz: {str(e)}")
                import traceback
                traceback.print_exc()
                # Skip this frequency point
        
        # Restore original settings if needed
        if auto_adjust_sampling and (original_buffer_size != self.buffer_size or 
                                     original_sampling_rate != self.sampling_frequency):
            print("Restoring original scope settings")
            self.init_scope(
                sampling_frequency=original_sampling_rate,
                buffer_size=original_buffer_size,
                offset=0,
                amplitude_range=2
            )
        
        # If no valid measurements, return empty arrays
        if not actual_frequencies:
            print("Warning: No valid measurements were made")
            return np.array([]), np.array([]), np.array([])
        
        return np.array(actual_frequencies), np.array(magnitude), np.array(phase)

    def close_scope(self):
        """
            Reset and close the oscilloscope
        """
        # Reset the analog in (scope) configuration
        self.dwf.FDwfAnalogInReset(self.handle)
        return

    def close(self):
        """
            Close the device and clean up all resources
        """
        # First close the scope
        self.close_scope()
        
        # Then close all generator channels
        self.close_generator(channel=0)  # 0 means all channels
        
        # Finally close the device
        self.dwf.FDwfDeviceClose(self.handle)
        return
