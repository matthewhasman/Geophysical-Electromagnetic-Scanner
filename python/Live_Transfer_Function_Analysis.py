from ctypes import *
from dwfconstants import *
import math
import time
import matplotlib.pyplot as plt
import sys
import numpy
import control
import scipy
from control.matlab import ss, bode
#import keyboard  # You may need to: pip install keyboard
import streamlit as st

# Enable interactive plotting
plt.ion()

if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

def initialize_hardware(max_retries=5, retry_delay=1):
    """Try to initialize hardware with retries
    Args:
        max_retries (int): Maximum number of connection attempts
        retry_delay (float): Delay in seconds between retries
    """
    for attempt in range(max_retries):
        version = create_string_buffer(16)
        dwf.FDwfGetVersion(version)
        print(f"Attempt {attempt + 1}/{max_retries}: DWF Version: {str(version.value)}")

        hdwf = c_int()
        szerr = create_string_buffer(512)
        print("Opening first device")
        if dwf.FDwfDeviceOpen(-1, byref(hdwf)) == 1 and hdwf.value != hdwfNone.value:
            print("Successfully connected to device")
            
            # Device configuration
            dwf.FDwfParamSet(DwfParamOnClose, c_int(0))
            dwf.FDwfDeviceAutoConfigureSet(hdwf, c_int(0))

            # Configure AWG
            dwf.FDwfAnalogOutNodeEnableSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_int(1))
            dwf.FDwfAnalogOutNodeFunctionSet(hdwf, c_int(0), AnalogOutNodeCarrier, funcSine)
            dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(0.5))
            dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))

            # Configure Scope
            nSamples = 2**16
            dwf.FDwfAnalogInFrequencySet(hdwf, c_double(20000000.0))
            dwf.FDwfAnalogInBufferSizeSet(hdwf, nSamples)
            dwf.FDwfAnalogInChannelEnableSet(hdwf, 0, c_int(1))
            dwf.FDwfAnalogInChannelRangeSet(hdwf, 0, c_double(2))
            dwf.FDwfAnalogInChannelEnableSet(hdwf, 1, c_int(1))
            dwf.FDwfAnalogInChannelRangeSet(hdwf, 1, c_double(2))
            
            return hdwf, nSamples
        
        dwf.FDwfGetLastErrorMsg(szerr)
        print(f"Attempt {attempt + 1} failed: {szerr.value}")
        
        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    return None

def setup_plotting():
    # Replace matplotlib figure with streamlit plots
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, which="both", ls="--")
    ax1.set_title('Transfer Function')
    
    ax2.set_ylabel('Phase (degrees)')
    ax2.grid(True, which="both", ls="--")
    
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('hs/hp')
    ax3.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)

def main():
    st.title("Live Transfer Function Analysis")
    
    # Add parameter controls in a sidebar
    with st.sidebar:
        st.header("Frequency Sweep Parameters")
        start_freq = st.number_input(
            "Start Frequency (Hz)", 
            min_value=1.0, 
            max_value=10000.0, 
            value=1000.0, 
            step=1.0
        )
        stop_freq = st.number_input(
            "Stop Frequency (Hz)", 
            min_value=100.0, 
            max_value=1000000.0, 
            value=100000.0, 
            step=100.0
        )
        steps = st.number_input(
            "Number of Steps", 
            min_value=4, 
            max_value=1000, 
            value=16, 
            step=1
        )
    
    # Create placeholders for the plots and metrics
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    # Initialize hardware with status message
    with st.spinner('Connecting to hardware...'):
        result = initialize_hardware(max_retries=5, retry_delay=1)
    
    if result is None:
        st.error("Failed to connect to the device after multiple attempts. Please check your hardware connection and try again.")
        return
    
    hdwf, nSamples = result
    st.success("Successfully connected to hardware!")

    # Setup frequency sweep parameters using the input values
    frequencies = numpy.logspace(numpy.log10(start_freq), numpy.log10(stop_freq), num=int(steps))

    # Setup FFT window
    rgdWindow = (c_double * nSamples)()
    vBeta = c_double(1.0)
    vNEBW = c_double()
    dwf.FDwfSpectrumWindow(byref(rgdWindow), c_int(nSamples), DwfWindowFlatTop, vBeta, byref(vNEBW))

    # Load primary response
    path = "../matlab/primary_curve_fit_python.mat"
    mat = scipy.io.loadmat(path)
    num = mat['num'].tolist()[0]
    den = mat['den'].tolist()[0]
    sys = control.TransferFunction(num, den)
    
    # Setup plotting
    fig, (ax1, ax2, ax3) = setup_plotting()
    lines = {}

    try:
        while True:
            h_mag = []
            h_mag_lin = []
            h_phase = []

            for freq in frequencies:
                    
                print(f"\rFrequency: {freq/1e3:.1f} kHz", end='')
                
                # Set frequency and acquire data
                dwf.FDwfAnalogOutNodeFrequencySet(hdwf, c_int(0), AnalogOutNodeCarrier, c_double(freq))
                dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(1))
                dwf.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(1))
                
                # Wait for acquisition
                while True:
                    sts = c_byte()
                    dwf.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))
                    if sts.value == DwfStateDone.value:
                        break
                    time.sleep(0.001)
                
                # Get data
                rgdSamples1 = (c_double * nSamples)()
                rgdSamples2 = (c_double * nSamples)()
                dwf.FDwfAnalogInStatusData(hdwf, 0, rgdSamples1, nSamples)
                dwf.FDwfAnalogInStatusData(hdwf, 1, rgdSamples2, nSamples)
                
                # Process data (same as original)
                def process_data(samples):
                    for i in range(nSamples):
                        samples[i] = samples[i] * rgdWindow[i]
                    nBins = nSamples // 2 + 1
                    rgdBins = (c_double * nBins)()
                    rgdPhase = (c_double * nBins)()
                    dwf.FDwfSpectrumFFT(byref(samples), nSamples, byref(rgdBins), byref(rgdPhase), nBins)
                    fIndex = int(freq / 20000000 * nSamples) + 1
                    return rgdBins[fIndex], rgdPhase[fIndex]
                
                c1_mag, c1_phase = process_data(rgdSamples1)
                c2_mag, c2_phase = process_data(rgdSamples2)
                
                if c1_mag > 0:
                    c1_mag *= 10  # 10x probe attenuation
                    h_db = 20 * math.log10(c2_mag / c1_mag)
                    h_linear = c2_mag / c1_mag
                    phase_diff = (c2_phase - c1_phase) * 180 / math.pi
                    phase_diff = (phase_diff + 180) % 360 - 180
                    
                    h_mag.append(h_db)
                    h_mag_lin.append(h_linear)
                    h_phase.append(phase_diff)
                else:
                    h_mag.append(float('nan'))
                    h_mag_lin.append(float('nan'))
                    h_phase.append(float('nan'))

            # Remove first point and calculate primary response
            frequencies_plot = frequencies[1:]
            h_mag = h_mag[1:]
            h_mag_lin = h_mag_lin[1:]
            h_phase = h_phase[1:]

            w = 2 * numpy.pi * frequencies_plot
            h_mag_prim, h_phase_prim, omega = bode(sys, w, plot=False)
            h_mag_prim_lin = h_mag_prim
            h_mag_prim = 20 * numpy.log10(h_mag_prim)
            h_phase_prim = numpy.degrees(h_phase_prim)

            # Calculate hs/hp
            h_mag_lin = numpy.array(h_mag_lin)
            h_phase = numpy.array(h_phase)
            h_prim_complex = h_mag_prim_lin * numpy.exp(1j * numpy.radians(h_phase_prim))
            h_complex = h_mag_lin * numpy.exp(1j * numpy.radians(h_phase))
            hs_hp = h_complex / h_prim_complex

            # Update plots using streamlit
            fig, (ax1, ax2, ax3) = setup_plotting()
            
            ax1.semilogx(frequencies_plot, h_mag, 'b-', label='Measured')
            ax1.semilogx(frequencies_plot, h_mag_prim, 'r--', label='Primary')
            ax1.set_ylabel('Magnitude (dB)')
            ax1.legend()

            ax2.semilogx(frequencies_plot, h_phase, 'b-', label='Measured')
            ax2.semilogx(frequencies_plot, h_phase_prim, 'r--', label='Primary')
            ax2.set_ylabel('Phase (degrees)')
            ax2.legend()

            ax3.semilogx(frequencies_plot, numpy.real(hs_hp), 'b-', label='Real')
            ax3.semilogx(frequencies_plot, numpy.imag(hs_hp), 'r--', label='Imaginary')
            ax3.set_ylabel('hs/hp')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.legend()

            plt.tight_layout()
            
            # Update the streamlit elements
            plot_placeholder.pyplot(fig)
            
            # Display current metrics
            with metrics_placeholder.container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Current Frequency", f"{frequencies[-1]/1e3:.1f} kHz")
                with col2:
                    st.metric("Latest Magnitude", f"{h_mag[-1]:.2f} dB")
            
            plt.close(fig)  # Clean up matplotlib figure
            
            time.sleep(0.01)  # Small delay to prevent overwhelming the browser

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Turn off the signal generator before closing
        dwf.FDwfAnalogOutConfigure(hdwf, c_int(0), c_int(0))
        dwf.FDwfDeviceCloseAll()
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main() 