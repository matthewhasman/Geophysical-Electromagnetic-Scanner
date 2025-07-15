from ctypes import *
import time
import sys
import mido
import numpy as np
from dwfconstants import *

# Default frequency limits
min_frequency = 3e2  # 300 Hz 
max_frequency = 7e4  # 70 kHz

# Function to convert MIDI note number to frequency in Hz with bounds checking
def midi_to_freq(note):
    freq = 440 * (2 ** ((note - 69) / 12))
    freq *= 1 # increase freq into audible bounds 
    # Apply frequency bounds
    if freq < min_frequency:
        print(f"Warning: Note {note} frequency ({freq:.2f} Hz) is below device minimum ({min_frequency} Hz)")
        return min_frequency
    elif freq > max_frequency:
        print(f"Warning: Note {note} frequency ({freq:.2f} Hz) is above device maximum ({max_frequency} Hz)")
        return max_frequency
    return freq

# Load the DWF library based on platform
if sys.platform.startswith("win"):
    dwf = cdll.dwf
elif sys.platform.startswith("darwin"):
    dwf = cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
else:
    dwf = cdll.LoadLibrary("libdwf.so")

# Continue running after device close, prevent temperature drifts
dwf.FDwfParamSet(c_int(4), c_int(0))  # 4 = DwfParamOnClose, 0 = continue 1 = stop 2 = shutdown

# Get DWF version
version = create_string_buffer(16)
dwf.FDwfGetVersion(version)
print("DWF Version: " + str(version.value))

# Open device
hdwf = c_int()
print("Opening first device...")
dwf.FDwfDeviceOpen(c_int(-1), byref(hdwf))

if hdwf.value == hdwfNone.value:
    print("Failed to open device")
    quit()

# Initialize analog output
channel = c_int(0)

def configure_analog_out(freq, amplitude=2.0):
    """Configure the analog output with the specified frequency and amplitude"""
    dwf.FDwfAnalogOutNodeEnableSet(hdwf, channel, AnalogOutNodeCarrier, c_int(1))
    dwf.FDwfAnalogOutNodeFunctionSet(hdwf, channel, AnalogOutNodeCarrier, funcSine)
    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(freq))
    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(2.0))
    dwf.FDwfAnalogOutNodeOffsetSet(hdwf, channel, AnalogOutNodeCarrier, c_double(0.0))
    dwf.FDwfAnalogOutRepeatSet(hdwf, channel, c_int(1))
    dwf.FDwfAnalogOutRunSet(hdwf, channel, c_double(2.0))

# Initialize with a default frequency
configure_analog_out(1000.0)

def play_midi_file(midi_file_path):
    try:
        # Load MIDI file
        midi_file = mido.MidiFile(midi_file_path)
        print(f"Loaded MIDI file: {midi_file_path}")
        
        # Process tempo for timing
        tempo = 500000  # Default tempo in microseconds per beat (120 BPM)
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
            if tempo != 500000:
                break
        
        ticks_per_beat = midi_file.ticks_per_beat
        seconds_per_tick = tempo / (1000000 * ticks_per_beat)
        
        print(f"Tempo: {60000000 / tempo:.1f} BPM, Ticks per beat: {ticks_per_beat}")
        
        # Merge all tracks into a single track for easier sequential processing
        merged_track = mido.merge_tracks(midi_file.tracks)
        
        # Print some basic information about the MIDI file
        print(f"MIDI file format: {midi_file.type}")
        print(f"Number of tracks: {len(midi_file.tracks)}")
        print(f"Duration: ~{midi_file.length:.2f} seconds")
        
        # Dictionary to track active notes and their velocities: {note_number: velocity}
        active_notes = {}
        
        # Global volume scaling (for control change messages)
        global_volume = 1.0
        
        # Start the clock
        start_time = time.time()
        expected_time = 0
        
        print("\nStarting playback...")
        
        for msg in merged_track:
            # Calculate the expected time for this message
            expected_time += msg.time * seconds_per_tick
            
            # Calculate how much time to wait
            actual_time = time.time() - start_time
            wait_time = expected_time - actual_time
            
            # Only wait if we're ahead of schedule
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Process the MIDI message
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note on event
                active_notes[msg.note] = msg.velocity
                
                # Find the highest active note (melody)
                if active_notes:
                    highest_note = max(active_notes.keys())
                    velocity = active_notes[highest_note]
                    freq = midi_to_freq(highest_note)
                    
                    # Scale velocity (0-127) to amplitude (0-2.0), considering global volume
                    amplitude = 2.0 * (velocity / 127.0) * global_volume
                    
                    print(f"Playing note: {highest_note} (Freq: {freq:.2f} Hz, Amplitude: {amplitude:.2f})")
                    
                    # Set the frequency and amplitude for the analog output
                    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(freq))
                    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(amplitude))
                    
                    # Configure and start the output
                    dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note off event
                note = msg.note
                if note in active_notes:
                    del active_notes[note]
                
                # If there are still active notes, play the highest one
                if active_notes:
                    highest_note = max(active_notes.keys())
                    velocity = active_notes[highest_note]
                    freq = midi_to_freq(highest_note)
                    
                    # Scale velocity (0-127) to amplitude (0-2.0), considering global volume
                    amplitude = 2.0 * (velocity / 127.0) * global_volume
                    
                    print(f"Switching to note: {highest_note} (Freq: {freq:.2f} Hz, Amplitude: {amplitude:.2f})")
                    
                    # Set the frequency and amplitude for the analog output
                    dwf.FDwfAnalogOutNodeFrequencySet(hdwf, channel, AnalogOutNodeCarrier, c_double(freq))
                    dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(amplitude))
                    
                    # Configure and start the output
                    dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))
                else:
                    # If no notes are active, stop the output
                    dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(0))
            
            # Handle program change (instrument selection)
            elif msg.type == 'program_change':
                print(f"Program change: {msg.program} - Note: This hardware cannot change instruments")
            
            # Handle control changes (volume, etc.)
            elif msg.type == 'control_change':
                if msg.control == 7:  # Main volume
                    global_volume = msg.value / 127.0
                    print(f"Volume change: {global_volume:.2f}")
                    
                    # Update the amplitude of the currently playing note, if any
                    if active_notes:
                        highest_note = max(active_notes.keys())
                        velocity = active_notes[highest_note]
                        amplitude = 2.0 * (velocity / 127.0) * global_volume * 2
                        dwf.FDwfAnalogOutNodeAmplitudeSet(hdwf, channel, AnalogOutNodeCarrier, c_double(amplitude))
                        dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(1))
        
        print("\nPlayback finished")
        
    except Exception as e:
        print(f"Error playing MIDI file: {e}")
    finally:
        # Make sure to stop the output
        dwf.FDwfAnalogOutConfigure(hdwf, channel, c_int(0))

# Main function
def main():
    if len(sys.argv) < 2:
        print("Please provide a MIDI file path")
        print("Usage: python midi_player.py <midi_file_path>")
        return
    
    midi_file_path = sys.argv[1]
    
    try:
        play_midi_file(midi_file_path)
    finally:
        # Make sure to close the device
        print("Closing device...")
        dwf.FDwfDeviceClose(hdwf)

if __name__ == "__main__":
    main()