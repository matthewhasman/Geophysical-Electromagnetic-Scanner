import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
import pickle

# Import the forward simulation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forward_simulation

# Function to prepare 1D data from the complex response tensor
def prepare_1d_data(response_list, parameters, x_coords, y_coords):
    # Extract frequency data at each spatial location
    num_simulations = len(response_list)
    num_freqs = response_list[0].shape[-1]
    
    all_frequency_data = []
    all_labels = []
    
    # Process each simulation
    for sim_idx in range(num_simulations):
        response = response_list[sim_idx]
        
        # Extract parameters for this simulation
        xc, yc, zc = parameters[sim_idx, 0], parameters[sim_idx, 1], parameters[sim_idx, 2]
        length, rad = parameters[sim_idx, 3], parameters[sim_idx, 4]
        
        # Process each spatial location in the grid
        for x_idx, x in enumerate(x_coords):
            for y_idx, y in enumerate(y_coords):
                try:
                    # Extract frequency data at this location (complex values)
                    freq_data_complex = response[x_idx, y_idx, :]
                    
                    # Split complex data into real and imaginary parts
                    # Ensure the data is actually complex, if not, add zero imaginary component
                    if np.iscomplexobj(freq_data_complex):
                        real_part = np.real(freq_data_complex)
                        imag_part = np.imag(freq_data_complex)
                    else:
                        real_part = freq_data_complex
                        imag_part = np.zeros_like(freq_data_complex)
                    
                    # Concatenate real and imaginary parts
                    freq_data = np.concatenate([real_part, imag_part])
                    
                    # Calculate distance from measurement point to object center
                    distance = np.sqrt((x - xc)**2 + (y - yc)**2)
                    
                    # Determine if this is a "positive" or "negative" example
                    # Consider points far from the object center as negative examples
                    is_present = 1.0
                    if distance > 3.0:  # Assuming 3m is far enough to consider it "not present"
                        is_present = 0.0
                    
                    # Create label vector: [is_present, x_center, y_center, z_center, length, radius]
                    label = np.array([is_present, xc, yc, zc, length, rad])
                    
                    all_frequency_data.append(freq_data)
                    all_labels.append(label)
                    
                except IndexError:
                    # Skip if the index is out of range
                    continue
    
    return np.array(all_frequency_data), np.array(all_labels)

def main():
    # number of samples - increase for more training data
    n_sims = 100  # Increased from 2 to get more diverse training data

    # survey space
    x_min = -4  # m
    x_max = 4   # m
    dx = 0.05   # m
    
    # Default frequency and frequency limits
    min_frequency = 3e2  # 300 Hz 
    max_frequency = 3e4  # 30 kHz

    # Global frequency variable
    freqs = np.linspace(min_frequency, max_frequency, 8)

    # mechanical constants
    dipole_dist = 0.9  # spacing between coils (m)
    sensor_ht = np.random.uniform(0.05, 1.5, n_sims)  # height of the sensor above ground (m)

    # random parameters
    conductivity = np.random.uniform(1e4, 1e6, n_sims)
    permeability = 4*np.pi*1e-7  # free space permeability
    xc = np.random.uniform(x_min, x_max, n_sims)
    yc = np.random.uniform(x_min, x_max, n_sims)
    zc = np.random.uniform(0.5, 2, n_sims)
    dincl = np.random.uniform(0, 90, n_sims)     # inclination angle in degrees
    ddecl = np.random.uniform(0, 360, n_sims)    # declination angle in degrees
    length = np.random.uniform(0.05, 0.2, n_sims)
    rad = np.random.uniform(1/6, 1, n_sims) * length

    # precision of simulation
    n_loops = 100

    response_list = []
    for i in range(n_sims):
        # run simulation for each set of parameters
        response = forward_simulation.fem_pipe(
            conductivity[i],
            permeability,
            xc[i],
            yc[i],
            zc[i],
            dincl[i],
            ddecl[i],
            dipole_dist,
            sensor_ht[i],
            freqs,
            x_min,
            x_max,
            dx,
            n_loops,
            length[i],
            rad[i]
        )
        response_list.append(response)
        print(f"Simulation {i+1}/{n_sims} completed.")

    # Create coordinate arrays based on your parameters
    x_coords = np.arange(x_min, x_max + dx, dx)
    y_coords = np.arange(x_min, x_max + dx, dx)
    
    # Prepare parameters array
    parameters = np.column_stack((xc, yc, zc, length, rad))
    
    # Prepare 1D frequency data and labels with complex handling
    frequency_data, labels = prepare_1d_data(response_list, parameters, x_coords, y_coords)
    
    # Check the shape of the frequency data
    print(f"Frequency data shape: {frequency_data.shape}")
    print(f"Number of samples: {len(frequency_data)}")
    
    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        frequency_data, labels, test_size=0.1, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    # Normalize frequency data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the datasets and scaler to files
    data_dict = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'freqs': freqs
    }
    
    # Save the dataset
    with open('models/fdem_dataset.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    
    # Save the scaler separately
    with open('models/fdem_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Dataset generation completed and saved to 'fdem_dataset.pkl'")
    print("Scaler saved to 'fdem_scaler.pkl'")

if __name__ == "__main__":
    main()