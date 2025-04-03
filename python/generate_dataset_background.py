import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
import pickle
from scipy.constants import mu_0

# Import SimPEG modules
import simpeg.electromagnetics.frequency_domain as fdem
from simpeg import maps

# Import the forward simulation module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forward_simulation

# Function to generate 1D background response using SimPEG
def generate_background_response(freqs, x_coords, y_coords, n_background_models=10):
    """
    Generate background responses using SimPEG 1D layered earth model
    
    Args:
        freqs: Array of frequencies
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
        n_background_models: Number of different background models to generate
        
    Returns:
        List of background responses with shape (n_models, len(x_coords), len(y_coords), len(freqs))
    """
    background_responses = []
    
    # Source-receiver geometry for each grid point
    dipole_dist = 0.9  # spacing between coils (m)
    
    for model_idx in range(n_background_models):
        print(f"Generating background model {model_idx+1}/{n_background_models}")
        
        # Create random 1D layered earth model (2-4 layers)
        n_layers = np.random.randint(2, 5)
        
        # Layer thicknesses (m) - between 0.5m and 5m
        layer_thicknesses = np.random.uniform(0.5, 5.0, n_layers-1)
        
        # Layer conductivities (S/m) in log space - random but realistic values
        log_conductivity_values = np.random.uniform(-3, 2, n_layers)  # -3 to 2 log(S/m)
        layer_conductivities = np.power(10, log_conductivity_values)
        
        # Model for this background
        log_conductivity_model = np.log(layer_conductivities)
        
        # Create grid response array
        grid_response = np.zeros((len(x_coords), len(y_coords), len(freqs)), dtype=complex)
        
        # Random sensor height for this background model
        sensor_ht = np.random.uniform(0.05, 1.0)
        
        # Loop through grid points
        for x_idx, x in enumerate(x_coords):
            for y_idx, y in enumerate(y_coords):
                # Source location at this grid point
                source_location = np.array([x, y, sensor_ht])
                
                # Receiver location at this grid point
                receiver_offset = np.array([dipole_dist, 0, 0])  # Offset in x direction
                receiver_locations = np.array([source_location + receiver_offset])
                
                # Define orientation for both source and receiver (z-direction)
                source_orientation = 'z'
                receiver_orientation = 'z'
                data_type = "ppm"
                moment = 1.0
                
                # Set up the survey
                source_list = []
                for freq in freqs:
                    # Define receivers for real and imaginary components
                    receiver_list = []
                    receiver_list.append(
                        fdem.receivers.PointMagneticFieldSecondary(
                            receiver_locations,
                            orientation=receiver_orientation,
                            data_type=data_type,
                            component="real",
                        )
                    )
                    receiver_list.append(
                        fdem.receivers.PointMagneticFieldSecondary(
                            receiver_locations,
                            orientation=receiver_orientation,
                            data_type=data_type,
                            component="imag",
                        )
                    )
                    
                    # Define source
                    source_list.append(
                        fdem.sources.MagDipole(
                            receiver_list=receiver_list,
                            frequency=freq,
                            location=source_location,
                            orientation=source_orientation,
                            moment=moment,
                        )
                    )
                
                # Define survey
                survey = fdem.survey.Survey(source_list)
                
                # Create the simulation with log resistivity mapping
                log_resistivity_map = maps.ExpMap()  # transforms log(conductivity) to conductivity
                
                simulation = fdem.Simulation1DLayered(
                    survey=survey,
                    thicknesses=layer_thicknesses,
                    sigmaMap=log_resistivity_map,
                )
                
                # Predict the data
                dpred = simulation.dpred(log_conductivity_model)
                
                # Reshape the predicted data into complex values and store in grid
                dpred_complex = np.zeros(len(freqs), dtype=complex)
                for i in range(len(freqs)):
                    dpred_complex[i] = dpred[2*i] + 1j * dpred[2*i+1]  # real + j*imag
                
                # Store in the grid
                grid_response[x_idx, y_idx, :] = dpred_complex / 1e6 # normalize to hs/hp  from ppm
        
        background_responses.append(grid_response)
    
    return background_responses

# Function to prepare 1D data from the complex response tensor
def prepare_1d_data(response_list, parameters, background_responses, x_coords, y_coords):
    # Extract frequency data at each spatial location
    num_simulations = len(response_list)
    num_freqs = response_list[0].shape[-1]
    
    all_frequency_data = []
    all_labels = []
    
    # Process UXO simulations with background
    for sim_idx in range(num_simulations):
        response = response_list[sim_idx]
        
        # Extract parameters for this simulation
        xc, yc, zc = parameters[sim_idx, 0], parameters[sim_idx, 1], parameters[sim_idx, 2]
        length, rad = parameters[sim_idx, 3], parameters[sim_idx, 4]
        
        # Randomly select a background model for this simulation
        bg_idx = np.random.randint(0, len(background_responses))
        background = background_responses[bg_idx]

        # Add background to UXO response (complex addition)
        combined_response = response + background
        
        # Process each spatial location in the grid
        for x_idx, x in enumerate(x_coords):
            for y_idx, y in enumerate(y_coords):
                try:
                    # Extract frequency data at this location (complex values)
                    freq_data_complex = combined_response[x_idx, y_idx, :]
                    
                    # Split complex data into real and imaginary parts
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
                    if distance > 1.0:  # Assuming 1m is far enough to consider it "not present"
                        is_present = 0.0
                    
                    # Create label vector: [is_present, x_center, y_center, z_center, length, radius]
                    label = np.array([is_present, xc, yc, zc, length, rad])
                    
                    all_frequency_data.append(freq_data)
                    all_labels.append(label)
                    
                except IndexError:
                    # Skip if the index is out of range
                    continue
    
    # Add negative examples using only background response (no UXO)
    for bg_idx, background in enumerate(background_responses):
        # Sample random points from the background (not all to keep dataset balanced)
        n_samples = min(100, len(x_coords) * len(y_coords) // 5)  # Take ~20% of points
        
        # Random indices
        x_indices = np.random.choice(len(x_coords), n_samples)
        y_indices = np.random.choice(len(y_coords), n_samples)
        
        for i in range(n_samples):
            x_idx, y_idx = x_indices[i], y_indices[i]
            x, y = x_coords[x_idx], y_coords[y_idx]
            
            try:
                # Extract frequency data (complex values)
                freq_data_complex = background[x_idx, y_idx, :]
                
                # Split complex data into real and imaginary parts
                if np.iscomplexobj(freq_data_complex):
                    real_part = np.real(freq_data_complex)
                    imag_part = np.imag(freq_data_complex)
                else:
                    real_part = freq_data_complex
                    imag_part = np.zeros_like(freq_data_complex)
                
                # Concatenate real and imaginary parts
                freq_data = np.concatenate([real_part, imag_part])
                
                # Create label for background only (no UXO present)
                # Use dummy values for position and size
                is_present = 0.0
                label = np.array([is_present, 0.0, 0.0, 0.0, 0.0, 0.0])
                
                all_frequency_data.append(freq_data)
                all_labels.append(label)
                
            except IndexError:
                # Skip if the index is out of range
                continue
    
    return np.array(all_frequency_data), np.array(all_labels)

def main():
    # number of samples - increase for more training data
    n_sims = 50  # UXO simulations
    n_background_models = 10  # Different background models

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
    xc = np.zeros(n_sims)
    yc = np.zeros(n_sims)
    zc = np.random.uniform(0.5, 2, n_sims)
    dincl = np.random.uniform(0, 90, n_sims)     # inclination angle in degrees
    ddecl = np.random.uniform(0, 360, n_sims)    # declination angle in degrees
    length = np.random.uniform(0.05, 0.2, n_sims)
    rad = np.random.uniform(1/6, 1, n_sims) * length

    # precision of simulation
    n_loops = 100

    # Create coordinate arrays based on your parameters
    x_coords = np.arange(x_min, x_max, dx)
    y_coords = np.arange(x_min, x_max, dx)
    
    # First generate background responses using 1D layered earth models
    print("Generating background responses...")
    background_responses = generate_background_response(freqs, x_coords, y_coords, n_background_models)
    print(f"Generated {len(background_responses)} background models")
    
    # Run UXO simulations
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
        print(f"UXO Simulation {i+1}/{n_sims} completed.")
    
    # Prepare parameters array
    parameters = np.column_stack((xc, yc, zc, length, rad))
    
    # Prepare 1D frequency data and labels with complex handling and background responses
    print("Preparing dataset with both UXO+background and background-only responses...")
    frequency_data, labels = prepare_1d_data(response_list, parameters, background_responses, x_coords, y_coords)
    
    # Check the shape of the frequency data
    print(f"Frequency data shape: {frequency_data.shape}")
    print(f"Number of samples: {len(frequency_data)}")
    
    # Count positive and negative examples
    pos_count = np.sum(labels[:, 0] > 0.5)
    neg_count = np.sum(labels[:, 0] <= 0.5)
    print(f"Positive examples (UXO present): {pos_count}")
    print(f"Negative examples (background only): {neg_count}")
    
    # Split data into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        frequency_data, labels, test_size=0.1, random_state=42, stratify=labels[:, 0]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val[:, 0]
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
    with open('fdem_dataset_with_background.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
    
    # Save the scaler separately
    with open('fdem_scaler_background.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Dataset generation completed and saved to 'fdem_dataset_with_background.pkl'")
    print("Background sample saved to 'background_sample.pkl'")
    print("Scaler saved to 'fdem_scaler.pkl'")
    
    # Plot an example of a background response at a single location
    plt.figure(figsize=(12, 6))
    
    # Choose a point near the center
    x_idx = len(x_coords) // 2
    y_idx = len(y_coords) // 2
    
    sample_data = background_responses[0] + response_list[0]

    # Plot real and imaginary components
    plt.subplot(1, 2, 1)
    plt.semilogx(freqs, np.real(sample_data[x_idx, y_idx, :]), 'b-o', lw=2)
    plt.grid(True)
    plt.title('Real Component - Background Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Secondary Field')
    
    plt.subplot(1, 2, 2)
    plt.semilogx(freqs, np.imag(sample_data[x_idx, y_idx, :]), 'r-o', lw=2)
    plt.grid(True)
    plt.title('Imaginary Component - Background Response')
    plt.xlabel('Frequency (Hz)')
    
    plt.tight_layout()
    #plt.savefig('background_response_example.png')
    plt.show()

if __name__ == "__main__":
    main()