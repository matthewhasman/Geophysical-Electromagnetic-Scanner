import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forward_simulation

# Define a custom dataset class for FDEM data that handles complex values and adds random noise
class FDEMDataset(Dataset):
    def __init__(self, frequency_data, labels, noise_level=0.05):
        """
        Custom dataset for FDEM data with noise augmentation
        
        Args:
            frequency_data: Array of frequency domain measurements (already split into real and imaginary)
            labels: Target values (presence, location, size)
            noise_level: Standard deviation of Gaussian noise to add (as fraction of data std)
        """
        self.frequency_data = torch.tensor(frequency_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.noise_level = noise_level
        
        # Calculate data standard deviation for scaling noise
        self.data_std = torch.std(self.frequency_data, dim=0)
        # Replace zeros with ones to avoid division by zero
        self.data_std = torch.where(self.data_std == 0, torch.ones_like(self.data_std), self.data_std)
        
    def __len__(self):
        return len(self.frequency_data)
    
    def __getitem__(self, idx):
        # Get the base data
        data = self.frequency_data[idx].clone()
        
        # Generate random noise scaled by data standard deviation and noise level
        noise = torch.randn_like(data) * self.noise_level * self.data_std
        
        # Add noise to data
        noisy_data = data + noise
        
        return noisy_data, self.labels[idx]

# Define the neural network architecture for 1D inversion with complex data
class FDEM1DInversionNet(nn.Module):
    def __init__(self, num_freqs, hidden_size=128):
        super(FDEM1DInversionNet, self).__init__()
        
        # Input size is doubled because we split complex values into real and imaginary parts
        input_size = num_freqs * 2
        
        # Fully connected layers for 1D frequency data (real and imaginary parts)
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
        )
        
        # Output layers split into different tasks
        self.presence_layer = nn.Sequential(
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()  # Probability of object being present
        )
        
        self.location_layer = nn.Linear(hidden_size//2, 3)  # x, y, z coordinates
        self.size_layer = nn.Linear(hidden_size//2, 2)  # length, radius
        
    def forward(self, x):
        x = self.fc_layers(x)
        
        presence = self.presence_layer(x)
        location = self.location_layer(x)
        size = self.size_layer(x)
        
        return presence, location, size

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

# Custom loss function that handles positive and negative examples
class FDEMInversionLoss(nn.Module):
    def __init__(self):
        super(FDEMInversionLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, pred_presence, pred_location, pred_size, true_labels):
        # Split true labels
        true_presence = true_labels[:, 0:1]
        true_location = true_labels[:, 1:4]
        true_size = true_labels[:, 4:6]
        
        # Presence loss - for all examples
        presence_loss = self.bce_loss(pred_presence, true_presence)
        
        # For parameters (location, size), only compute loss for positive examples
        # First compute MSE for all
        location_mse = self.mse_loss(pred_location, true_location)
        size_mse = self.mse_loss(pred_size, true_size)
        
        # Only include parameter loss for positive examples
        # Create a mask for positive examples
        positive_mask = true_presence.bool().float()
        
        # Apply mask to parameter losses
        location_loss = (location_mse.mean(dim=1, keepdim=True) * positive_mask).mean()
        size_loss = (size_mse.mean(dim=1, keepdim=True) * positive_mask).mean()
        
        # Combined loss with weighting
        total_loss = presence_loss + 0.5 * location_loss + 0.3 * size_loss
        
        return total_loss, presence_loss, location_loss, size_loss

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    # Define loss function
    criterion = FDEMInversionLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_total_loss = 0
        epoch_presence_loss = 0
        epoch_location_loss = 0
        epoch_size_loss = 0
        
        for freq_data, labels in train_loader:
            # Forward pass
            pred_presence, pred_location, pred_size = model(freq_data)
            
            # Compute losses
            total_loss, presence_loss, location_loss, size_loss = criterion(
                pred_presence, pred_location, pred_size, labels
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_presence_loss += presence_loss.item()
            epoch_location_loss += location_loss.item()
            epoch_size_loss += size_loss.item()
        
        avg_train_loss = epoch_total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_presence_loss = 0
        val_location_loss = 0
        val_size_loss = 0
        
        with torch.no_grad():
            for freq_data, labels in val_loader:
                # Forward pass
                pred_presence, pred_location, pred_size = model(freq_data)
                
                # Compute losses
                total_loss, presence_loss, location_loss, size_loss = criterion(
                    pred_presence, pred_location, pred_size, labels
                )
                
                val_total_loss += total_loss.item()
                val_presence_loss += presence_loss.item()
                val_location_loss += location_loss.item()
                val_size_loss += size_loss.item()
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Train - Total: {epoch_total_loss/len(train_loader):.4f}, '
                  f'Presence: {epoch_presence_loss/len(train_loader):.4f}, '
                  f'Location: {epoch_location_loss/len(train_loader):.4f}, '
                  f'Size: {epoch_size_loss/len(train_loader):.4f}')
            print(f'Val - Total: {val_total_loss/len(val_loader):.4f}, '
                  f'Presence: {val_presence_loss/len(val_loader):.4f}, '
                  f'Location: {val_location_loss/len(val_loader):.4f}, '
                  f'Size: {val_size_loss/len(val_loader):.4f}')
    
    return model, train_losses, val_losses

# Function for real-time inference with complex data
def run_inference(model, freq_data_complex, scaler=None):
    model.eval()
    
    # Split complex data into real and imaginary parts
    if np.iscomplexobj(freq_data_complex):
        real_part = np.real(freq_data_complex)
        imag_part = np.imag(freq_data_complex)
    else:
        real_part = freq_data_complex
        imag_part = np.zeros_like(freq_data_complex)
    
    # Concatenate real and imaginary parts
    freq_data = np.concatenate([real_part, imag_part])
    
    # Normalize data if scaler is provided
    if scaler is not None:
        freq_data = scaler.transform(freq_data.reshape(1, -1)).reshape(freq_data.shape)
    
    # Convert to tensor
    freq_data_tensor = torch.tensor(freq_data, dtype=torch.float32).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        presence_prob, location, size = model(freq_data_tensor)
    
    # Convert to numpy for easier handling
    presence_prob = presence_prob.item()
    location = location.squeeze().numpy()
    size = size.squeeze().numpy()
    
    return {
        'probability': presence_prob,
        'location': {
            'x': location[0],
            'y': location[1],
            'z': location[2]
        },
        'size': {
            'length': size[0],
            'radius': size[1]
        }
    }

# Main execution function
def main():
    # number of samples
    n_sims = 2

    # survey space
    x_min = -4 # m
    x_max = 4 # m
    dx = 0.05 # m
    
    # Default frequency and frequency limits
    min_frequency = 3e2  # 300 Hz 
    max_frequency = 3e4  # 30 kHz

    # Global frequency variable
    freqs = np.linspace(min_frequency, max_frequency, 8)

    # mechanical constants
    dipole_dist = 0.9 # spacing between coils (m)
    sensor_ht = np.random.uniform(0.05, 1.5, n_sims) # height of the sensor above ground (m)

    # random parameters
    conductivity = np.random.uniform(1e4, 1e6, n_sims)
    permeability = 4*np.pi*1e-7 # free space permeability
    xc = np.random.uniform(x_min, x_max, n_sims)
    yc = np.random.uniform(x_min, x_max, n_sims)
    zc = np.random.uniform(0.5, 2, n_sims)
    dincl = np.random.uniform(0, 90, n_sims) # inclination angle in degrees
    ddecl = np.random.uniform(0, 360, n_sims) # declination angle in degrees
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
    
    # Assuming response_list and parameters are already generated
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
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = FDEMDataset(X_train_scaled, y_train)
    val_dataset = FDEMDataset(X_val_scaled, y_val)
    test_dataset = FDEMDataset(X_test_scaled, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize the model with the number of frequencies
    # Input dimension is doubled due to real and imaginary components
    model = FDEM1DInversionNet(num_freqs=len(freqs))
    
    # Train the model
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=150)
    
    # Save the model
    # When saving the model:
    torch.save({
        'model_state_dict': trained_model.state_dict(),
    }, 'fdem_1d_model.pth')

    # Save scaler separately
    import pickle
    with open('fdem_1d_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('1d_complex_training_losses.png')
    plt.show()
    
    # Evaluate on test set
    print("Evaluating on test set...")
    trained_model.eval()
    correct_detections = 0
    total_positive = 0
    total_examples = len(test_dataset)
    
    with torch.no_grad():
        for freq_data, labels in test_loader:
            # Forward pass
            pred_presence, pred_location, pred_size = trained_model(freq_data)
            
            # Calculate detection accuracy
            true_presence = labels[:, 0] > 0.5
            pred_presence_binary = pred_presence.squeeze() > 0.5
            correct_detections += (pred_presence_binary == true_presence).sum().item()
            total_positive += true_presence.sum().item()
    
    print(f"Detection Accuracy: {correct_detections/total_examples:.4f}")
    print(f"Positive Samples: {total_positive}/{total_examples}")
    
    # Code for real-time inference
    print("\nSetup for real-time inference with complex data:")
    print("1. Load the model:")
    print("   checkpoint = torch.load('fdem_1d_complex_inversion_model.pth')")
    print("   model = FDEM1DInversionNet(num_freqs=len(checkpoint['freqs']))")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("   scaler = checkpoint['scaler']")
    print("2. Process real-time frequency data (complex) and run inference:")
    print("   result = run_inference(model, new_freq_data_complex, scaler)")
    print("3. Check result['probability'] to determine if an object is detected")
    print("   If detected, use result['location'] and result['size'] for position and dimensions")

# Call the main function
if __name__ == "__main__":
    main()