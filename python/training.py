import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
import os

# Define a custom dataset class for FDEM data that handles complex values and adds random noise
class FDEMDataset(Dataset):
    def __init__(self, frequency_data, labels, noise_level=0.1):
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
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cpu'):
    # Define loss function
    criterion = FDEMInversionLoss()
    model = model.to(device)
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
            freq_data = freq_data.to(device)
            labels = labels.to(device)
            
            pred_presence, pred_location, pred_size = model(freq_data)
            
            # Compute losses
            total_loss, presence_loss, location_loss, size_loss = criterion(
                pred_presence, pred_location, pred_size, labels
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            ## log the loss 
            print(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {total_loss.item():.4f}", end='\r')
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
                freq_data = freq_data.to(device)
                labels = labels.to(device)
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
    # Check if the dataset file exists
    dataset_file = 'fdem_dataset_with_background.pkl'
    scaler_file = 'fdem_scaler_background.pkl'
    
    if not os.path.exists(dataset_file):
        print(f"Error: Dataset file '{dataset_file}' not found.")
        print("Please run the dataset generation script first.")
        return
    
    # Load the dataset
    print(f"Loading dataset from '{dataset_file}'...")
    with open(dataset_file, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Load the scaler
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    # Extract data
    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    freqs = data_dict['freqs']
    
    print(f"Dataset loaded. Training samples: {len(X_train)}")
    
    # Create datasets
    train_dataset = FDEMDataset(X_train, y_train)
    val_dataset = FDEMDataset(X_val, y_val)
    test_dataset = FDEMDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=1024)
    
    # Initialize the model with the number of frequencies
    # Input dimension is doubled due to real and imaginary components
    num_freqs = len(freqs)
    print(f"Creating model for {num_freqs} frequencies...")
    model = FDEM1DInversionNet(num_freqs=num_freqs, hidden_size=256)

    ## check which device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    
    # Train the model
    print("Starting model training...")
    trained_model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=150, device=device)
    
    # Save the model
    model_save_path = 'fdem_1d_model.pth'
    print(f"Saving model to '{model_save_path}'...")
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'freqs': freqs
    }, model_save_path)
    
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
    trained_model.to('cpu')
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
    print("   checkpoint = torch.load('fdem_1d_model.pth')")
    print("   model = FDEM1DInversionNet(num_freqs=len(checkpoint['freqs']))")
    print("   model.load_state_dict(checkpoint['model_state_dict'])")
    print("   with open('fdem_scaler.pkl', 'rb') as f:")
    print("       scaler = pickle.load(f)")
    print("2. Process real-time frequency data (complex) and run inference:")
    print("   result = run_inference(model, new_freq_data_complex, scaler)")
    print("3. Check result['probability'] to determine if an object is detected")
    print("   If detected, use result['location'] and result['size'] for position and dimensions")

if __name__ == "__main__":
    main()