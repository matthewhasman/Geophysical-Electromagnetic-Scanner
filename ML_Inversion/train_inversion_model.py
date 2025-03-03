"""
PyTorch Neural Network for Electromagnetic Inversion

This script builds and trains a PyTorch model for inverting
electromagnetic responses back to physical parameters.

Usage:
    python train_inversion_model_pytorch.py --dataset em_dataset/em_dataset.h5 --model_dir models

Arguments:
    --dataset: Path to the HDF5 dataset file
    --model_dir: Directory to save the trained model
    --epochs: Number of training epochs (default: 100)
    --batch_size: Batch size for training (default: 16)
    --val_split: Validation split ratio (default: 0.2)
    --test_split: Test split ratio (default: 0.1)
    --lr: Learning rate (default: 0.001)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
from datasets.em_dataset import EMDataset, load_dataset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeDistributed(nn.Module):
    """
    Applies a module to each timestep of a sequence
    """
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # x shape: (batch_size, time_steps, *input_shape)
        batch_size, time_steps = x.size(0), x.size(1)
        
        # Reshape to (batch_size * time_steps, *input_shape)
        x_reshaped = x.contiguous().view(batch_size * time_steps, *x.size()[2:])
        
        # Apply module
        y = self.module(x_reshaped)
        
        # Reshape back to (batch_size, time_steps, *output_shape)
        y = y.contiguous().view(batch_size, time_steps, *y.size()[1:])
        
        return y

class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for electromagnetic inversion
    """
    def __init__(self, input_shape, n_outputs):
        """
        Initialize the model
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data (n_freqs, height, width, 2)
        n_outputs : int
            Number of parameters to predict
        """
        super(CNNLSTMModel, self).__init__()
        
        n_freqs, height, width, channels = input_shape
        
        # CNN feature extractor (applied to each frequency)
        self.cnn_layers = nn.Sequential(
            # First convolutional block
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            
            # Flatten
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            # Create a dummy input to get output size
            dummy_input = torch.zeros(1, channels, height, width)
            cnn_output = self.cnn_layers(dummy_input)
            cnn_output_size = cnn_output.size(1)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Dense layers for parameter prediction
        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_outputs),
            nn.Sigmoid()  # Outputs normalized parameters
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_freqs, height, width, 2)
        
        Returns:
        --------
        out : torch.Tensor
            Output tensor of shape (batch_size, n_outputs)
        """
        batch_size, n_freqs, height, width, channels = x.size()
        
        # Process each frequency with CNN
        cnn_outputs = []
        for i in range(n_freqs):
            # Extract current frequency data and permute for CNN
            freq_data = x[:, i]  # shape: (batch_size, height, width, channels)
            freq_data = freq_data.permute(0, 3, 1, 2)  # shape: (batch_size, channels, height, width)
            
            # Apply CNN
            cnn_out = self.cnn_layers(freq_data)  # shape: (batch_size, cnn_output_size)
            cnn_outputs.append(cnn_out.unsqueeze(1))  # Add time dimension
        
        # Concatenate along time dimension
        cnn_outputs = torch.cat(cnn_outputs, dim=1)  # shape: (batch_size, n_freqs, cnn_output_size)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(cnn_outputs)  # shape: (batch_size, n_freqs, lstm_hidden_size)
        
        # Take last time step output
        lstm_out = lstm_out[:, -1, :]  # shape: (batch_size, lstm_hidden_size)
        
        # Apply dense layers
        out = self.fc_layers(lstm_out)  # shape: (batch_size, n_outputs)
        
        return out

def train_model(data_dict, model_dir, epochs=100, batch_size=16, lr=0.001, plot_history=True):
    """
    Train the PyTorch model
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the data splits and metadata
    model_dir : str
        Directory to save the trained model
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    plot_history : bool
        Whether to plot the training history
    
    Returns:
    --------
    model : nn.Module
        Trained model
    history : dict
        Training history
    """
    # Create model directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Create dataloaders
    train_loader = DataLoader(
        data_dict['train_dataset'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        data_dict['val_dataset'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    input_shape = data_dict['input_shape']
    n_outputs = data_dict['n_outputs']
    
    model = CNNLSTMModel(input_shape, n_outputs)
    model = model.to(device)
    
    print("Model architecture:")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': []
    }
    
    # Initialize best validation loss
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Calculate MAE
            mae = torch.mean(torch.abs(outputs - targets))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_mae += mae.item()
            train_batches += 1
        
        # Calculate average metrics
        train_loss /= train_batches
        train_mae /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate MAE
                mae = torch.mean(torch.abs(outputs - targets))
                
                # Update metrics
                val_loss += loss.item()
                val_mae += mae.item()
                val_batches += 1
        
        # Calculate average metrics
        val_loss /= val_batches
        val_mae /= val_batches
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # Print progress
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
              f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Check if current model is best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, os.path.join(model_dir, 'best_model.pt'))
            
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            
            # Check for early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model and training history
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_mae': val_mae
    }, os.path.join(model_dir, 'final_model.pt'))
    
    # Save history and normalization parameters
    np.save(os.path.join(model_dir, 'training_history.npy'), history)
    np.save(os.path.join(model_dir, 'Y_min.npy'), data_dict['Y_min'])
    np.save(os.path.join(model_dir, 'Y_max.npy'), data_dict['Y_max'])
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training history
    if plot_history:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training and Validation MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_history.png'))
        plt.show()
    
    return model, history

def evaluate_model(model, data_dict, model_dir):
    """
    Evaluate the trained model on the test set
    
    Parameters:
    -----------
    model : nn.Module
        Trained model
    data_dict : dict
        Dictionary containing the data splits and metadata
    model_dir : str
        Directory to save evaluation results
    """
    # Create test dataloader
    test_loader = DataLoader(
        data_dict['test_dataset'],
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get normalization parameters
    Y_min = data_dict['Y_min']
    Y_max = data_dict['Y_max']
    param_names = data_dict['param_names']
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    test_loss = 0.0
    test_mae = 0.0
    test_batches = 0
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate MSE loss
            loss = nn.MSELoss()(outputs, targets)
            
            # Calculate MAE
            mae = torch.mean(torch.abs(outputs - targets))
            
            # Update metrics
            test_loss += loss.item()
            test_mae += mae.item()
            test_batches += 1
            
            # Store predictions and targets
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate average metrics
    test_loss /= test_batches
    test_mae /= test_batches
    
    # Combine all predictions and targets
    Y_pred_norm = np.vstack(all_predictions)
    Y_test_norm = np.vstack(all_targets)
    
    # Denormalize predictions and ground truth
    Y_pred = Y_pred_norm * (Y_max - Y_min) + Y_min
    Y_true = Y_test_norm * (Y_max - Y_min) + Y_min
    
    # Calculate error metrics for each parameter
    mae_per_param = np.mean(np.abs(Y_pred - Y_true), axis=0)
    mape_per_param = np.mean(np.abs((Y_pred - Y_true) / (Y_true + 1e-10)) * 100, axis=0)
    
    # Print error metrics
    print("\nTest Results:")
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    print("\nParameter-wise evaluation:")
    print("--------------------------")
    for i, param in enumerate(param_names):
        print(f"{param}:")
        print(f"  MAE: {mae_per_param[i]:.4f}")
        print(f"  MAPE: {mape_per_param[i]:.2f}%")
    
    # Plot prediction vs ground truth for each parameter
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 4*n_params))
    
    if n_params == 1:
        axes = [axes]  # Make axes iterable when there's only one parameter
    
    for i, (param, ax) in enumerate(zip(param_names, axes)):
        ax.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(np.min(Y_true[:, i]), np.min(Y_pred[:, i]))
        max_val = max(np.max(Y_true[:, i]), np.max(Y_pred[:, i]))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_xlabel(f'True {param}')
        ax.set_ylabel(f'Predicted {param}')
        ax.set_title(f'{param}: MAE = {mae_per_param[i]:.4f}, MAPE = {mape_per_param[i]:.2f}%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'parameter_predictions.png'))
    plt.show()
    
    # Save evaluation results
    results = {
        'test_loss': test_loss,
        'test_mae': test_mae,
        'mae_per_param': mae_per_param,
        'mape_per_param': mape_per_param,
    }
    np.save(os.path.join(model_dir, 'evaluation_results.npy'), results)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train electromagnetic inversion model with PyTorch')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the HDF5 dataset file')
    parser.add_argument('--model_dir', type=str, default='em_model_pytorch', help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='Test split ratio')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Load and preprocess the dataset
    data_dict = load_dataset(args.dataset, args.val_split, args.test_split)
    
    # Train the model
    model, history = train_model(
        data_dict,
        args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # Evaluate the model
    evaluate_model(model, data_dict, args.model_dir)
    ## Example Usage:
    # python train_inversion_model.py --dataset em_dataset/em_dataset.h5 --model_dir em_model_pytorch --epochs 150 --batch_size 16 --lr 0.001