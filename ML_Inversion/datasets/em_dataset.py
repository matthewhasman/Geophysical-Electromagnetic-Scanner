import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split



class EMDataset(Dataset):
    """
    Dataset class for electromagnetic inversion data
    """
    def __init__(self, X, Y):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_freqs, height, width, 2)
        Y : numpy.ndarray
            Target parameters (n_samples, n_params)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def load_dataset(dataset_path, val_split=0.2, test_split=0.1):
    """
    Load the dataset from an HDF5 file and split into train/val/test sets
    
    Parameters:
    -----------
    dataset_path : str
        Path to the HDF5 dataset file
    val_split : float
        Proportion of data to use for validation
    test_split : float
        Proportion of data to use for testing
    
    Returns:
    --------
    data_dict : dict
        Dictionary containing the data splits and metadata
    """
    with h5py.File(dataset_path, 'r') as f:
        # Load input data
        X_real = f['input_real'][:]
        X_imag = f['input_imag'][:]
        
        # Combine real and imaginary parts
        n_samples, n_freqs, height, width = X_real.shape
        X = np.stack([X_real, X_imag], axis=-1)  # Shape: (n_samples, n_freqs, height, width, 2)
        
        # Load target parameters
        Y = f['target'][:]
        
        # Load metadata
        grid_x = f['grid_x'][:]
        grid_y = f['grid_y'][:]
        frequencies = f['frequencies'][:]
        
        # Get parameter names if available
        param_names = f.attrs.get('parameter_names', 
                                  ['conductivity', 'x_position', 'y_position', 'depth', 
                                   'inclination', 'declination', 'length', 'radius'])
    
    # Convert parameter names to list if needed
    if not isinstance(param_names, list):
        param_names = [name.decode() if isinstance(name, bytes) else name for name in param_names]
    
    # Normalize the target parameters
    Y_min = np.min(Y, axis=0)
    Y_max = np.max(Y, axis=0)
    Y_norm = (Y - Y_min) / (Y_max - Y_min + 1e-10)
    
    # Split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y_norm, test_size=val_split+test_split, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_split/(val_split+test_split), random_state=42)
    
    # Create datasets
    train_dataset = EMDataset(X_train, Y_train)
    val_dataset = EMDataset(X_val, Y_val)
    test_dataset = EMDataset(X_test, Y_test)
    
    # Create data dictionary
    data_dict = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'Y_min': Y_min,
        'Y_max': Y_max,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'frequencies': frequencies,
        'param_names': param_names,
        'input_shape': X_train.shape[1:],
        'n_outputs': Y_train.shape[1]
    }
    
    print(f"Dataset loaded: {n_samples} samples, {n_freqs} frequencies, {height}x{width} grid")
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return data_dict

if __name__ == "__main__":
    ## Testing Dataset Functions when running as main
    data_dict = load_dataset("em_dataset/em_dataset.h5")
    # Get a sample datapoint from the training dataset
    sample_idx = 0  # Get the first sample
    sample_X, sample_Y_norm = data_dict['train_dataset'][sample_idx]
    
    # Denormalize the target parameters to get original values
    Y_min = data_dict['Y_min']
    Y_max = data_dict['Y_max']

    sample_Y = sample_Y_norm.numpy() * (Y_max - Y_min) + Y_min
    
    # Print parameter names and values
    param_names = data_dict['param_names']
    print("\nSample datapoint parameters:")
    for i, param_name in enumerate(param_names):
        print(f"{param_name}: {sample_Y[i]}")
    
    # Print input shape information
    print(f"\nInput shape: {sample_X.shape}")
    print(f"Number of parameters: {len(param_names)}")