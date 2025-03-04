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

class MaskedEMDataset(Dataset):
    """
    Dataset class for electromagnetic inversion data with random masking
    """
    def __init__(self, X, Y, mask_prob=0.2, mask_size_range=(0.1, 0.3)):
        """
        Initialize the dataset with random masking
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_freqs, height, width, 2)
        Y : numpy.ndarray
            Target parameters (n_samples, n_params)
        mask_prob : float
            Probability of applying a mask to each sample
        mask_size_range : tuple
            Range of mask sizes as a fraction of the input dimensions (min, max)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.mask_prob = mask_prob
        self.mask_size_range = mask_size_range
    
    def __len__(self):
        return len(self.X)
    
    def apply_random_mask(self, x):
        """
        Apply random rectangular mask to the input data
        
        Parameters:
        -----------
        x : torch.Tensor
            Input data tensor of shape (n_freqs, height, width, 2)
            
        Returns:
        --------
        masked_x : torch.Tensor
            Masked input data
        """
        # Make a copy of the input
        masked_x = x.clone()
        
        # Get dimensions
        n_freqs, height, width, _ = x.shape
        
        # Determine mask size
        min_size, max_size = self.mask_size_range
        mask_height = int(np.random.uniform(min_size, max_size) * height)
        mask_width = int(np.random.uniform(min_size, max_size) * width)
        
        # Determine mask position
        top = np.random.randint(0, height - mask_height + 1)
        left = np.random.randint(0, width - mask_width + 1)
        
        # Apply mask (set values to zero)
        # Randomly choose which frequencies to mask (can be all or a subset)
        freqs_to_mask = np.random.randint(1, n_freqs + 1)  # At least one frequency
        freq_indices = np.random.choice(n_freqs, freqs_to_mask, replace=False)
        
        for freq_idx in freq_indices:
            masked_x[freq_idx, top:top+mask_height, left:left+mask_width, :] = 0.0
            
        return masked_x
    
    def __getitem__(self, idx):
        x = self.X[idx]
        
        # Apply mask with probability mask_prob
        if np.random.random() < self.mask_prob:
            x = self.apply_random_mask(x)
            
        return x, self.Y[idx]

def load_dataset(dataset_path, val_split=0.2, test_split=0.1, masked=False):
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
    if masked == False:
        train_dataset = EMDataset(X_train, Y_train)
        val_dataset = EMDataset(X_val, Y_val)
        test_dataset = EMDataset(X_test, Y_test)

    else:
        train_dataset = MaskedEMDataset(X_train, Y_train, mask_prob=0.2, mask_size_range=(0.1, 0.3))
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