import numpy as np
import torch
import h5py
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from .create_conductivity_map import create_cylinder_conductivity_map
from tqdm import tqdm

class ConductivityMapDataset(Dataset):
    """
    Dataset class for 3D conductivity map data
    """
    def __init__(self, X, Y):
        """
        Initialize the dataset
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data (EM responses)
        Y : numpy.ndarray
            Target data (3D conductivity maps)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_conductivity_dataset(dataset_path, val_split=0.2, test_split=0.1):
    """
    Load the conductivity map dataset from an HDF5 file and split into train/val/test sets
    
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


    # Define grid parameters
    grid_params = {
        'x_range': (-5, 5),    # X range (m)
        'y_range': (-5, 5),    # Y range (m)
        'z_range': (0, 5),      # Z range (m)
        'grid_spacing': 0.1      # Grid spacing (m)
    }

    with h5py.File(dataset_path, 'r') as f:
        # Load input data (EM responses)
        X_real = f['input_real'][:]
        X_imag = f['input_imag'][:]
        
        # Combine real and imaginary parts
        n_samples, n_freqs, height, width = X_real.shape
        X = np.stack([X_real, X_imag], axis=-1)  # Shape: (n_samples, n_freqs, height, width, 2)
        
        # Load target data and then generate a 3d conductivty map
        Y = f['target'][:]        
                # Get parameter names if available
        param_names = f.attrs.get('parameter_names', 
                                  ['conductivity', 'x_position', 'y_position', 'depth', 
                                   'inclination', 'declination', 'length', 'radius'])
        
        # Convert parameter names to list if needed
        if not isinstance(param_names, list):
            param_names = [name.decode() if isinstance(name, bytes) else name for name in param_names]
        
        # Create a dictionary mapping parameter names to their indices
        param_indices = {name: i for i, name in enumerate(param_names)}
        
        # Create cylinder parameters dictionary
        cylinder_params = {
            'conductivity': Y[:, param_indices['conductivity']],
            'x_position': Y[:, param_indices['x_position']],
            'y_position': Y[:, param_indices['y_position']],
            'depth': Y[:, param_indices['depth']],
            'inclination': Y[:, param_indices['inclination']],
            'declination': Y[:, param_indices['declination']],
            'length': Y[:, param_indices['length']],
            'radius': Y[:, param_indices['radius']]
        }

        Y_map = []
        for i in tqdm(range(len(Y)), desc="Generating conductivity maps", unit="sample"):
            Y_map_next, _ = create_cylinder_conductivity_map(
                depth=cylinder_params['depth'][i],
                inclination=cylinder_params['inclination'][i],
                declination=cylinder_params['declination'][i],
                radius=cylinder_params['radius'][i],
                length=cylinder_params['length'][i],
                conductivity=cylinder_params['conductivity'][i],
                background_conductivity=0.001,
                x_range=grid_params['x_range'],
                y_range=grid_params['y_range'],
                z_range=grid_params['z_range'],
                grid_spacing=grid_params['grid_spacing']
            )
            Y_map.append(Y_map_next)

        Y_map = np.array(Y_map)
        # Load metadata
        grid_x = f['grid_x'][:]
        grid_y = f['grid_y'][:]
        grid_z = f['grid_z'][:] if 'grid_z' in f else None
        frequencies = f['frequencies'][:]
    
    # Split the data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y_map, test_size=val_split+test_split, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=test_split/(val_split+test_split), random_state=42)
    
    # Create datasets
    train_dataset = ConductivityMapDataset(X_train, Y_train)
    val_dataset = ConductivityMapDataset(X_val, Y_val)
    test_dataset = ConductivityMapDataset(X_test, Y_test)
    
    # Create data dictionary
    data_dict = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
        'frequencies': frequencies,
        'input_shape': X_train.shape[1:],
        'output_shape': Y_train.shape[1:]
    }
    
    print(f"Conductivity Map Dataset loaded: {n_samples} samples")
    print(f"Input shape: {X_train.shape[1:]}, Output shape: {Y_train.shape[1:]}")
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return data_dict


if __name__ == "__main__":
    import argparse
    import os
    import pickle
    from tqdm import tqdm
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process and cache conductivity dataset")
    parser.add_argument("--dataset", type=str, default="em_dataset/em_dataset.h5", help="Path to the HDF5 dataset file")
    parser.add_argument("--output", type=str, default=None, help="Path to save the processed dataset (default: dataset_path + .cache)")
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        args.output = args.dataset + ".cache"
    
    print(f"Processing dataset: {args.dataset}")
    print(f"Cache will be saved to: {args.output}")
    
    # Load and process the dataset
    data_dict = load_conductivity_dataset(args.dataset)
    
    # Save the processed dataset
    print(f"Saving processed dataset to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"Dataset successfully cached to {args.output}")
    
    # Get a sample datapoint from the training dataset for verification
    sample_idx = 0  # Get the first sample
    sample_X, sample_Y = data_dict['train_dataset'][sample_idx]
    
    # Print shape information
    print(f"\nInput EM data shape: {sample_X.shape}")
    print(f"Output conductivity map shape: {sample_Y.shape}")
    
    # Print some statistics about the conductivity map
    print(f"\nConductivity map statistics:")
    print(f"Min value: {torch.min(sample_Y).item()}")
    print(f"Max value: {torch.max(sample_Y).item()}")
    print(f"Mean value: {torch.mean(sample_Y).item()}")