"""
Dataset Generator for Electromagnetic Inversion Neural Networks

This script generates a dataset of synthetic electromagnetic responses and
corresponding ground truth parameters, suitable for training neural networks
for electromagnetic inversion.

Usage:
    python generate_em_dataset.py --n_samples 500 --output_dir em_dataset

Arguments:
    --n_samples: Number of samples to generate (default: 500)
    --output_dir: Directory to save the dataset (default: 'em_dataset')
    --use_coupled: Use the more accurate coupled model (slower but more realistic)
    --no_noise: Do not add noise to the data
    --noise_level: Level of Gaussian noise to add (default: 0.03)
    --grid_spacing: Spacing between grid points (default: 0.25)
    --fast_mode: Use reduced grid size for faster generation
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import h5py
import time
import warnings
from create_conductivity_map import create_cylinder_conductivity_map

# Define the functions from the original code
def mind(x, y, z, dincl, ddecl, x0, y0, z0, aincl, adecl):
    """
    Calculate the magnetic field of a magnetic dipole at a given location
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    x0 = np.array(x0, dtype=float)
    y0 = np.array(y0, dtype=float)
    z0 = np.array(z0, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    aincl = np.array(aincl, dtype=float)
    adecl = np.array(adecl, dtype=float)

    di = np.pi * dincl / 180.0
    dd = np.pi * ddecl / 180.0

    cx = np.cos(di) * np.cos(dd)
    cy = np.cos(di) * np.sin(dd)
    cz = np.sin(di)

    ai = np.pi * aincl / 180.0
    ad = np.pi * adecl / 180.0

    ax = np.cos(ai) * np.cos(ad)
    ay = np.cos(ai) * np.sin(ad)
    az = np.sin(ai)

    # begin the calculation
    a = x - x0
    b = y - y0
    h = z - z0

    rt = np.sqrt(a ** 2.0 + b ** 2.0 + h ** 2.0) ** 5.0

    txy = 3.0 * a * b / rt
    txz = 3.0 * a * h / rt
    tyz = 3.0 * b * h / rt

    txx = (2.0 * a ** 2.0 - b ** 2.0 - h ** 2.0) / rt
    tyy = (2.0 * b ** 2.0 - a ** 2.0 - h ** 2.0) / rt
    tzz = -(txx + tyy)

    bx = txx * cx + txy * cy + txz * cz
    by = txy * cx + tyy * cy + tyz * cz
    bz = txz * cx + tyz * cy + tzz * cz

    return bx * ax + by * ay + bz * az

def rotate_coordinates(coords, incl, decl):
    # convert to radians
    incl = np.deg2rad(incl)
    decl = np.deg2rad(decl)

    # Rotation matrix for inclination (around y-axis)
    R_incl = np.array([
        [np.cos(incl), 0, np.sin(incl)],
        [0, 1, 0],
        [-np.sin(incl), 0, np.cos(incl)]
    ])
    
    # Rotation matrix for declination (around z-axis)
    R_decl = np.array([
        [np.cos(decl), -np.sin(decl), 0],
        [np.sin(decl), np.cos(decl), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    rotated_coords = R_decl @ (R_incl @ coords)
    
    return rotated_coords

def fem_pipe(
        sigma, mu, xc, yc, zc, dincl, ddecl, dipole_dist, sensor_ht, f, xmin, xmax, dx, N_loops, length, rad, 
):
    """
    Simulate the frequency domain electromagnetic response of a pipe
    """
    sigma = np.array(sigma, dtype=float)
    mu = np.array(mu, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    dipole_dist = np.array(dipole_dist, dtype=float)
    sensor_ht = np.array(sensor_ht, dtype=float)
    f = np.array(f, dtype=float)
    dx = np.array(dx, dtype=float)
    N_loops = np.array(N_loops, dtype=int)
    length = np.array(length, dtype=float)
    rad = np.array(rad, dtype=float)

    skin_depth = np.sqrt(1 / (np.pi * mu * f * sigma))
    R = N_loops * np.pi * rad * np.sqrt(4*np.pi*mu/sigma) / length
    L = 2*np.pi * 10**(-7) * N_loops * (np.log(8*rad/skin_depth) - 2)

    ymin = xmin
    ymax = xmax
    dy = dx

    # generate the grid
    xp = np.arange(xmin, xmax, dx)
    yp = np.arange(ymin, ymax, dy)
    [y, x] = np.meshgrid(yp, xp)
    z = 0.0 * x - sensor_ht

    # frequency characteristics
    alpha = 2.0 * np.pi * f * L / R
    f_factor = (alpha ** 2.0 + 1j * alpha) / (1 + alpha ** 2.0)

    # compute distances
    y_tx = y - dipole_dist / 2.0
    y_rx = y + dipole_dist / 2.0

    # define the cylinder
    x_centers = np.linspace(-length / 2, length / 2, N_loops) + xc
    loop_centers = np.c_[x_centers, np.zeros(N_loops) + yc, np.zeros(N_loops) + zc]
    loop_centers = rotate_coordinates(loop_centers.T, dincl, ddecl).T
    
    # compute the response
    MTR = mind(0.0, -dipole_dist / 2.0, 0.0, 90.0, 0.0, 0.0, dipole_dist / 2.0, 0.0, 90.0, 0.0)

    mut_ind = 0
    for i in range(N_loops):
        xci, yci, zci = loop_centers[i, 0], loop_centers[i, 1], loop_centers[i, 2]
        MTk = L* mind(x, y_tx, z, 90.0, 0.0, xci, yci, zci, dincl, ddecl)
        MkR = L* mind(xci, yci, zci, dincl, ddecl, x, y_rx, z, 90.0, 0.0)
        mut_ind += MTk * MkR

    c_response = -mut_ind * f_factor / (MTR * L)

    return c_response

def fem_pipe_coupled(
        sigma, mu, xc, yc, zc, dincl, ddecl, dipole_dist, sensor_ht, f, xmin, xmax, dx, N_loops, length, rad, 
):
    """
    Simulate the frequency domain electromagnetic response of a pipe with coupling
    
    This function accounts for coupling between loops, which provides more
    accurate simulations but is significantly slower than fem_pipe().
    """
    sigma = np.array(sigma, dtype=float)
    mu = np.array(mu, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    dipole_dist = np.array(dipole_dist, dtype=float)
    sensor_ht = np.array(sensor_ht, dtype=float)
    freq = np.array(f, dtype=float)
    dx = np.array(dx, dtype=float)
    N_loops = np.array(N_loops, dtype=int)
    length = np.array(length, dtype=float)
    rad = np.array(rad, dtype=float)

    skin_depth = np.sqrt(1 / (np.pi * mu * f * sigma))
    R = N_loops * np.pi * rad * np.sqrt(4*np.pi*mu/sigma) / length
    L = 2*np.pi * 10**(-7) * N_loops * (np.log(8*rad/skin_depth) - 2)

    ymin = xmin
    ymax = xmax
    dy = dx

    # generate the grid
    xp = np.arange(xmin, xmax, dx)
    yp = np.arange(ymin, ymax, dy)
    [y, x] = np.meshgrid(yp, xp)
    z = 0.0 * x - sensor_ht

    # define the cylinder
    x_centers = np.linspace(-length / 2, length / 2, N_loops) + xc
    loop_centers = np.c_[x_centers, np.zeros(N_loops) + yc, np.zeros(N_loops) + zc]
    loop_centers = rotate_coordinates(loop_centers.T, dincl, ddecl).T
    
    # compute the response
    MTR = mind(0.0, -dipole_dist / 2.0, 0.0, 90.0, 0.0, 0.0, dipole_dist / 2.0, 0.0, 90.0, 0.0)

    def solve_hshp(f, x_val, y_val, z_val):
        # compute distances
        y_tx = y_val - dipole_dist / 2.0
        y_rx = y_val + dipole_dist / 2.0

        current_matrix = np.zeros((N_loops, N_loops), dtype=complex)
        transmit_vector = np.zeros((N_loops,1), dtype=complex)
        reciever_vector = np.zeros(N_loops, dtype=complex)
        for i in range(N_loops):
            xci, yci, zci = loop_centers[i, 0], loop_centers[i, 1], loop_centers[i, 2]
            MTi = mind(x_val, y_tx, z_val, 90.0, 0.0, xci, yci, zci, dincl, ddecl)
            MiR = mind(xci, yci, zci, dincl, ddecl, x_val, y_rx, z_val, 90.0, 0.0)
            transmit_vector[i,0] = 1j*2*np.pi*f*MTi
            reciever_vector[i] = MiR
            for j in range(N_loops):
                xcj, ycj, zcj = loop_centers[j, 0], loop_centers[j, 1], loop_centers[j, 2]
                if i == j:
                    current_matrix[i, j] = R + 1j*2*np.pi*f*L
                else:
                    Mij = mind(xci, yci, zci, dincl, ddecl, xcj, ycj, zcj, dincl, ddecl)
                    current_matrix[i, j] = -1j*2*np.pi*f*Mij
        
        # invert the matrix to solve for the currents
        currents = np.linalg.solve(current_matrix, transmit_vector)
        
        return np.sum(reciever_vector * currents) / MTR
    
    def hshp_for_all_pts(f):
        c_response = np.zeros(x.shape, dtype=complex)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                c_response[i,j] = solve_hshp(f, x[i,j], y[i,j], z[i,j])
        return c_response

    if freq.size == 1:
        c_response = hshp_for_all_pts(freq)
    else:
        c_response = np.zeros(freq.size, dtype=complex)
        for nf, f in enumerate(freq):
            c_response[nf] = hshp_for_all_pts(f)

    return c_response

def generate_dataset(n_samples, output_dir, use_coupled=False, add_noise=True, noise_level=0.03, grid_spacing=0.25, fast_mode=False):
    """
    Generate a dataset of simulated EM responses and ground truth parameters
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    output_dir : str
        Directory to save the dataset
    use_coupled : bool
        Whether to use the coupled pipe model (slower but more accurate)
    add_noise : bool
        Whether to add Gaussian noise to the data
    noise_level : float
        Standard deviation of the Gaussian noise
    grid_spacing : float
        Spacing between grid points in meters
    fast_mode : bool
        If True, use reduced grid size for faster generation
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set fixed parameters
    permeability = 4*np.pi*1e-7  # free space permeability
    dipole_dist = 1.0  # 1 m spacing between transmitter and receiver
    sensor_ht = 0.05  # fixed height of the sensor
    
    # Define grid parameters
    if fast_mode:
        x_min, x_max = -2, 3
        n_loops = 20
    else:
        x_min, x_max = -3, 4
        n_loops = 50
    
    dx = grid_spacing
    
    # Define frequencies to simulate
    freqs = np.logspace(3, 5, 20)  # 20 frequencies from 1kHz to 100kHz
    
    # Define arrays to store the dataset
    xp = np.arange(x_min, x_max, dx)
    yp = np.arange(x_min, x_max, dx)
    grid_shape = (len(xp), len(yp))
    
    # Define arrays to store the dataset
    X_real = np.zeros((n_samples, len(freqs), len(xp), len(yp)))
    X_imag = np.zeros((n_samples, len(freqs), len(xp), len(yp)))
    Y = np.zeros((n_samples, 8))  # Store 8 parameters: sigma, xc, yc, zc, dincl, ddecl, length, rad
    
    # Define parameter ranges
    sigma_range = (1e4, 1e6)
    pos_range = (0, 1)
    depth_range = (0.8, 2)
    incl_range = (0, 90)
    decl_range = (0, 360)
    length_range = (0.2, 0.8)
    
    print(f"Generating dataset with {n_samples} samples...")
    print(f"Grid size: {grid_shape[0]}x{grid_shape[1]} points")
    print(f"Frequencies: {freqs}")
    
    start_time = time.time()
    
    # Generate samples
    for i in tqdm(range(n_samples)):
        # Generate random parameters
        conductivity = np.random.uniform(*sigma_range)
        xc = np.random.uniform(*pos_range)
        yc = np.random.uniform(*pos_range)
        zc = np.random.uniform(*depth_range)
        dincl = np.random.uniform(*incl_range)
        ddecl = np.random.uniform(*decl_range)
        length = np.random.uniform(*length_range)
        rad = np.random.uniform(1/6, 1) * length
        
    
        # Store ground truth parameters
        Y[i, :] = [conductivity, xc, yc, zc, dincl, ddecl, length, rad]
        
        # Compute responses for each frequency
        for j, freq in enumerate(freqs):
            try:
                if use_coupled:
                    response = fem_pipe_coupled(
                        conductivity, permeability, xc, yc, zc, dincl, ddecl, 
                        dipole_dist, sensor_ht, freq, x_min, x_max, dx, 
                        n_loops, length, rad
                    )
                else:
                    response = fem_pipe(
                        conductivity, permeability, xc, yc, zc, dincl, ddecl, 
                        dipole_dist, sensor_ht, freq, x_min, x_max, dx, 
                        n_loops, length, rad
                    )
                
                # Add noise if specified
                if add_noise:
                    noise_real = np.random.normal(0, noise_level * np.max(np.abs(np.real(response))), response.shape)
                    noise_imag = np.random.normal(0, noise_level * np.max(np.abs(np.imag(response))), response.shape)
                    response += noise_real + 1j * noise_imag
                
                # Store the response
                X_real[i, j] = np.real(response)
                X_imag[i, j] = np.imag(response)
                
            except np.linalg.LinAlgError:
                warnings.warn(f"Linear algebra error in sample {i}, freq {freq}. Filling with zeros.")
                X_real[i, j] = np.zeros(grid_shape)
                X_imag[i, j] = np.zeros(grid_shape)
                
            except Exception as e:
                warnings.warn(f"Error in sample {i}, freq {freq}: {str(e)}. Filling with zeros.")
                X_real[i, j] = np.zeros(grid_shape)
                X_imag[i, j] = np.zeros(grid_shape)
    
    # Save the dataset
    print(f"Saving dataset to {output_dir}...")
    h5_path = os.path.join(output_dir, 'em_dataset.h5')
    with h5py.File(h5_path, 'w') as f:
        # Create the datasets
        f.create_dataset('input_real', data=X_real)
        f.create_dataset('input_imag', data=X_imag)
        f.create_dataset('target', data=Y)
        
        # Save grid and frequency information
        f.create_dataset('grid_x', data=xp)
        f.create_dataset('grid_y', data=yp)
        f.create_dataset('frequencies', data=freqs)
        
        # Save metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['noise_level'] = noise_level if add_noise else 0
        f.attrs['coupled_model'] = use_coupled
        f.attrs['grid_spacing'] = dx
        f.attrs['parameter_names'] = [
            'conductivity', 'x_position', 'y_position', 'depth', 
            'inclination', 'declination', 'length', 'radius'
        ]
    
    # Save parameter ranges in a separate file for reference
    param_ranges = {
        'conductivity': sigma_range,
        'x_position': pos_range,
        'y_position': pos_range,
        'depth': depth_range,
        'inclination': incl_range,
        'declination': decl_range,
        'length': length_range,
    }
    
    with open(os.path.join(output_dir, 'parameter_ranges.txt'), 'w') as f:
        f.write("Parameter ranges used for dataset generation:\n")
        for param, range_val in param_ranges.items():
            f.write(f"{param}: {range_val}\n")
    
    elapsed_time = time.time() - start_time
    print(f"Dataset generation completed in {elapsed_time/60:.2f} minutes")
    print(f"Dataset saved to {h5_path}")
    
    return h5_path

def visualize_sample(h5_path, sample_idx=0, freq_idx=0):
    """
    Visualize a specific sample from the dataset
    
    Parameters:
    -----------
    h5_path : str
        Path to the HDF5 dataset file
    sample_idx : int
        Index of the sample to visualize
    freq_idx : int
        Index of the frequency to visualize
    """
    with h5py.File(h5_path, 'r') as f:
        # Load data
        real_data = f['input_real'][sample_idx, freq_idx]
        imag_data = f['input_imag'][sample_idx, freq_idx]
        parameters = f['target'][sample_idx]
        grid_x = f['grid_x'][:]
        grid_y = f['grid_y'][:]
        freqs = f['frequencies'][:]
        param_names = f.attrs['parameter_names']
        
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot real component
    im1 = ax1.imshow(real_data, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()], 
                    aspect='auto', cmap='viridis')
    ax1.set_title(f'Real Component at {freqs[freq_idx]/1000:.1f} kHz')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    plt.colorbar(im1, ax=ax1, label='Response Magnitude')
    
    # Plot imaginary component
    im2 = ax2.imshow(imag_data, extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()],
                    aspect='auto', cmap='viridis')
    ax2.set_title(f'Imaginary Component at {freqs[freq_idx]/1000:.1f} kHz')
    ax2.set_xlabel('X (m)')
    plt.colorbar(im2, ax=ax2, label='Response Magnitude')
    
    # Add annotation with parameters
    param_str = "\n".join([f"{param_names[i]}: {parameters[i]:.4f}" for i in range(len(parameters))])
    fig.text(0.02, 0.02, param_str, fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.suptitle(f"Sample {sample_idx} Response", y=1.05)
    plt.show()
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate EM inversion training dataset')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='em_dataset', help='Directory to save the dataset')
    parser.add_argument('--use_coupled', action='store_true', help='Use the coupled pipe model (slower but more accurate)')
    parser.add_argument('--no_noise', action='store_true', help='Do not add noise to the data')
    parser.add_argument('--noise_level', type=float, default=0.03, help='Level of Gaussian noise to add')
    parser.add_argument('--grid_spacing', type=float, default=0.25, help='Spacing between grid points')
    parser.add_argument('--fast_mode', action='store_true', help='Use reduced grid size for faster generation')
    parser.add_argument('--visualize', action='store_true', help='Visualize a sample after generation')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--freq_idx', type=int, default=0, help='Frequency index to visualize')
    
    args = parser.parse_args()
    
    h5_path = generate_dataset(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        use_coupled=args.use_coupled,
        add_noise=not args.no_noise,
        noise_level=args.noise_level,
        grid_spacing=args.grid_spacing,
        fast_mode=args.fast_mode
    )
    
    if args.visualize:
        visualize_sample(h5_path, args.sample_idx, args.freq_idx)