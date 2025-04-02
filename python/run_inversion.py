import os
import numpy as np
import pandas as pd
import glob
from scipy.linalg import svd
from scipy.ndimage import laplace
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


def load_exported_csv_data(file_path):
    """
    Load data from CSV files exported by the export_data_csv function
    
    Parameters:
        file_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary containing extracted data
    """
    print(f"Loading data from {file_path}")
    
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Extract column headers to identify frequencies
    headers = df.columns.tolist()
    
    # Extract timestamps
    timestamps = df['Timestamp (s)'].values
    
    # Extract frequencies from column headers
    frequencies = []
    magnitude_data = []
    phase_data = []
    hshp_real_data = []
    hshp_imag_data = []
    
    # Parse column headers to extract frequencies and corresponding data
    for col in headers[1:]:  # Skip timestamp column
        if 'Magnitude' in col:
            # Extract frequency from column name, e.g., 'Magnitude (1.0e+02 Hz)'
            freq_str = col.split('(')[1].split(' Hz')[0]
            freq = float(freq_str.replace('e+', 'e').replace('e-', 'e-'))
            
            if freq not in frequencies:
                frequencies.append(freq)
            
            # Get the corresponding data
            magnitude_data.append(df[col].values)
        
        elif 'Phase' in col:
            phase_data.append(df[col].values)
        
        elif 'HsHp Real' in col:
            hshp_real_data.append(df[col].values)
        
        elif 'HsHp Imag' in col:
            hshp_imag_data.append(df[col].values)
    
    # Organize data by frequencies
    data = {
        'timestamps': timestamps,
        'frequencies': np.array(frequencies),
        'magnitude_data': np.array(magnitude_data).T,  # Transpose to have shape (num_soundings, num_freqs)
        'phase_data': np.array(phase_data).T,
        'hshp_real_data': np.array(hshp_real_data).T,
        'hshp_imag_data': np.array(hshp_imag_data).T
    }
    
    return data

def reflfact(lambda_vals, sigma, mu, d, omega):
    """
    Compute reflection factor for EM fields
    """
    lambda_vals = np.array(lambda_vals).flatten()
    sigma = np.array(sigma).flatten()
    mu = np.array(mu).flatten()
    omega = np.array(omega).flatten()

    n = len(sigma)     # Number of layers
    q = len(lambda_vals)
    nfreq = len(omega)  # Number of frequencies
    mu0 = np.pi * 4e-7  # Permeability of free space

    # Compute constants
    den0 = 1j * mu0 * omega
    densig = np.outer(mu, omega) + 1j * 1e-12
    frac = -1j / (densig + 1e-12)
    densig = sigma[:, None] * densig

    # Initialize Y (last layer)
    lam2 = np.tile(lambda_vals[:, None] ** 2, (1, nfreq))
    u = np.sqrt(lam2 + 1j * np.tile(densig[-1, :], (q, 1)))
    Y = u * np.tile(frac[-1, :], (q, 1))

    # Backward recurrence to compute Y for all layers
    for k in range(n-2, -1, -1):
        u = np.sqrt(lam2 + 1j * np.tile(densig[k, :], (q, 1)))
        N = u * np.tile(frac[k, :], (q, 1))
        arg = np.minimum(np.abs(u * d[k]), 300) * np.sign(u * d[k])
        tan_ip = np.tanh(arg)

        Aden = N + Y * tan_ip
        Aden = np.where(np.abs(Aden) < 1e-12, 1e-12 + 1j * 1e-12, Aden)
        A = (Y + N * tan_ip) / Aden
        Y = N * A  # Update Y for layer k

    # Compute reflection factor R0
    N0 = lambda_vals[:, None] / den0
    R0 = (N0 - Y) / (N0 + Y)
    return R0

def hankelpts():
    with open("../inversion/hankelpts_values.txt", "r") as file:
        vals = [float(line.strip()) for line in file]
    return np.array(vals)

def hankelwts():
    with open("../inversion/hankelwts_wt0.txt", "r") as file:
        wt0 =[float(line.strip()) for line in file]
    with open("../inversion/hankelwts_wt1.txt", "r") as file:
        wt1 =[float(line.strip()) for line in file]
    return np.array(wt0), np.array(wt1)

def aconduct(sigma, mu, d, omega, R, h):
    """
    Compute the apparent conductivity for 'horizontal' orientation
    
    Parameters:
        sigma  - array of conductivities per layer (S/m)
        mu     - array of magnetic permeabilities per layer (H/m)
        h      - array of heights above ground (m)
        d      - array of layers thickness (m)
        R      - array of intercoil (IC) spacings (m)
        omega  - array of angular frequencies (rad/s)
    
    Returns:
        mp  - Predicted readings with coils horizontally oriented
    """
    sigma = np.array(sigma).flatten()
    mu = np.array(mu).flatten()
    R = np.array(R).flatten()
    omega = np.array(omega).flatten()
    n = len(sigma)
    m = len(h)
    nR = len(R)
    nfreq = len(omega)
    nhf = m * nfreq
    mu0 = np.pi * 4e-7
    
    w0, w1 = hankelwts()  # Get weights for Hankel transform
    y = hankelpts()  # Get nodes for Hankel transform
    
    T2 = np.zeros((m, nfreq))
    mp = np.zeros(nR * nhf)
    
    for k in range(nR):
        yh = y / R[k]
        rf = reflfact(yh, sigma, mu, d, omega)
        rf = np.imag(rf)
        rf *= yh[:, None]
        for i in range(m):
            f = rf * np.exp(-2 * yh[:, None] * h[i])
            T2[i, :] = np.dot(w1, f) / R[k]

        f1 = 4 / (mu0 * omega)
        mht = -np.outer(f1, np.ones(m)).T * T2
        mp[k * nhf:(k + 1) * nhf] = mht.T.flatten()
    return mp

def compute_apparent_conductivity(sigma, frequencies, dipole_dist, sensor_ht):
    """
    Compute apparent conductivity based on layer conductivities
    
    Parameters:
        sigma (array): Layer conductivities
        frequencies (array): Frequency values
        dipole_dist (float): Distance between dipoles
        sensor_ht (float): Sensor height
    
    Returns:
        array: Apparent conductivity values
    """
    num_layers = len(sigma)
    mu = np.ones(num_layers) * (4 * np.pi * 1e-7)
    d = np.ones(num_layers) * 5.0  # Layer thickness in meters
    d[-1] = np.inf  # Bottom layer extends to infinity
    
    omega = 2 * np.pi * frequencies
    
    B = aconduct(sigma, mu, d, omega, np.array([dipole_dist]), np.array([sensor_ht]))
    return B

def compute_jacobian(sigma, frequencies, dipole_dist, sensor_ht):
    """
    Computes the Jacobian matrix using finite differences
    
    Parameters:
        sigma (array): Layer conductivities
        frequencies (array): Frequency values
        dipole_dist (float): Distance between dipoles
        sensor_ht (float): Sensor height
    
    Returns:
        array: Jacobian matrix
    """
    delta = 1e-4  # Small perturbation
    num_layers = len(sigma)
    num_freqs = len(frequencies)

    # Initialize Jacobian matrix
    J = np.zeros((num_freqs, num_layers))

    # Compute baseline apparent conductivity
    B = compute_apparent_conductivity(sigma, frequencies, dipole_dist, sensor_ht)
    B = np.asarray(B).flatten()

    for i in range(num_layers):
        sigma_perturbed = sigma.copy()
        sigma_perturbed[i] += delta  # Apply small perturbation

        # Compute perturbed response
        B_plus = compute_apparent_conductivity(sigma_perturbed, frequencies, dipole_dist, sensor_ht)
        B_plus = np.asarray(B_plus).flatten()

        # Compute finite difference approximation
        J[:, i] = (B_plus - B) / delta

    return J

def optimize_alpha(r, Q_GSVD, beta):
    """
    Optimizes the alpha parameter based on given inputs
    
    Parameters:
        r (array): Residual vector
        Q_GSVD (array): GSVD projection vector
        beta (float): Regularization parameter
    
    Returns:
        float: Optimized alpha value
    """
    numerator = np.linalg.norm(r) ** 2
    denominator = np.linalg.norm(Q_GSVD) ** 2 + beta
    
    # Prevent division by very small values
    denominator = max(denominator, 1e-8)
    
    # Compute alpha
    alpha = numerator / denominator
    
    # Enforce non-negativity
    return max(alpha, 0)

def compute_u_k(xi, sigma, q):
    """
    Computes the regularization term u_k
    
    Parameters:
        xi (array): Parameter for regularization
        sigma (array): Conductivity distribution
        q (float): Exponent for scaling
    
    Returns:
        array: Regularization term u_k
    """
    # Compute Laplacian for smoothness
    grad = laplace(sigma.reshape(-1, 1)).flatten()
    
    # Compute u_k
    u_k = grad * ((xi + grad**2 + 1e-2) ** (q / 2 - 1))
    
    return u_k

def update_auxiliary_variable(u_k, sigma_k, beta):
    """
    Update auxiliary variable for the inversion process
    
    Parameters:
        u_k (array): Regularization term
        sigma_k (array): Current conductivity distribution
        beta (float): Regularization parameter
    
    Returns:
        array: Updated auxiliary variable
    """
    # Simple update rule (can be replaced with more sophisticated algorithms)
    xi_updated = sigma_k + beta * u_k
    
    # Ensure non-negative values
    xi_updated = np.maximum(xi_updated, 0)
    
    return xi_updated

def run_geophysical_inversion(csv_file, output_dir='inversion_results'):
    """
    Run geophysical inversion on data from a CSV file
    
    Parameters:
        csv_file (str): Path to the CSV file
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from CSV
    data = load_exported_csv_data(csv_file)
    
    # Extract required information
    frequencies = data['frequencies']
    
    # Setup inversion parameters
    # For this example, we'll use the magnitude data as apparent conductivity
    apparent_conductivity = data['magnitude_data']
    
    # Setup inversion parameters
    gamma = 1e0  # Regularization base parameter
    q = 0.9  # Hyperparameter for smoothness
    ell = 5  # GSVD truncation level
    epsilon = 1e-3  # Convergence tolerance
    max_iters = 50  # Maximum iterations
    num_layers = 30  # Number of layers for the model
    
    # Sensor parameters (these would usually be provided or derived from the data)
    dipole_dist = 1.0  # Distance between transmitter and receiver dipoles (meters)
    sensor_ht = 0.1  # Height of the sensor above the ground (meters)
    
    # Get dimensions
    num_soundings, num_freqs = apparent_conductivity.shape
    
    # Initialize model
    sigma_k = np.ones((num_layers, num_soundings))
    xi_k = sigma_k.copy()  # Initial auxiliary variable
    beta = gamma / (2 * np.linalg.norm(apparent_conductivity, 'fro')**2)
    
    # Create a results dictionary to store the inversion results
    results = {
        'frequencies': frequencies,
        'timestamps': data['timestamps'],
        'apparent_conductivity': apparent_conductivity,
        'inverted_conductivity': [],
        'residuals': [],
        'iteration_count': 0
    }
    
    # Inversion loop - first stage
    for k in range(max_iters):
        print(f'Iteration {k + 1}')
        
        # Update sigma for all soundings in parallel
        for j in range(num_soundings):
            b_delta = apparent_conductivity[j, :].reshape(-1, 1)
            
            # Compute the Jacobian
            J = compute_jacobian(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            
            # Perform SVD for regularization
            U, S, Vt = svd(J, full_matrices=False)
            V = Vt.T
            
            # Forward model
            pred = compute_apparent_conductivity(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            r = (b_delta - pred.reshape(-1, 1))
            
            # Projection step
            Q_SVD = np.zeros_like(V[:, 0])
            for i in range(min(ell, len(S))):
                if S[i] != 0:
                    projection_coeff = (U[:, i].T @ r) / S[i]
                    if np.all(~np.isnan(projection_coeff)) and np.all(~np.isinf(projection_coeff)):
                        Q_SVD += projection_coeff * V[:, i]
            
            # Optimize step size
            alpha_1 = optimize_alpha(r, Q_SVD, beta)
            
            # Update conductivity model with regularization
            sigma_k[:, j] = sigma_k[:, j] + alpha_1 * Q_SVD + q * (xi_k[:, j] - sigma_k[:, j])
            sigma_k[:, j] = np.maximum(sigma_k[:, j], 0)  # Enforce non-negativity
        
        # Update auxiliary variable
        for j in range(num_soundings):
            u_k = compute_u_k(xi_k[:, j], sigma_k[:, j], q)
            xi_k[:, j] = update_auxiliary_variable(u_k, sigma_k[:, j], beta)
        
        # Compute forward model predictions
        predicted_apparent_conductivity = np.zeros_like(apparent_conductivity)
        for j in range(num_soundings):
            pred = compute_apparent_conductivity(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            predicted_apparent_conductivity[j, :] = pred[:num_freqs]
        
        # Compute residual norm
        residual_norm = np.linalg.norm(apparent_conductivity - predicted_apparent_conductivity, 'fro')
        results['residuals'].append(residual_norm)
        print(f'Residual Norm: {residual_norm:.6f}')
        
        # Convergence check
        if residual_norm < epsilon:
            print(f'Converged after {k + 1} iterations')
            results['iteration_count'] = k + 1
            break
    
    # Store the results from first stage
    results['inverted_conductivity'] = sigma_k.copy()
    
    # Second stage with modified parameters
    max_iters_stage2 = 10
    
    for k in range(max_iters_stage2):
        print(f'Stage 2 - Iteration {k + 1}')
        
        # Update sigma for all soundings in parallel
        for j in range(num_soundings):
            b_delta = apparent_conductivity[j, :].reshape(-1, 1)
            
            # Compute the Jacobian
            J = compute_jacobian(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            
            # Perform SVD for regularization
            U, S, Vt = svd(J, full_matrices=False)
            V = Vt.T
            
            # Forward model
            pred = compute_apparent_conductivity(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            r = (b_delta - pred.reshape(-1, 1))
            
            # Projection step
            Q_SVD = np.zeros_like(V[:, 0])
            for i in range(min(ell, len(S))):
                if S[i] != 0:
                    projection_coeff = (U[:, i].T @ r) / S[i]
                    if np.all(~np.isnan(projection_coeff)) and np.all(~np.isinf(projection_coeff)):
                        Q_SVD += projection_coeff * V[:, i]
            
            # Optimize step size
            alpha_1 = optimize_alpha(r, Q_SVD, beta)
            
            # Update conductivity model with regularization
            sigma_k[:, j] = sigma_k[:, j] + alpha_1 * Q_SVD + q * (xi_k[:, j] - sigma_k[:, j])
            sigma_k[:, j] = np.maximum(sigma_k[:, j], 0)  # Enforce non-negativity
        
        # Update auxiliary variable
        for j in range(num_soundings):
            u_k = compute_u_k(xi_k[:, j], sigma_k[:, j], q)
            xi_k[:, j] = update_auxiliary_variable(u_k, sigma_k[:, j], beta)
        
        # Compute forward model predictions
        predicted_apparent_conductivity = np.zeros_like(apparent_conductivity)
        for j in range(num_soundings):
            pred = compute_apparent_conductivity(sigma_k[:, j], frequencies, dipole_dist, sensor_ht)
            predicted_apparent_conductivity[j, :] = pred[:num_freqs]
        
        # Compute residual norm
        residual_norm = np.linalg.norm(apparent_conductivity - predicted_apparent_conductivity, 'fro')
        results['residuals'].append(residual_norm)
        print(f'Stage 2 - Residual Norm: {residual_norm:.6f}')
    
    # Store final inverted conductivity
    results['final_inverted_conductivity'] = sigma_k
    
    # Generate plots
    plot_inversion_results(results, output_dir, os.path.basename(csv_file).split('.')[0])
    
    # Save results to file
    save_results(results, output_dir, os.path.basename(csv_file).split('.')[0])
    
    return results

def plot_inversion_results(results, output_dir, base_filename):
    """
    Create plots of inversion results
    
    Parameters:
        results (dict): Dictionary containing inversion results
        output_dir (str): Directory to save plots
        base_filename (str): Base name for the output files
    """
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(results['residuals'], marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.title('Convergence of Inversion Process')
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{base_filename}_residuals.png'))
    plt.close()
    
    # Plot inverted conductivity profiles for selected soundings
    num_soundings = results['final_inverted_conductivity'].shape[1]
    plot_indices = np.linspace(0, num_soundings-1, min(5, num_soundings), dtype=int)
    
    plt.figure(figsize=(12, 8))
    depths = np.arange(results['final_inverted_conductivity'].shape[0]) * 5  # Assuming 5m layer thickness
    
    for idx in plot_indices:
        if idx < num_soundings:
            timestamp = results['timestamps'][idx]
            plt.semilogy(results['final_inverted_conductivity'][:, idx], depths, label=f'Time: {timestamp:.2f}s')
    
    plt.gca().invert_yaxis()  # Invert Y-axis for depth
    plt.xlabel('Conductivity (S/m)')
    plt.ylabel('Depth (m)')
    plt.title('Inverted Conductivity Profiles')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f'{base_filename}_conductivity_profiles.png'))
    plt.close()
    
    # Create 2D pseudosection plot
    plt.figure(figsize=(14, 8))
    
    # Create a meshgrid for plotting
    X, Y = np.meshgrid(results['timestamps'], np.arange(results['final_inverted_conductivity'].shape[0]) * 5)
    
    # Plot as a 2D colormap
    plt.pcolormesh(X, Y, results['final_inverted_conductivity'], cmap='jet', shading='auto')
    plt.gca().invert_yaxis()  # Invert Y-axis for depth
    plt.colorbar(label='Conductivity (S/m)')
    plt.xlabel('Time (s)')
    plt.ylabel('Depth (m)')
    plt.title('Conductivity Pseudosection')
    plt.savefig(os.path.join(plots_dir, f'{base_filename}_conductivity_pseudosection.png'))
    plt.close()

def save_results(results, output_dir, base_filename):
    """
    Save inversion results to files
    
    Parameters:
        results (dict): Dictionary containing inversion results
        output_dir (str): Directory to save results
        base_filename (str): Base name for the output files
    """
    # Create output filename with timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base_filename}_results_{timestamp_str}"
    
    # Save inverted conductivity as CSV
    conductivity_df = pd.DataFrame(results['final_inverted_conductivity'])
    conductivity_df.to_csv(os.path.join(output_dir, f"{output_filename}_conductivity.csv"), index=False)
    
    # Save residuals
    residuals_df = pd.DataFrame({'iteration': range(1, len(results['residuals'])+1), 
                                'residual': results['residuals']})
    residuals_df.to_csv(os.path.join(output_dir, f"{output_filename}_residuals.csv"), index=False)
    
    # Save metadata and parameters
    with open(os.path.join(output_dir, f"{output_filename}_metadata.txt"), 'w') as f:
        f.write(f"Inversion Results for {base_filename}\n")
        f.write(f"Date: {timestamp_str}\n")
        f.write(f"Number of soundings: {results['final_inverted_conductivity'].shape[1]}\n")
        f.write(f"Number of layers: {results['final_inverted_conductivity'].shape[0]}\n")
        f.write(f"Number of frequencies: {len(results['frequencies'])}\n")
        f.write(f"Frequencies: {', '.join([str(f) for f in results['frequencies']])}\n")
        f.write(f"Final residual norm: {results['residuals'][-1]}\n")

if __name__ == "__main__":
    ## add in args
    parser = argparse.ArgumentParser(description="Run geophysical inversion on CSV data")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    parser.add_argument("--output_dir", type=str, default="inversion_results", help="Directory to save results")
    args = parser.parse_args()
    # Process all CSV files in the data directory
    run_geophysical_inversion( args.csv_file, output_dir=args.output_dir)
    
    # Alternatively, process a specific file
    # run_geophysical_inversion('data/peak_magnitude_data_20250401_120000.csv')