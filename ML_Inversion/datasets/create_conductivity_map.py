import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def rotate_coordinates(xyz, inclination, declination):
    """
    Rotate coordinates based on inclination and declination angles.
    
    Parameters:
    -----------
    xyz : numpy.ndarray
        Array of shape (3, n) containing the x, y, z coordinates to rotate
    inclination : float
        Inclination angle in degrees (rotation around y-axis)
    declination : float
        Declination angle in degrees (rotation around z-axis)
        
    Returns:
    --------
    rotated_xyz : numpy.ndarray
        Array of rotated coordinates
    """
    # Convert angles to radians
    incl_rad = np.radians(inclination)
    decl_rad = np.radians(declination)
    
    # Create rotation matrices
    # Rotation around y-axis (inclination)
    Ry = np.array([
        [np.cos(incl_rad), 0, np.sin(incl_rad)],
        [0, 1, 0],
        [-np.sin(incl_rad), 0, np.cos(incl_rad)]
    ])
    
    # Rotation around z-axis (declination)
    Rz = np.array([
        [np.cos(decl_rad), -np.sin(decl_rad), 0],
        [np.sin(decl_rad), np.cos(decl_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply rotations
    rotated_xyz = Rz @ Ry @ xyz
    
    return rotated_xyz

def create_cylinder_conductivity_map(
    depth,          # Depth to center of cylinder (m)
    inclination,    # Inclination angle of cylinder (degrees)
    declination,    # Declination angle of cylinder (degrees)
    radius,         # Radius of the cylinder (m)
    length,         # Length of the cylinder (m)
    conductivity,   # Conductivity value of the cylinder (S/m)
    background_conductivity=0.001,  # Background conductivity (S/m)
    x_range=(-10, 10),  # X range for the model (m)
    y_range=(-10, 10),  # Y range for the model (m)
    z_range=(0, 10),    # Z range for the model (m)
    grid_spacing=0.5,   # Grid spacing for model cells (m)
):
    """
    Create a 3D conductivity map representing a conductive cylinder underground.
    
    Parameters:
    -----------
    depth : float
        Depth to the center of the cylinder (m) from the surface
    inclination : float
        Inclination angle of the cylinder axis (degrees) from horizontal
    declination : float
        Declination angle of the cylinder axis (degrees) from north/x-axis
    radius : float
        Radius of the cylinder (m)
    length : float
        Length of the cylinder (m)
    conductivity : float
        Conductivity value of the cylinder (S/m)
    background_conductivity : float
        Conductivity value of the background medium (S/m)
    x_range : tuple
        Min and max x-coordinates (m)
    y_range : tuple
        Min and max y-coordinates (m)
    z_range : tuple
        Min and max z-coordinates (m) - z increases with depth
    grid_spacing : float
        Spacing between grid cells (m)
        
    Returns:
    --------
    conductivity_map : numpy.ndarray
        3D tensor of conductivity values
    grid_info : dict
        Dictionary containing grid information (x, y, z coordinates)
    """
    # Create the grid coordinates
    x = np.arange(x_range[0], x_range[1], grid_spacing)
    y = np.arange(y_range[0], y_range[1], grid_spacing)
    z = np.arange(z_range[0], z_range[1], grid_spacing)
    
    nx, ny, nz = len(x), len(y), len(z)
    
    # Initialize the conductivity map with background conductivity
    conductivity_map = np.ones((nx, ny, nz)) * background_conductivity
    
    # Create cylinder axis vector
    # Initial cylinder is aligned with x-axis
    axis_vector = np.array([1, 0, 0])
    
    # Rotate the axis vector according to inclination and declination
    axis_matrix = np.reshape(axis_vector, (3, 1))
    rotated_axis = rotate_coordinates(axis_matrix, inclination, declination).flatten()
    
    # Find the center point of the cylinder
    # Start with center at origin, then adjust for depth
    center = np.array([0, 0, depth])
    
    # Create vectors for all grid points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    
    # Reshape for vectorized operations
    points_reshaped = points.reshape(-1, 3)
    
    # For each point, determine if it's inside the cylinder
    # Vector from center to point
    vectors_to_points = points_reshaped - center
    
    # Component of vectors_to_points along the cylinder axis
    proj_along_axis = np.dot(vectors_to_points, rotated_axis)
    
    # Check if point is within the length of the cylinder
    half_length = length / 2
    within_length = np.abs(proj_along_axis) <= half_length
    
    # Components perpendicular to the axis
    # proj_perp = vectors_to_points - np.outer(proj_along_axis, rotated_axis)
    # dist_from_axis = np.linalg.norm(proj_perp, axis=1)
    
    # Alternate way to calculate perpendicular distance
    # For any point, the distance from the axis is:
    # d = ||(v - c) × a|| / ||a||
    # where v is the point, c is a point on the axis, and a is the axis direction
    cross_products = np.cross(vectors_to_points, rotated_axis)
    dist_from_axis = np.linalg.norm(cross_products, axis=1) / np.linalg.norm(rotated_axis)
    
    # Check if points are within radius
    within_radius = dist_from_axis <= radius
    
    # Points inside the cylinder
    inside_cylinder = within_length & within_radius
    
    # Assign conductivity value to points inside the cylinder
    conductivity_map_flat = np.ones_like(inside_cylinder, dtype=float) * background_conductivity
    conductivity_map_flat[inside_cylinder] = conductivity
    
    # Reshape back to 3D
    conductivity_map = conductivity_map_flat.reshape(nx, ny, nz)
    
    # Create grid information dictionary
    grid_info = {
        'x': x,
        'y': y,
        'z': z,
        'spacing': grid_spacing
    }
    
    return conductivity_map, grid_info

def visualize_conductivity_map(conductivity_map, grid_info, cylinder_params=None):
    """
    Visualize the 3D conductivity map.
    
    Parameters:
    -----------
    conductivity_map : numpy.ndarray
        3D tensor of conductivity values
    grid_info : dict
        Dictionary containing grid information (x, y, z coordinates)
    cylinder_params : dict, optional
        Dictionary with cylinder parameters for annotating the plot
    """
    x, y, z = grid_info['x'], grid_info['y'], grid_info['z']
    
    # Create figure for cross-sectional views
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Calculate middle indices
    mid_x = len(x) // 2
    mid_y = len(y) // 2
    mid_z = len(z) // 2
    
    # Create slices
    xy_slice = conductivity_map[:, :, mid_z]
    xz_slice = conductivity_map[:, mid_y, :]
    yz_slice = conductivity_map[mid_x, :, :]
    
    # Plot XY plane (map view)
    im0 = axes[0].imshow(
        xy_slice.T,  # Transpose for correct orientation
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        cmap='viridis',
        vmin=np.min(conductivity_map),
        vmax=np.max(conductivity_map)
    )
    axes[0].set_title(f'XY Plane (Depth = {z[mid_z]:.1f} m)')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    plt.colorbar(im0, ax=axes[0], label='Conductivity (S/m)')
    
    # Plot XZ plane (vertical section along Y)
    im1 = axes[1].imshow(
        xz_slice.T,  # Transpose for correct orientation
        extent=[x[0], x[-1], z[0], z[-1]],
        origin='lower',
        cmap='viridis',
        vmin=np.min(conductivity_map),
        vmax=np.max(conductivity_map)
    )
    axes[1].set_title(f'XZ Plane (Y = {y[mid_y]:.1f} m)')
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    plt.colorbar(im1, ax=axes[1], label='Conductivity (S/m)')
    
    # Plot YZ plane (vertical section along X)
    im2 = axes[2].imshow(
        yz_slice.T,  # Transpose for correct orientation
        extent=[y[0], y[-1], z[0], z[-1]],
        origin='lower',
        cmap='viridis',
        vmin=np.min(conductivity_map),
        vmax=np.max(conductivity_map)
    )
    axes[2].set_title(f'YZ Plane (X = {x[mid_x]:.1f} m)')
    axes[2].set_xlabel('Y (m)')
    axes[2].set_ylabel('Z (m)')
    plt.colorbar(im2, ax=axes[2], label='Conductivity (S/m)')
    
    # 3D visualization in the 4th subplot
    # Create a 3D surface plot of the isosurface
    # Downsample for clearer visualization
    stride = max(1, min(len(x), len(y), len(z)) // 20)
    
    # Add a 3D plot in the fourth panel
    ax3 = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Get coordinates of high conductivity cells (cylinder)
    high_cond_threshold = (conductivity_map.max() - conductivity_map.min()) * 0.5 + conductivity_map.min()
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Find indices where conductivity is above threshold
    high_cond_mask = conductivity_map > high_cond_threshold
    
    # Downsample for visualization
    high_cond_mask[::stride, ::stride, ::stride] &= True
    high_cond_mask[~(np.arange(high_cond_mask.shape[0]) % stride == 0), :, :] = False
    high_cond_mask[:, ~(np.arange(high_cond_mask.shape[1]) % stride == 0), :] = False
    high_cond_mask[:, :, ~(np.arange(high_cond_mask.shape[2]) % stride == 0)] = False
    
    # Get coordinates and values for high conductivity points
    x_high = X[high_cond_mask]
    y_high = Y[high_cond_mask]
    z_high = Z[high_cond_mask]
    cond_high = conductivity_map[high_cond_mask]
    
    # Plot as a scatter of points
    scatter = ax3.scatter(
        x_high, y_high, z_high,
        c=cond_high,
        cmap='viridis',
        marker='.',
        alpha=0.6,
        s=50
    )
    
    # Add a colorbar
    plt.colorbar(scatter, ax=ax3, label='Conductivity (S/m)')
    
    # Set labels
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Z (m)')
    ax3.set_title('3D Conductivity Model')
    
    # Adjust view angle
    ax3.view_init(elev=20, azim=-35)
    
    # Add cylinder parameters as text annotation if provided
    if cylinder_params:
        param_text = (
            f"Cylinder Parameters:\n"
            f"Depth: {cylinder_params['depth']:.1f} m\n"
            f"Length: {cylinder_params['length']:.1f} m\n"
            f"Radius: {cylinder_params['radius']:.1f} m\n"
            f"Inclination: {cylinder_params['inclination']:.1f}°\n"
            f"Declination: {cylinder_params['declination']:.1f}°\n"
            f"Conductivity: {cylinder_params['conductivity']:.2f} S/m"
        )
        plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    return fig

# Example usage
def example_usage():
    # Define cylinder parameters
    cylinder_params = {
        'depth': 3.0,            # Depth to center of cylinder (m)
        'inclination': 0.0,     # Inclination angle (degrees)
        'declination': 0.0,     # Declination angle (degrees)
        'radius': 0.5,           # Radius of cylinder (m)
        'length': 1.0,           # Length of cylinder (m)
        'conductivity': 1e6      # Conductivity of cylinder (S/m)
    }
    
    # Define grid parameters
    grid_params = {
        'x_range': (-5, 5),    # X range (m)
        'y_range': (-5, 5),    # Y range (m)
        'z_range': (0, 5),      # Z range (m)
        'grid_spacing': 0.1      # Grid spacing (m)
    }
    
    # Generate conductivity map
    print("Generating 3D conductivity map...")
    conductivity_map, grid_info = create_cylinder_conductivity_map(
        depth=cylinder_params['depth'],
        inclination=cylinder_params['inclination'],
        declination=cylinder_params['declination'],
        radius=cylinder_params['radius'],
        length=cylinder_params['length'],
        conductivity=cylinder_params['conductivity'],
        background_conductivity=0.001,
        x_range=grid_params['x_range'],
        y_range=grid_params['y_range'],
        z_range=grid_params['z_range'],
        grid_spacing=grid_params['grid_spacing']
    )
    
    # Visualize the conductivity map
    print("Visualizing conductivity map...")
    fig = visualize_conductivity_map(conductivity_map, grid_info, cylinder_params)
    
    # Display the figure
    plt.show()
    
    # Print statistics
    print(f"Model dimensions: {conductivity_map.shape}")
    print(f"Conductivity range: {np.min(conductivity_map):.3f} to {np.max(conductivity_map):.3f} S/m")
    
    return conductivity_map, grid_info

if __name__ == "__main__":
    conductivity_map, grid_info = example_usage()