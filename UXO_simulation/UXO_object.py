import numpy as np

class UXO_object:
    """
    A class representing an unexploded ordnance (UXO) object, modeled as a rotated ellipsoid 
    with properties describing location, shape, orientation, and conductivity.

    Attributes:
        center (np.ndarray): 3D coordinates of the UXO object's center.
        major_axis (float): Length of the major axis (along the z-axis).
        minor_axis (float): Length of the minor axes (along the x and y axes).
        strike (float): Horizontal angle (in degrees) of the major axis relative to the north.
        dip (float): Vertical angle (in degrees) of the major axis from the horizontal.
        conductivity (float): Conductivity of the UXO object (default is 1e6).
    """
    def __init__(self,
                 center: np.ndarray,
                 major_axis: float,
                 minor_axis: float,
                 strike: float = 0,
                 dip: float = 0,
                 conductivity: float = 1e6) -> None:
        """
        Initializes the UXO object with center location, dimensions, orientation, and conductivity.

        Args:
            center (np.ndarray): 3D coordinates of the UXO object's center.
            major_axis (float): Length of the major axis (along the z-axis).
            minor_axis (float): Length of the minor axes (along the x and y axes).
            strike (float): Horizontal angle (in degrees) of the major axis relative to the north.
            dip (float): Vertical angle (in degrees) of the major axis from the horizontal.
            conductivity (float): Conductivity of the UXO object (default is 1e6).
        """
        self.center = center
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.strike = strike
        self.dip = dip
        self.conductivity = conductivity

    def _compute_rotation_matrix(self, inverse: bool = False):
        """
        Computes the rotation matrix for the UXO object based on its strike and dip.

        Args:
            inverse (bool): If True, computes the inverse rotation matrix (default is False).

        Returns:
            np.ndarray: The rotation matrix (or its inverse) representing the orientation of the UXO object.
        """
        # y-axis rotation
        theta = self.dip * np.pi / 180
        Ay = np.r_[
            np.c_[np.cos(theta), 0.0, -np.sin(theta)],
            np.c_[0.0, 1.0, 0.0],
            np.c_[np.sin(theta), 0.0, np.cos(theta)],
        ]

        # z-axis rotation
        phi = self.strike * np.pi / 180
        Az = np.r_[
            np.c_[np.cos(phi), -np.sin(phi), 0.0],
            np.c_[np.sin(phi), np.cos(phi), 0.0],
            np.c_[0.0, 0.0, 1.0],
        ]

        return np.dot(Ay, Az) if not inverse else np.dot(Az.T, Ay.T)
    
    def get_vertical_intersects(self, x: float, y: float):
        """
        Computes the vertical intersection points with the UXO object at the specified x and y coordinates.

        Args:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.

        Returns:
            tuple or None: A tuple containing the z-coordinates of the intersection points (z1, z2), 
                           or None if no intersection occurs.
        """
        rot = self._compute_rotation_matrix()
        rot_inv = self._compute_rotation_matrix(inverse=True)
        pt0 = np.array([x, y, 0]) - self.center
        pt = np.dot(pt0, rot_inv)
        vec = np.dot(np.array([0, 0, 1]), rot_inv)

        # find quadratic coefficients
        a = (vec[0]**2 + vec[1]**2)/self.minor_axis**2 + vec[2]**2/self.major_axis**2
        b = 2*((pt[0]*vec[0]+pt[1]*vec[1])/self.minor_axis**2 + pt[2]*vec[2]/self.major_axis**2)
        c = (pt[0]**2 + pt[1]**2)/self.minor_axis**2 + pt[2]**2/self.major_axis**2 - 1

        # find intersection points
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return None
        else:
            t1 = (-b + np.sqrt(discriminant))/(2*a)
            t2 = (-b - np.sqrt(discriminant))/(2*a)
            p1 = np.dot(np.array([pt + t1*vec]), rot) + self.center
            p2 = np.dot(np.array([pt + t2*vec]), rot) + self.center
            return p1[0,2], p2[0,2]
        
    def get_boundary_points(self, n_points_theta: int = 100, n_points_phi: int = 100):
        """
        Generates points along the surface of the UXO object to define its boundary.

        Args:
            n_points_theta (int): Number of points in the azimuthal direction (default is 100).
            n_points_phi (int): Number of points in the polar direction (default is 100).

        Returns:
            np.ndarray: Array of boundary points on the UXO object's surface after applying orientation.
        """
        # Generate theta and phi angles
        theta = np.linspace(0, 2 * np.pi, n_points_theta)
        phi = np.linspace(0, np.pi, n_points_phi)

        # Create a meshgrid for theta and phi
        theta, phi = np.meshgrid(theta, phi)

        # Parametrize the surface of the spheroid (ellipsoid) using spherical coordinates
        x = self.center[0] + self.minor_axis * np.sin(phi) * np.cos(theta)
        y = self.center[1] + self.minor_axis * np.sin(phi) * np.sin(theta)
        z = self.center[2] + self.major_axis * np.cos(phi)

        # Apply the forward rotation matrix to rotate the points according to the spheroid's orientation
        points = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
        rot = self._compute_rotation_matrix()  # Use the forward rotation matrix
        points_rotated = np.dot(points - self.center, rot) + self.center

        return points_rotated