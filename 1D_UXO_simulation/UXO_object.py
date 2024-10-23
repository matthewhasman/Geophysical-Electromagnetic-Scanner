import numpy as np

class UXO_object:
    def __init__(self,
                 center: np.ndarray,
                 major_axis: float,
                 minor_axis: float,
                 strike: float,
                 dip: float):
        self.center = center
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.strike = strike
        self.dip = dip

    def _compute_rotation_matrix(self, inverse: bool = False):
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

    def define_pts(self, n_points: int = 10):
        raise NotImplementedError
    
    def get_vertical_intersects(self, x: float, y: float):
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