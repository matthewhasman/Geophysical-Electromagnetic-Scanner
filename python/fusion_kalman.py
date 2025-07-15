import numpy as np
from filterpy.kalman import KalmanFilter

class PositionKalmanFilter:
    def __init__(self, dt=0.1, velocity_decay=0.99):
        self.dt = dt
        self.velocity_decay = velocity_decay  # Decay factor to prevent persistent bias
        
        # State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
        # Using quaternions for orientation (more stable than Euler angles)
        self.kf = KalmanFilter(dim_x=10, dim_z=4)  # 4 measurements: GPS x,y,z + heading
        
        # State transition matrix F
        self.kf.F = np.eye(10)
        for i in range(3):
            self.kf.F[i, i+3] = dt  # pos += vel*dt
        # Quaternion components don't have simple linear dynamics in F matrix
        # They'll be updated in the predict step
        
        # Measurement function H (GPS measures position + heading)
        self.kf.H = np.zeros((4, 10))
        self.kf.H[0,0] = 1  # GPS x
        self.kf.H[1,1] = 1  # GPS y
        self.kf.H[2,2] = 1  # GPS z
        # Heading measurement will be handled separately due to quaternion->euler conversion
        
        # Initial state covariance
        self.kf.P *= 10.
        
        # Process noise covariance Q
        q_pos = 0.01
        q_vel = 0.1
        q_quat = 0.001  # Quaternion process noise (small)
        self.kf.Q = np.diag([
            q_pos, q_pos, q_pos,      # position
            q_vel, q_vel, q_vel,      # velocity
            q_quat, q_quat, q_quat, q_quat  # quaternion
        ])
        
        # Measurement noise covariance R
        r_pos = 5.0        # GPS position noise (meters)
        r_heading = 0.1    # Heading noise (radians)
        self.kf.R = np.diag([r_pos, r_pos, r_pos, r_heading])
        
        # Initial state - identity quaternion for orientation
        self.kf.x = np.zeros((10,1))
        self.kf.x[6,0] = 1.0  # qw = 1 (identity quaternion)
        
        # Store reference lat/lon for local coordinates
        self.lat0 = None
        self.lon0 = None
        
        # GPS change detection
        self.last_gps_position = None
        self.min_gps_change_threshold = 1.0  # meters - minimum change to trigger update
        self.gps_update_count = 0
        self.gps_duplicate_count = 0
        
        # Zero velocity detection for bias mitigation
        self.accel_history = []
        self.gyro_history = []
        self.velocity_bias = np.zeros(3)
        self.zupt_threshold = 0.2  # m/s^2 - threshold for detecting stationary
        self.zupt_window_size = 10  # number of samples to check
        self.zupt_count = 0

    def set_reference(self, lat0, lon0):
        self.lat0 = lat0
        self.lon0 = lon0

    def latlon_to_meters(self, lat, lon):
        """Approximate conversion from lat/lon to local meters."""
        R_earth = 6371000
        dlat = np.radians(lat - self.lat0)
        dlon = np.radians(lon - self.lon0)
        x = dlon * R_earth * np.cos(np.radians(self.lat0))
        y = dlat * R_earth
        return x, y

    def normalize_quaternion(self, q):
        """Normalize quaternion to unit length."""
        if q is None:
            return np.array([1, 0, 0, 0])  # Identity quaternion
        
        q = np.array(q)  # Ensure it's a numpy array
        norm = np.linalg.norm(q)
        if norm == 0 or np.isnan(norm):
            return np.array([1, 0, 0, 0])  # Identity quaternion
        return q / norm

    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix."""
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
            [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
            [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)]
        ])

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (pitch, roll, yaw)."""
        qw, qx, qy, qz = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return pitch, roll, yaw

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions."""
        if q1 is None or q2 is None:
            return np.array([1, 0, 0, 0])
        
        q1 = np.array(q1)
        q2 = np.array(q2)
        
        if len(q1) != 4 or len(q2) != 4:
            return np.array([1, 0, 0, 0])
        
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi] range."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def detect_zero_velocity(self, acc_body, gyro_body):
        """Detect if device is stationary using acceleration and gyro variance."""
        # Add to history
        self.accel_history.append(np.linalg.norm(acc_body))
        self.gyro_history.append(np.linalg.norm(gyro_body))
        
        # Keep only recent samples
        if len(self.accel_history) > self.zupt_window_size:
            self.accel_history.pop(0)
        if len(self.gyro_history) > self.zupt_window_size:
            self.gyro_history.pop(0)
        
        # Need enough samples
        if len(self.accel_history) < self.zupt_window_size:
            return False
        
        # Check variance of acceleration and gyro
        accel_var = np.var(self.accel_history)
        gyro_var = np.var(self.gyro_history)
        
        # Device is stationary if both acceleration and gyro have low variance
        is_stationary = (accel_var < self.zupt_threshold and gyro_var < 0.01)
        
        return is_stationary

    def apply_zero_velocity_update(self):
        """Apply zero velocity constraint when device is stationary."""
        # Set velocity to zero
        self.kf.x[3:6] = 0.0
        
        # Reduce velocity uncertainty
        self.kf.P[3:6, 3:6] *= 0.1
        
        self.zupt_count += 1
        if self.zupt_count % 20 == 0:  # Print every 20 ZUPTs
            print(f"[ZUPT] Applied zero velocity update #{self.zupt_count}")

    def estimate_velocity_bias(self):
        """Simple velocity bias estimation during stationary periods."""
        if len(self.accel_history) >= self.zupt_window_size:
            # If we've been stationary, any remaining velocity is likely bias
            current_velocity = self.kf.x[3:6].flatten()
            velocity_magnitude = np.linalg.norm(current_velocity)
            
            if velocity_magnitude > 0.1:  # Only update if significant velocity exists
                # Apply exponential smoothing to bias estimate
                alpha = 0.1
                self.velocity_bias = alpha * current_velocity + (1 - alpha) * self.velocity_bias

    def has_gps_moved(self, lat, lon, alt):
        """Check if GPS position has changed significantly."""
        if self.last_gps_position is None:
            return True  # First GPS reading
        
        last_lat, last_lon, last_alt = self.last_gps_position
        
        # Calculate distance using simple approximation
        R_earth = 6371000
        dlat = np.radians(lat - last_lat)
        dlon = np.radians(lon - last_lon)
        
        # Haversine distance for horizontal movement
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(last_lat)) * np.cos(np.radians(lat)) * np.sin(dlon/2)**2)
        horizontal_distance = 2 * R_earth * np.arcsin(np.sqrt(a))
        
        # Vertical distance
        vertical_distance = abs(alt - last_alt)
        
        # Total 3D distance
        total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
        
        return total_distance >= self.min_gps_change_threshold

    def predict(self, sensor_data):
        """Predict next state using iPhone sensor data.
        
        sensor_data: dict containing iOS sensor readings
        """
        dt = self.dt
        x = self.kf.x.flatten()
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = x
        
        # Extract user acceleration (already gravity-compensated)
        acc_body = np.array([
            float(sensor_data['motionUserAccelerationX']),
            float(sensor_data['motionUserAccelerationY']),
            float(sensor_data['motionUserAccelerationZ'])
        ])
        
        # Extract rotation rate
        gyro_body = np.array([
            float(sensor_data['motionRotationRateX']),
            float(sensor_data['motionRotationRateY']),
            float(sensor_data['motionRotationRateZ'])
        ])
        
        # Zero velocity detection and correction
        is_stationary = self.detect_zero_velocity(acc_body, gyro_body)
        if is_stationary:
            self.apply_zero_velocity_update()
            self.estimate_velocity_bias()
        
        # Apply velocity bias correction
        self.kf.x[3:6] -= self.velocity_bias.reshape((-1, 1))
        
        # Current quaternion
        q_current = np.array([qw, qx, qy, qz])
        
        # Transform acceleration from body frame to world frame
        R = self.quaternion_to_rotation_matrix(q_current)
        acc_world = R @ acc_body
        
        # Update velocity with world frame acceleration
        vx += acc_world[0] * dt
        vy += acc_world[1] * dt
        vz += acc_world[2] * dt
        
        # Apply velocity decay to prevent persistent bias from small acceleration errors
        vx *= self.velocity_decay
        vy *= self.velocity_decay
        vz *= self.velocity_decay
        
        # Update position with velocity
        px += vx * dt
        py += vy * dt
        pz += vz * dt
        
        # Update quaternion using angular velocity
        # q_dot = 0.5 * q * [0, wx, wy, wz]
        omega = np.array([0, gyro_body[0], gyro_body[1], gyro_body[2]])
        
        q_dot = 0.5 * self.quaternion_multiply(q_current, omega)
        
        # Integrate quaternion
        q_new = q_current + q_dot * dt
        q_new = self.normalize_quaternion(q_new)
        
        # Update state
        self.kf.x = np.array([[px], [py], [pz], [vx], [vy], [vz], 
                              [q_new[0]], [q_new[1]], [q_new[2]], [q_new[3]]])
        
        # Standard Kalman predict step
        self.kf.predict()

    def update_with_sensor_data(self, sensor_data):
        """Update filter with available sensor measurements."""
        measurements = []
        H_rows = []
        R_values = []
        
        # GPS position update - only if position has changed significantly
        gps_updated = False
        if 'locationLatitude' in sensor_data and float(sensor_data['locationLatitude']) != 0:
            lat = float(sensor_data['locationLatitude'])
            lon = float(sensor_data['locationLongitude'])
            alt = float(sensor_data['locationAltitude'])
            
            if self.lat0 is None or self.lon0 is None:
                # Initialize reference point with first GPS reading
                self.lat0 = lat
                self.lon0 = lon
                self.last_gps_position = (lat, lon, alt)
                return False  # Skip first measurement, just set reference
            
            # Check if GPS has moved significantly
            if self.has_gps_moved(lat, lon, alt):
                x, y = self.latlon_to_meters(lat, lon)
                
                measurements.extend([x, y, alt])
                
                # GPS measurement rows in H matrix
                H_gps = np.zeros((3, 10))
                H_gps[0,0] = 1  # x
                H_gps[1,1] = 1  # y
                H_gps[2,2] = 1  # z
                H_rows.append(H_gps)
                
                # GPS measurement noise
                h_acc = float(sensor_data.get('locationHorizontalAccuracy', 5.0))
                v_acc = float(sensor_data.get('locationVerticalAccuracy', 10.0))
                R_values.extend([h_acc**2, h_acc**2, v_acc**2])
                
                # Update last known GPS position
                self.last_gps_position = (lat, lon, alt)
                self.gps_update_count += 1
                gps_updated = True
                
                # Debug output
                current_pos = self.kf.x[0:3].flatten()
                distance_moved = np.sqrt((x - current_pos[0])**2 + (y - current_pos[1])**2 + (alt - current_pos[2])**2)
                print(f"[GPS UPDATE] Moved {distance_moved:.2f}m, Total updates: {self.gps_update_count}")
            else:
                self.gps_duplicate_count += 1
                if self.gps_duplicate_count % 100 == 0:  # Print every 100 duplicates
                    print(f"[GPS] Ignored {self.gps_duplicate_count} duplicate readings (< {self.min_gps_change_threshold}m change)")
        
        # Heading update (prefer true heading over magnetic heading)
        heading_available = False
        if 'locationTrueHeading' in sensor_data and float(sensor_data['locationTrueHeading']) >= 0:
            heading = np.radians(float(sensor_data['locationTrueHeading']))
            heading_available = True
        elif 'locationMagneticHeading' in sensor_data and float(sensor_data['locationMagneticHeading']) >= 0:
            heading = np.radians(float(sensor_data['locationMagneticHeading']))
            heading_available = True
        
        if heading_available:
            # Convert current quaternion to yaw angle
            q_current = self.kf.x[6:10].flatten()
            _, _, current_yaw = self.quaternion_to_euler(q_current)
            
            # Compute heading residual with angle wrapping
            heading_residual = self.normalize_angle(heading - current_yaw)
            
            # Direct heading update (simplified)
            measurements.append(heading_residual)
            
            # Heading measurement row (this is approximate - true implementation would 
            # require Jacobian of quaternion->euler conversion)
            H_heading = np.zeros((1, 10))
            H_heading[0,9] = 1  # Approximate: treat as direct yaw measurement
            H_rows.append(H_heading)
            
            # Heading measurement noise
            heading_acc = float(sensor_data.get('locationHeadingAccuracy', 5.0))
            R_values.append(np.radians(heading_acc)**2)
        
        # Perform update if we have measurements
        if measurements:
            z = np.array(measurements).reshape((-1, 1))
            H = np.vstack(H_rows)
            R = np.diag(R_values)
            
            # Manual Kalman update
            y_residual = z - H @ self.kf.x
            S = H @ self.kf.P @ H.T + R
            K = self.kf.P @ H.T @ np.linalg.inv(S)
            
            self.kf.x = self.kf.x + K @ y_residual
            self.kf.P = (np.eye(10) - K @ H) @ self.kf.P
            
            # Re-normalize quaternion after update
            q_current = self.kf.x[6:10].flatten()
            q_norm = self.normalize_quaternion(q_current)
            self.kf.x[6:10] = q_norm.reshape((-1, 1))
            
        return gps_updated  # Return whether GPS was actually used

    def update_with_direct_attitude(self, sensor_data):
        """Alternative update using direct attitude measurements from iOS."""
        if 'motionQuaternionW' in sensor_data:
            # Use iOS-provided quaternion directly
            q_measured = np.array([
                float(sensor_data['motionQuaternionW']),
                float(sensor_data['motionQuaternionX']),
                float(sensor_data['motionQuaternionY']),
                float(sensor_data['motionQuaternionZ'])
            ])
            
            # Update quaternion part of state directly
            self.kf.x[6:10] = q_measured.reshape((-1, 1))
            
            # Reduce uncertainty in orientation
            self.kf.P[6:10, 6:10] *= 0.1

    def get_state(self):
        """Return current estimated state as dict."""
        x = self.kf.x.flatten()
        q = x[6:10]
        pitch, roll, yaw = self.quaternion_to_euler(q)
        
        return {
            "pos": x[0:3],
            "vel": x[3:6],
            "quaternion": q,
            "euler": np.array([pitch, roll, yaw])
        }

    def get_rotation_matrix(self):
        """Return current rotation matrix (body to world frame)."""
        q = self.kf.x[6:10].flatten()
        return self.quaternion_to_rotation_matrix(q)

# Example usage with iOS sensor data:
if __name__ == "__main__":
    # Initialize filter
    kf = PositionKalmanFilter(dt=0.1)
    
    # Example sensor data (your provided format)
    sensor_data = {
        'accelerometerAccelerationX': '-0.010971',
        'accelerometerAccelerationY': '-0.058624', 
        'accelerometerAccelerationZ': '-0.994736',
        'motionUserAccelerationX': '-0.000488',
        'motionUserAccelerationY': '-0.001152',
        'motionUserAccelerationZ': '0.000652',
        'motionRotationRateX': '-0.000672',
        'motionRotationRateY': '0.001057',
        'motionRotationRateZ': '-0.000556',
        'motionQuaternionW': '0.840540',
        'motionQuaternionX': '0.021282',
        'motionQuaternionY': '-0.019516',
        'motionQuaternionZ': '-0.540980',
        'locationLatitude': '49.265025',
        'locationLongitude': '-123.235719',
        'locationAltitude': '34.472878',
        'locationTrueHeading': '215.596664',
        'locationHorizontalAccuracy': '3.799025',
        'locationVerticalAccuracy': '30.000000'
    }
    
    # Process sensor data
    kf.predict(sensor_data)
    kf.update_with_sensor_data(sensor_data)
    
    # Alternative: use direct attitude measurements
    # kf.update_with_direct_attitude(sensor_data)
    
    # Get current state
    state = kf.get_state()
    print(f"Position: {state['pos']}")
    print(f"Velocity: {state['vel']}")
    print(f"Euler angles: {np.degrees(state['euler'])}")