import threading
import socket
import json
import time
import queue
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from fusion_kalman import PositionKalmanFilter
import numpy as np

@dataclass
class SensorFusionState:
    """Container for the current sensor fusion state."""
    position: np.ndarray          # [x, y, z] in meters from reference point
    velocity: np.ndarray          # [vx, vy, vz] in m/s
    quaternion: np.ndarray        # [qw, qx, qy, qz] orientation quaternion
    euler: np.ndarray            # [pitch, roll, yaw] in radians
    timestamp: float             # Unix timestamp of this state
    sample_count: int            # Number of processed samples
    sample_rate: float           # Current sample rate in Hz
    gps_reference: Optional[Tuple[float, float]]  # (lat, lon) reference point
    last_gps_accuracy: Optional[Tuple[float, float]]  # (horizontal, vertical) accuracy in meters

class SensorFusionService:
    """
    Background service for iOS sensor fusion using Kalman filtering.
    
    Usage:
        service = SensorFusionService(port=5001)
        service.start()
        
        # In your main application loop:
        state = service.get_current_state()
        if state:
            print(f"Position: {state.position}")
            print(f"Velocity: {state.velocity}")
    """
    
    def __init__(self, port: int = 5001, dt: float = 0.1, debug: bool = False):
        """
        Initialize the sensor fusion service.
        
        Args:
            port: TCP port to listen for sensor data
            dt: Default time step for Kalman filter
            debug: Enable debug logging
        """
        self.port = port
        self.debug = debug
        self.data_queue = queue.Queue()
        self.kf = PositionKalmanFilter(dt=dt)
        
        # Service state
        self._running = False
        self._tcp_thread = None
        self._processing_thread = None
        self._lock = threading.Lock()
        
        # Current state
        self._current_state: Optional[SensorFusionState] = None
        self._reference_set = False
        
        # Statistics
        self._sample_count = 0
        self._total_dt = 0
        self._min_dt = float('inf')
        self._max_dt = 0
        self._last_time = None
        self._last_debug_print = time.time()
    
    def start(self) -> None:
        """Start the sensor fusion service."""
        if self._running:
            return
        
        self._running = True
        self._tcp_thread = threading.Thread(target=self._tcp_listener, daemon=True)
        self._processing_thread = threading.Thread(target=self._process_sensor_data, daemon=True)
        
        self._tcp_thread.start()
        self._processing_thread.start()
        
        if self.debug:
            print(f"[SensorFusion] Service started on port {self.port}")
    
    def stop(self) -> None:
        """Stop the sensor fusion service."""
        self._running = False
        if self.debug:
            print("[SensorFusion] Service stopped")
    
    def get_current_state(self) -> Optional[SensorFusionState]:
        """
        Get the most recent sensor fusion state.
        
        Returns:
            SensorFusionState object with current position, velocity, orientation, etc.
            None if no data has been processed yet.
        """
        with self._lock:
            return self._current_state
    
    def is_ready(self) -> bool:
        """Check if the service has processed any sensor data."""
        return self._current_state is not None
    
    def get_reference_point(self) -> Optional[Tuple[float, float]]:
        """Get the GPS reference point (lat, lon) if set."""
        if self._current_state:
            return self._current_state.gps_reference
        return None
    
    def _extract_json_objects(self, buffer: str) -> list:
        """Extract complete JSON objects from TCP buffer."""
        objs = []
        depth = 0
        start_idx = None
        for i, ch in enumerate(buffer):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    objs.append(buffer[start_idx:i+1])
                    start_idx = None
        return objs
    
    def _tcp_listener(self) -> None:
        """TCP listener thread for incoming sensor data."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            sock.bind(('', self.port))
            sock.listen(1)
            if self.debug:
                print(f"[TCP] Listening for JSON data on port {self.port}...")
            
            while self._running:
                try:
                    sock.settimeout(1.0)  # Allow periodic checks of _running
                    conn, addr = sock.accept()
                    if self.debug:
                        print(f"[TCP] Connected by {addr}")
                    
                    buffer = ""
                    while self._running:
                        try:
                            data = conn.recv(4096).decode('utf-8')
                            if not data:
                                break
                            buffer += data

                            # Extract complete JSON objects
                            json_objs = self._extract_json_objects(buffer)
                            for obj_str in json_objs:
                                try:
                                    obj = json.loads(obj_str)
                                    self.data_queue.put(obj)
                                    # Remove parsed JSON from buffer
                                    buffer = buffer.replace(obj_str, '', 1)
                                except Exception as e:
                                    if self.debug:
                                        print(f"[JSON] Parse error: {e}")
                        except socket.timeout:
                            continue
                        except Exception as e:
                            if self.debug:
                                print(f"[TCP] Connection error: {e}")
                            break
                    
                    conn.close()
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.debug:
                        print(f"[TCP] Listen error: {e}")
                    time.sleep(1)
        
        finally:
            sock.close()
    
    def _process_sensor_data(self) -> None:
        """Main processing thread for sensor data."""
        while self._running:
            try:
                # Get latest data from queue (non-blocking)
                latest_data = None
                while not self.data_queue.empty():
                    try:
                        latest_data = self.data_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if latest_data:
                    self._process_single_sample(latest_data)
                else:
                    time.sleep(0.01)  # Brief sleep if no data
                    
            except Exception as e:
                if self.debug:
                    print(f"[Processing] Error: {e}")
                time.sleep(0.1)
    
    def _process_single_sample(self, data: Dict) -> None:
        """Process a single sensor data sample."""
        self._sample_count += 1
        
        # Parse timestamp
        current_time = time.time()
        if 'motionTimestamp_sinceReboot' in data:
            try:
                # Use device timestamp for more accurate timing
                motion_time = float(data['motionTimestamp_sinceReboot'])
                if hasattr(self, '_last_motion_time'):
                    dt_motion = motion_time - self._last_motion_time
                    if 0 < dt_motion < 1.0:  # Reasonable time difference
                        current_time = self._last_system_time + dt_motion
                self._last_motion_time = motion_time
                self._last_system_time = current_time
            except:
                pass
        
        # Compute dt
        if self._last_time is None:
            dt = 0.1  # Default for first sample
        else:
            dt = current_time - self._last_time
            if dt <= 0 or dt > 1.0:  # Sanity check
                dt = 0.1
        
        # Update statistics
        if dt > 0:
            self._total_dt += dt
            self._min_dt = min(self._min_dt, dt)
            self._max_dt = max(self._max_dt, dt)
        
        self._last_time = current_time
        
        # Update Kalman filter dt
        self.kf.dt = dt
        for i in range(3):
            self.kf.kf.F[i, i+3] = dt
        
        # Check for valid GPS
        has_valid_gps = False
        gps_accuracy = None
        if 'locationLatitude' in data and 'locationLongitude' in data:
            lat = float(data['locationLatitude'])
            lon = float(data['locationLongitude'])
            if lat != 0 and lon != 0:
                has_valid_gps = True
                gps_accuracy = (
                    float(data.get('locationHorizontalAccuracy', 5.0)),
                    float(data.get('locationVerticalAccuracy', 10.0))
                )
                
                # Set reference point
                if not self._reference_set:
                    self.kf.set_reference(lat, lon)
                    self._reference_set = True
                    if self.debug:
                        print(f"[GPS] Reference set to: {lat:.6f}, {lon:.6f}")
        
        # Process with Kalman filter
        try:
            self.kf.predict(data)
            
            # Always try to update with sensor data (GPS filtering happens inside)
            gps_updated = self.kf.update_with_sensor_data(data)
            
            if 'motionQuaternionW' in data:
                self.kf.update_with_direct_attitude(data)
                
            # Optional debug output for GPS updates
            if self.debug and gps_updated:
                print(f"[GPS] Position updated at sample {self._sample_count}")
            
            # Update current state
            kf_state = self.kf.get_state()
            avg_dt = self._total_dt / self._sample_count if self._sample_count > 0 else 0
            sample_rate = 1 / avg_dt if avg_dt > 0 else 0
            
            gps_ref = None
            if self._reference_set and has_valid_gps:
                gps_ref = (lat, lon)
            
            new_state = SensorFusionState(
                position=kf_state['pos'].copy(),
                velocity=kf_state['vel'].copy(),
                quaternion=kf_state['quaternion'].copy(),
                euler=kf_state['euler'].copy(),
                timestamp=current_time,
                sample_count=self._sample_count,
                sample_rate=sample_rate,
                gps_reference=gps_ref,
                last_gps_accuracy=gps_accuracy
            )
            
            with self._lock:
                self._current_state = new_state
            
            # Debug output
            if self.debug and time.time() - self._last_debug_print > 2.0:
                self._print_debug_info(new_state, dt)
                self._last_debug_print = time.time()
                
        except Exception as e:
            if self.debug:
                print(f"[Kalman] Error processing sample {self._sample_count}: {e}")
    
    def _print_debug_info(self, state: SensorFusionState, dt: float) -> None:
        """Print debug information."""
        euler_deg = np.degrees(state.euler)
        print(f"\n[SensorFusion Debug]")
        print(f"Sample: {state.sample_count}, Rate: {state.sample_rate:.1f} Hz, dt: {dt:.3f}s")
        print(f"Position: [{state.position[0]:7.2f}, {state.position[1]:7.2f}, {state.position[2]:7.2f}] m")
        print(f"Velocity: [{state.velocity[0]:7.2f}, {state.velocity[1]:7.2f}, {state.velocity[2]:7.2f}] m/s")
        print(f"Euler:    [{euler_deg[0]:6.1f}, {euler_deg[1]:6.1f}, {euler_deg[2]:6.1f}] deg")
        if state.last_gps_accuracy:
            print(f"GPS Acc:  H:{state.last_gps_accuracy[0]:.1f}m, V:{state.last_gps_accuracy[1]:.1f}m")
        print("-" * 60)


# Convenience functions for simple usage
_global_service: Optional[SensorFusionService] = None

def start_sensor_fusion(port: int = 5001, debug: bool = False) -> SensorFusionService:
    """
    Start the global sensor fusion service.
    
    Args:
        port: TCP port for sensor data
        debug: Enable debug output
        
    Returns:
        SensorFusionService instance
    """
    global _global_service
    if _global_service is None:
        _global_service = SensorFusionService(port=port, debug=debug)
        _global_service.start()
    return _global_service

def get_position() -> Optional[np.ndarray]:
    """Get current position [x, y, z] in meters from reference point."""
    if _global_service:
        state = _global_service.get_current_state()
        return state.position if state else None
    return None

def get_velocity() -> Optional[np.ndarray]:
    """Get current velocity [vx, vy, vz] in m/s."""
    if _global_service:
        state = _global_service.get_current_state()
        return state.velocity if state else None
    return None

def get_orientation() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Get current orientation as (quaternion, euler_angles)."""
    if _global_service:
        state = _global_service.get_current_state()
        return (state.quaternion, state.euler) if state else None
    return None

def get_full_state() -> Optional[SensorFusionState]:
    """Get complete sensor fusion state."""
    if _global_service:
        return _global_service.get_current_state()
    return None

def is_fusion_ready() -> bool:
    """Check if sensor fusion has valid data."""
    if _global_service:
        return _global_service.is_ready()
    return False

def stop_sensor_fusion() -> None:
    """Stop the global sensor fusion service."""
    global _global_service
    if _global_service:
        _global_service.stop()
        _global_service = None


# Example usage
if __name__ == "__main__":
    # Method 1: Using the service directly
    service = SensorFusionService(port=5001, debug=True)
    service.start()
    
    try:
        while True:
            state = service.get_current_state()
            if state:
                print(f"Position: {state.position}")
                print(f"Sample count: {state.sample_count}")
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
    
    # Method 2: Using convenience functions
    # start_sensor_fusion(debug=True)
    # try:
    #     while True:
    #         pos = get_position()
    #         vel = get_velocity()
    #         if pos is not None:
    #             print(f"Pos: {pos}, Vel: {vel}")
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     stop_sensor_fusion()