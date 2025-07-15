import sys
import time
import numpy as np
import pyqtgraph as pg
import csv
import os
import socket
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout, QHBoxLayout, QApplication

# Import the sensor fusion service
from sensor_fusion_service import SensorFusionService, SensorFusionState

class PositionTrackingGUI:
    def __init__(self, sensor_port=5001, debug=False):
        self.sensor_port = sensor_port
        self.debug = debug
        
        # Initialize sensor fusion service
        self.sensor_service = SensorFusionService(port=sensor_port, debug=debug)
        
        # Data storage
        self.position_history = np.array([]).reshape(0, 3)  # [x, y, z] positions
        self.velocity_history = np.array([]).reshape(0, 3)  # [vx, vy, vz] velocities
        self.timestamps = np.array([])
        self.euler_history = np.array([]).reshape(0, 3)  # [pitch, roll, yaw]
        self.speed_history = np.array([])  # magnitude of velocity
        self.sample_count = 0
        
        # Setup PyQt application
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        self.setup_gui()
        self.setup_timer()
        
    def setup_gui(self):
        """Initialize the GUI components."""
        # Main window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-time Position Tracking")
        self.win.resize(1400, 1000)
        self.win.setWindowTitle('Real-time Position & Movement Tracking')
        
        # Set dark theme
        pg.setConfigOption('background', (30, 30, 30))
        pg.setConfigOption('foreground', 'w')
        
        # Row 1: XY Position Plot (main plot)
        self.xy_plot = self.win.addPlot(title="XY Position Track", colspan=2)
        self.xy_plot.setLabel('left', 'Y Position', units='m')
        self.xy_plot.setLabel('bottom', 'X Position', units='m')
        self.xy_plot.setAspectLocked(True)  # Keep aspect ratio 1:1
        self.xy_plot.showGrid(True, True, alpha=0.3)
        
        # Position trail (full history)
        self.position_trail = self.xy_plot.plot(pen=pg.mkPen(color=(100, 149, 237), width=2), 
                                               name="Position Trail")
        
        # Current position marker
        self.current_position = self.xy_plot.plot(pen=None, 
                                                 symbol='o', 
                                                 symbolBrush=(255, 0, 0), 
                                                 symbolSize=10, 
                                                 name="Current Position")
        
        # Starting position marker
        self.start_position = self.xy_plot.plot(pen=None, 
                                               symbol='s', 
                                               symbolBrush=(0, 255, 0), 
                                               symbolSize=12, 
                                               name="Start Position")
        
        self.xy_plot.addLegend()
        
        # Row 2: Time-based plots
        self.win.nextRow()
        
        # Speed over time
        self.speed_plot = self.win.addPlot(title="Speed over Time")
        self.speed_plot.setLabel('left', 'Speed', units='m/s')
        self.speed_plot.setLabel('bottom', 'Time', units='s')
        self.speed_curve = self.speed_plot.plot(pen=pg.mkPen(color=(255, 165, 0), width=2))
        self.speed_plot.showGrid(True, True, alpha=0.3)
        
        # Altitude over time
        self.altitude_plot = self.win.addPlot(title="Altitude over Time")
        self.altitude_plot.setLabel('left', 'Z Position', units='m')
        self.altitude_plot.setLabel('bottom', 'Time', units='s')
        self.altitude_curve = self.altitude_plot.plot(pen=pg.mkPen(color=(50, 205, 50), width=2))
        self.altitude_plot.showGrid(True, True, alpha=0.3)
        
        # Row 3: Orientation plots
        self.win.nextRow()
        
        # Heading (yaw) over time
        self.heading_plot = self.win.addPlot(title="Heading over Time")
        self.heading_plot.setLabel('left', 'Heading', units='degrees')
        self.heading_plot.setLabel('bottom', 'Time', units='s')
        self.heading_curve = self.heading_plot.plot(pen=pg.mkPen(color=(255, 20, 147), width=2))
        self.heading_plot.showGrid(True, True, alpha=0.3)
        
        # Velocity components
        self.velocity_plot = self.win.addPlot(title="Velocity Components")
        self.velocity_plot.setLabel('left', 'Velocity', units='m/s')
        self.velocity_plot.setLabel('bottom', 'Time', units='s')
        self.velocity_plot.addLegend()
        
        self.vx_curve = self.velocity_plot.plot(pen=pg.mkPen(color=(255, 0, 0), width=2), name="Vx")
        self.vy_curve = self.velocity_plot.plot(pen=pg.mkPen(color=(0, 255, 0), width=2), name="Vy")
        self.vz_curve = self.velocity_plot.plot(pen=pg.mkPen(color=(0, 0, 255), width=2), name="Vz")
        self.velocity_plot.showGrid(True, True, alpha=0.3)
        
        # Row 4: Control panel
        self.win.nextRow()
        self.setup_control_panel()
        
        # Status text
        self.status_text = pg.TextItem("Waiting for sensor data...", 
                                     color=(255, 255, 255), 
                                     anchor=(0, 0))
        self.xy_plot.addItem(self.status_text)
        self.status_text.setPos(0.02, 0.95)  # Position in plot coordinates
        
    def setup_control_panel(self):
        """Setup the control panel with buttons and statistics."""
        # Create styled frame
        button_frame = self.create_styled_frame()
        button_layout = QVBoxLayout(button_frame)
        
        # Header
        header = QLabel("Movement Tracking Controls")
        header.setFont(QFont("Arial", 12, QFont.Bold))
        header.setStyleSheet("color: #FFFFFF;")
        header.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(header)
        
        # Statistics display
        self.stats_label = QLabel("No data yet")
        self.stats_label.setStyleSheet("color: #CCCCCC; font-family: monospace;")
        self.stats_label.setAlignment(Qt.AlignLeft)
        button_layout.addWidget(self.stats_label)
        
        # Button container
        btn_container = QHBoxLayout()
        
        # Clear button
        clear_button = pg.QtWidgets.QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_data)
        clear_button.setMinimumHeight(30)
        clear_button.setStyleSheet(self.get_button_style("#E74C3C", "#C0392B", "#A93226"))
        btn_container.addWidget(clear_button)
        
        # Center view button
        center_button = pg.QtWidgets.QPushButton("Center View")
        center_button.clicked.connect(self.center_view)
        center_button.setMinimumHeight(30)
        center_button.setStyleSheet(self.get_button_style("#3498DB", "#2980B9", "#21618C"))
        btn_container.addWidget(center_button)
        
        # Export CSV button
        export_button = pg.QtWidgets.QPushButton("Export CSV")
        export_button.clicked.connect(self.export_data_csv)
        export_button.setMinimumHeight(30)
        export_button.setStyleSheet(self.get_button_style("#27AE60", "#229954", "#1E8449"))
        btn_container.addWidget(export_button)
        
        button_layout.addLayout(btn_container)
        
        # Add to window
        button_proxy = pg.QtWidgets.QGraphicsProxyWidget()
        button_proxy.setWidget(button_frame)
        control_layout = self.win.addLayout(row=3, col=0, colspan=2)
        control_layout.addItem(button_proxy, row=0, col=0)
        
    def create_styled_frame(self):
        """Create a styled frame for the control panel."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setFrameShadow(QFrame.Raised)
        frame.setLineWidth(2)
        
        palette = frame.palette()
        palette.setColor(QPalette.Window, QColor(50, 50, 50))
        frame.setPalette(palette)
        frame.setAutoFillBackground(True)
        
        return frame
        
    def get_button_style(self, normal_color, hover_color, pressed_color):
        """Get button style CSS."""
        return f"""
            QPushButton {{
                background-color: {normal_color};
                color: white;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """
    
    def setup_timer(self):
        """Setup timer for periodic updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(50)  # Update every 50ms (20 Hz)
        
    def get_local_ip(self):
        """Get the local IP address."""
        try:
            # Connect to a remote server to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def start(self):
        """Start the position tracking service."""
        self.sensor_service.start()
        local_ip = self.get_local_ip()
        print(f"Position tracking started on port {self.sensor_port}")
        print(f"Connect from your phone to: {local_ip}:{self.sensor_port}")
        
    def update_plots(self):
        """Update all plots with latest sensor data."""
        # Get current state from sensor fusion
        state = self.sensor_service.get_current_state()
        
        if state is None:
            return
            
        # Add new data point
        self.position_history = np.vstack([self.position_history, state.position])
        self.velocity_history = np.vstack([self.velocity_history, state.velocity])
        self.euler_history = np.vstack([self.euler_history, state.euler])
        self.timestamps = np.append(self.timestamps, state.timestamp)
        
        # Calculate speed (magnitude of velocity)
        speed = np.linalg.norm(state.velocity)
        self.speed_history = np.append(self.speed_history, speed)
        
        self.sample_count += 1
        
        # Update XY position plot
        if len(self.position_history) > 0:
            x_positions = self.position_history[:, 0]
            y_positions = self.position_history[:, 1]
            
            # Update position trail
            self.position_trail.setData(x_positions, y_positions)
            
            # Update current position marker
            self.current_position.setData([x_positions[-1]], [y_positions[-1]])
            
            # Update start position marker (first point)
            if len(x_positions) > 0:
                self.start_position.setData([x_positions[0]], [y_positions[0]])
        
        # Update time-based plots
        if len(self.timestamps) > 1:
            # Convert timestamps to relative time from start
            relative_times = self.timestamps - self.timestamps[0]
            
            # Speed plot
            self.speed_curve.setData(relative_times, self.speed_history)
            
            # Altitude plot
            self.altitude_curve.setData(relative_times, self.position_history[:, 2])
            
            # Heading plot (convert radians to degrees)
            heading_degrees = np.degrees(self.euler_history[:, 2])  # yaw
            self.heading_curve.setData(relative_times, heading_degrees)
            
            # Velocity components
            self.vx_curve.setData(relative_times, self.velocity_history[:, 0])
            self.vy_curve.setData(relative_times, self.velocity_history[:, 1])
            self.vz_curve.setData(relative_times, self.velocity_history[:, 2])
        
        # Update status and statistics
        self.update_status(state)
        
    def update_status(self, state: SensorFusionState):
        """Update status text and statistics."""
        # Status text on plot
        status_lines = [
            f"Samples: {self.sample_count}",
            f"Position: ({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}) m",
            f"Speed: {np.linalg.norm(state.velocity):.2f} m/s",
            f"Heading: {np.degrees(state.euler[2]):.1f}°"
        ]
        
        if state.gps_reference:
            status_lines.append(f"GPS Ref: ({state.gps_reference[0]:.6f}, {state.gps_reference[1]:.6f})")
        
        if state.last_gps_accuracy:
            status_lines.append(f"GPS Acc: {state.last_gps_accuracy[0]:.1f}m")
            
        self.status_text.setText("\\n".join(status_lines))
        
        # Statistics panel
        if len(self.position_history) > 1:
            total_distance = self.calculate_total_distance()
            max_speed = np.max(self.speed_history) if len(self.speed_history) > 0 else 0
            avg_speed = np.mean(self.speed_history) if len(self.speed_history) > 0 else 0
            duration = self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
            
            stats_text = f"""Movement Statistics:
Duration: {duration:.1f} seconds
Total Distance: {total_distance:.2f} meters
Max Speed: {max_speed:.2f} m/s
Avg Speed: {avg_speed:.2f} m/s
Sample Rate: {state.sample_rate:.1f} Hz
Current Position: ({state.position[0]:.2f}, {state.position[1]:.2f}) m"""
            
            self.stats_label.setText(stats_text)
        
    def calculate_total_distance(self):
        """Calculate total distance traveled."""
        if len(self.position_history) < 2:
            return 0.0
            
        distances = np.linalg.norm(np.diff(self.position_history, axis=0), axis=1)
        return np.sum(distances)
        
    def clear_data(self):
        """Clear all stored data and reset plots."""
        self.position_history = np.array([]).reshape(0, 3)
        self.velocity_history = np.array([]).reshape(0, 3)
        self.timestamps = np.array([])
        self.euler_history = np.array([]).reshape(0, 3)
        self.speed_history = np.array([])
        self.sample_count = 0
        
        # Clear all plot data
        self.position_trail.clear()
        self.current_position.clear()
        self.start_position.clear()
        self.speed_curve.clear()
        self.altitude_curve.clear()
        self.heading_curve.clear()
        self.vx_curve.clear()
        self.vy_curve.clear()
        self.vz_curve.clear()
        
        self.status_text.setText("Data cleared. Waiting for new sensor data...")
        self.stats_label.setText("No data yet")
        
        print("Position tracking history cleared.")
        
    def center_view(self):
        """Center the XY plot view on the current data."""
        if len(self.position_history) > 0:
            x_positions = self.position_history[:, 0]
            y_positions = self.position_history[:, 1]
            
            # Add some padding around the data
            x_margin = (np.max(x_positions) - np.min(x_positions)) * 0.1 + 1
            y_margin = (np.max(y_positions) - np.min(y_positions)) * 0.1 + 1
            
            self.xy_plot.setXRange(np.min(x_positions) - x_margin, 
                                  np.max(x_positions) + x_margin)
            self.xy_plot.setYRange(np.min(y_positions) - y_margin, 
                                  np.max(y_positions) + y_margin)
            
    def export_data_csv(self):
        """Export position tracking data to CSV."""
        try:
            os.makedirs("data", exist_ok=True)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/position_tracking_{timestamp_str}.csv"
            
            with open(filename, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                # Write header
                header = ['Timestamp', 'Relative_Time_s', 'X_m', 'Y_m', 'Z_m', 
                         'VX_ms', 'VY_ms', 'VZ_ms', 'Speed_ms',
                         'Pitch_deg', 'Roll_deg', 'Yaw_deg']
                csvwriter.writerow(header)
                
                # Write data
                if len(self.timestamps) > 0:
                    relative_times = self.timestamps - self.timestamps[0]
                    
                    for i in range(len(self.timestamps)):
                        row = [
                            self.timestamps[i],
                            relative_times[i],
                            self.position_history[i, 0],
                            self.position_history[i, 1], 
                            self.position_history[i, 2],
                            self.velocity_history[i, 0],
                            self.velocity_history[i, 1],
                            self.velocity_history[i, 2],
                            self.speed_history[i],
                            np.degrees(self.euler_history[i, 0]),  # pitch
                            np.degrees(self.euler_history[i, 1]),  # roll
                            np.degrees(self.euler_history[i, 2])   # yaw
                        ]
                        csvwriter.writerow(row)
            
            print(f"Position data exported to {filename}")
            
            # Show success message
            msg_box = pg.QtWidgets.QMessageBox()
            msg_box.setWindowTitle("Export Successful")
            msg_box.setText(f"Position data exported to:\\n{os.path.abspath(filename)}")
            msg_box.exec_()
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            error_box = pg.QtWidgets.QMessageBox()
            error_box.setWindowTitle("Export Error")
            error_box.setText(f"Error exporting data: {e}")
            error_box.exec_()
    
    def run(self):
        """Run the GUI application."""
        self.start()
        
        try:
            if hasattr(self.app, 'exec_'):
                self.app.exec_()
            else:
                self.app.exec()
        except KeyboardInterrupt:
            print("\\nShutting down...")
        finally:
            self.sensor_service.stop()

# Main execution
if __name__ == "__main__":
    # Create and run the position tracking GUI
    tracker = PositionTrackingGUI(sensor_port=5001, debug=True)
    tracker.run()