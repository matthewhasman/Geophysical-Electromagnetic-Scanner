% Define constants
mu0 = 4 * pi * 1e-7; % Permeability of free space (H/m)



% Define coil parameters
radiusTx = 0.12; % Radius of each coil (meters)
turnsTx = 40; % Number of turns in each coil
coil_resolution = 50; % Resolution (points per turn)

% Define coil parameters
radiusBucking = 0.05; % Radius of each coil (meters)
turnsBucking = 5; % Number of turns in each coil


% Define current magnitudes
I_magnitude_tx = 1; % Current in RX coil (Amps)
I_magnitude_bucking = -1; % Current in bucking coil (opposite direction)

tx_center = 0.85;
bucking_center = 0.245;

% Define center positions for RX and bucking coils
center_tx = [tx_center, 0]; % Center of RX coil
center_bucking =[bucking_center, 0]; % Center of bucking coil

simBounds = 0.1;
resolution = 100;
gridSize = (radiusTx*2)/resolution;

% Define the grid for the XY plane and observation points
x_points = -simBounds:gridSize:simBounds; % Range for x-axis (meters)
y_points = -simBounds:gridSize:simBounds; % Range for y-axis (meters)
z_obs = 0.00; % Observation height for Z-component calculation (meters)
[X, Y] = meshgrid(x_points, y_points); % Create grid

% Generate current sources for each coil
current_sources_rx = generate_coil_currents(center_tx, radiusTx, turnsTx, coil_resolution, I_magnitude_tx);
current_sources_bucking = generate_coil_currents(center_bucking, radiusBucking, turnsBucking, coil_resolution, I_magnitude_bucking);

% Combine the RX and bucking coils into one array
current_sources = [current_sources_rx; current_sources_bucking];
n_currents = size(current_sources, 1); % Total number of current sources

% Calculate Z-component of magnetic field at each observation point
Bz = zeros(size(X)); % Initialize magnetic field array

for i = 1:n_currents
    % Current source position and current direction
    x_c = current_sources(i, 1);
    y_c = current_sources(i, 2);
    I_x = current_sources(i, 3);
    I_y = current_sources(i, 4);
    dL = current_sources(i, 5);
    z_c = 0; % Assuming currents lie in the XY plane

    % Position vector components from current source to observation points
    Rx = X - x_c;
    Ry = Y - y_c;
    Rz = z_obs - z_c;
    R = sqrt(Rx.^2 + Ry.^2 + Rz.^2); % Distance from source to observation point

    % Cross product contribution to Z-component of B-field: Bz = (I cross R)_z / |R|^3
    % Cross-product terms in 2D (for Z-component only): Bz = (I_x * Ry - I_y * Rx) / |R|^3
    Bz = Bz + mu0 / (4 * pi) * (I_x .* Ry - I_y .* Rx) * dL ./ (R.^3);
end

% Create the heatmap view
figure;
imagesc(X(1,:), Y(:,1), Bz);  % imagesc with actual X,Y coordinates
set(gca, 'YDir', 'normal');    % Flip Y-axis to match original orientation
colormap('jet');               % You can change the colormap as needed
colorbar;

% Labels and title
xlabel('X (m)');
ylabel('Y (m)');
title('Z-component of Magnetic Field from RX and Bucking Coils');

% Add null points
hold on;
null_threshold = 1e-8;
null_points = abs(Bz) < null_threshold;
null_x = X(null_points);
null_y = Y(null_points);
plot(null_x, null_y, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r');

% Make the plot aspect ratio equal
axis equal;


% Define parameters for the RX coil area to calculate magnetic flux
rx_area_center = [0.0, 0]; % Center of RX coil area for integration
rx_area_radius = 0.05; % Radius of RX coil area for integration (meters)

% Find grid points within the RX coil area
distances = sqrt((X - rx_area_center(1)).^2 + (Y - rx_area_center(2)).^2);
inside_rx_area = distances <= rx_area_radius;

% Calculate magnetic flux by integrating Bz over the RX coil area
flux_density = Bz .* inside_rx_area; % Zero out points outside RX coil area
dx = abs(x_points(2) - x_points(1));
dy = abs(y_points(2) - y_points(1));
magnetic_flux = sum(flux_density(:)) * dx * dy;

fprintf('Magnetic flux through RX coil area: %.6e Wb\n', magnetic_flux);

% Mark RX coil area on the plot
hold on;
theta = linspace(0, 2*pi, 100);
rx_x = rx_area_center(1) + rx_area_radius * cos(theta);
rx_y = rx_area_center(2) + rx_area_radius * sin(theta);
plot3(rx_x, rx_y, ones(size(rx_x)) * max(Bz(:)), 'k--', 'LineWidth', 2); % Outline for RX area

theory_b = mu0 * turnsTx * I_magnitude_tx * pi * radiusTx^2 / (4 * pi * tx_center^3);
fprintf('Bucking Ratio: %.6f  \n', magnetic_flux/(-theory_b * pi * radiusRx^2));

legend( 'Null Points', 'RX Coil');

% Function to generate coil currents
