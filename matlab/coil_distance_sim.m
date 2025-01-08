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

% Define sweep parameters
bucking_positions = 0.2:0.005:0.3; % Array of positions to test
rx_radii = [0.05, 0.12, 0.15]; % Different RX coil radii to test
bucking_ratios = zeros(length(rx_radii), length(bucking_positions)); % Array to store results

tx_center = 0.85;

% Setup for field calculations
simBounds = max(rx_radii);
resolution = 100;
gridSize = (radiusTx*2)/resolution;

% Define the grid for the XY plane and observation points
x_points = -simBounds:gridSize:simBounds;
y_points = -simBounds:gridSize:simBounds;
z_obs = 0.00;
[X, Y] = meshgrid(x_points, y_points);

% Calculate theoretical B-field for normalization
theory_b = mu0 * turnsTx * I_magnitude_tx * pi * radiusTx^2 / (4 * pi * tx_center^3);

% Create a figure with distinct colors
figure;
colors = {'b', 'r', 'g', 'm', 'c', 'k'}; % Add more colors if needed
optimal_positions = zeros(length(rx_radii), 1);
optimal_ratios = zeros(length(rx_radii), 1);

% Loop through different RX radii
for r = 1:length(rx_radii)
    rx_area_radius = rx_radii(r);
    distances = sqrt((X - 0).^2 + (Y - 0).^2);
    inside_rx_area = distances <= rx_area_radius;
    
    % Loop through different bucking coil positions
    for i = 1:length(bucking_positions)
        bucking_center = bucking_positions(i);
        
        % Define center positions for TX and bucking coils
        center_tx = [tx_center, 0];
        center_bucking = [bucking_center, 0];
        
        % Generate current sources for each coil
        current_sources_rx = generate_coil_currents(center_tx, radiusTx, turnsTx, coil_resolution, I_magnitude_tx);
        current_sources_bucking = generate_coil_currents(center_bucking, radiusBucking, turnsBucking, coil_resolution, I_magnitude_bucking);
        
        % Combine the coils
        current_sources = [current_sources_rx; current_sources_bucking];
        n_currents = size(current_sources, 1);
        
        % Calculate Z-component of magnetic field
        Bz = zeros(size(X));
        
        for j = 1:n_currents
            x_c = current_sources(j, 1);
            y_c = current_sources(j, 2);
            I_x = current_sources(j, 3);
            I_y = current_sources(j, 4);
            dL = current_sources(j, 5);
            z_c = 0;
            
            Rx = X - x_c;
            Ry = Y - y_c;
            Rz = z_obs - z_c;
            R = sqrt(Rx.^2 + Ry.^2 + Rz.^2);
            
            Bz = Bz + mu0 / (4 * pi) * (I_x .* Ry - I_y .* Rx) * dL ./ (R.^3);
        end
        
        % Calculate magnetic flux
        flux_density = Bz .* inside_rx_area;
        dx = abs(x_points(2) - x_points(1));
        dy = abs(y_points(2) - y_points(1));
        magnetic_flux = sum(flux_density(:)) * dx * dy;
        
        % Store bucking ratio
        bucking_ratios(r,i) = magnetic_flux/(-theory_b * pi * rx_area_radius^2);
        
        % Display progress
        fprintf('Progress: RX radius %.3f m - %.1f%% (Position: %.3f m)\n', ...
            rx_area_radius, 100*i/length(bucking_positions), bucking_center);
    end
    
    % Plot results for this radius
    plot(bucking_positions, bucking_ratios(r,:), [colors{mod(r-1,length(colors))+1}, '-'], ...
        'LineWidth', 2, 'DisplayName', sprintf('RX radius = %.3f m', rx_area_radius));
    hold on;
    
    % Find and mark optimal position for this radius
    [min_ratio, min_idx] = min(abs(bucking_ratios(r,:)));
    optimal_positions(r) = bucking_positions(min_idx);
    optimal_ratios(r) = bucking_ratios(r,min_idx);
    
    % Plot optimal point
    plot(optimal_positions(r), optimal_ratios(r), [colors{mod(r-1,length(colors))+1}, 'o'], ...
        'MarkerSize', 10, 'MarkerFaceColor', colors{mod(r-1,length(colors))+1}, ...
        'DisplayName', sprintf('Optimal (r=%.3f m)', rx_area_radius));
end

% Formatting the plot
grid on;
xlabel('Bucking Coil Position (m)');
ylabel('Ratio of Bucked signal to Primary Signal');
title('Bucking Ratio vs. Bucking Coil Position for Different RX Radii');
legend('show', 'Location', 'best');

% Print optimal positions
fprintf('\nOptimal positions for each RX radius:\n');
for r = 1:length(rx_radii)
    fprintf('RX radius %.3f m: optimal position = %.3f m (Ratio: %.6f)\n', ...
        rx_radii(r), optimal_positions(r), optimal_ratios(r));
end