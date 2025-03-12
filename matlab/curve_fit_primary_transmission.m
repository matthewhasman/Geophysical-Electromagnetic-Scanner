[primary_magnitude, primary_phase, primary_frequency] = LoadADProMeasure("Primary_Battery.csv", 20);

% TX coil parameters (Initial Guess)
res_Tx = 4; % Ohms
L_Tx = 0.001; % Henries
N_Tx = 40; % Number of turns
r_Tx = 0.12; % Radius (m)
a_Tx = r_Tx^2 * pi;

% Electromagnet Parameters of Rx coil
r_Rx = 0.05; % Radius of coil (m)
N_Rx = 200; % Number of turns
coil_distance = 0.84; % Intercoil distance (m)
a_Rx = r_Rx^2 * pi;

% TX coil transfer function
tx_tf = tf(1, [L_Tx res_Tx]) * 1;

% Calculate total wire length of the RX coil
wire_length_Rx = 2 * pi * r_Rx * N_Rx; % meters

% RX coil parameters
L = 7e-3; % Inductance: 7 mH
R_coil = 31; % Coil resistance: 31 ohms
R_load = 1e6; % Load resistance: 1 MOhm
R = (R_coil * R_load) / (R_coil + R_load);

% Observed resonant frequency
omega_0 = 187834 * 2 * pi;
C = L/omega_0^2; % Calculate capacitance from resonant frequency

% Calculate distributed parameters for transmission line model
L_prime = L / wire_length_Rx; % H/m
R_prime = R_coil / wire_length_Rx; % Ohms/m
C_prime = C / wire_length_Rx; % F/m
G_prime = 1e-9; % S/m (starting value - will be fitted)

% Number of segments for ladder network approximation
num_segments = 10;

% Convert primary measurement to linear units
primary_linear = 10.^(primary_magnitude ./ 20);
primary_complex = primary_linear .* exp(1j * deg2rad(primary_phase));

% Set frequency cutoff
freq_cutoff = 1500; % Hz cutoff
freq_mask = primary_frequency <= freq_cutoff;
masked_frequency = primary_frequency(freq_mask);
masked_complex = primary_complex(freq_mask);

% Create frequency data object for fitting
data = idfrd(primary_complex, 2*pi*primary_frequency, 0);
masked_data = idfrd(masked_complex, 2*pi*masked_frequency, 0);

% Basic RLC model for initial comparison
Q = (1/R) * sqrt(L / C);
H_rlc = tf(omega_0^2, [1, omega_0/Q, omega_0^2]);

% Create ladder network model for the transmission line
[num_tl, den_tl] = create_ladder_network(R_prime, L_prime, G_prime, C_prime, wire_length_Rx, num_segments);
H_tl_initial = tf(num_tl, den_tl);

% ---- PARAMETER FITTING ----

% Setup optimization parameters
x0 = [R_prime, L_prime, G_prime, C_prime]; % Initial parameters
lb = [0.5*R_prime, 0.5*L_prime, 1e-12, 0.5*C_prime]; % Lower bounds
ub = [2*R_prime, 2*L_prime, 1e-6, 2*C_prime]; % Upper bounds

% Objective function for optimization
objective = @(x) transmission_line_fit_error(x, masked_frequency, masked_complex, wire_length_Rx, num_segments);

% Run optimization
options = optimoptions('fmincon', 'Display', 'iter', 'MaxIterations', 50);
x_opt = fmincon(objective, x0, [], [], [], [], lb, ub, [], options);

% Get optimized parameters
R_prime_opt = x_opt(1);
L_prime_opt = x_opt(2);
G_prime_opt = x_opt(3);
C_prime_opt = x_opt(4);

% Create optimized transmission line model
[num_tl_opt, den_tl_opt] = create_ladder_network(R_prime_opt, L_prime_opt, G_prime_opt, C_prime_opt, wire_length_Rx, num_segments);
H_tl_optimized = tf(num_tl_opt, den_tl_opt);

% Combine with coupling and TX model
rx_primary_tf_rlc = tf([-N_Tx*N_Rx*a_Tx*a_Rx, 0], coil_distance^3) * tx_tf * 1e-7 * H_rlc;
rx_primary_tf_tl = tf([-N_Tx*N_Rx*a_Tx*a_Rx, 0], coil_distance^3) * tx_tf * 1e-7 * H_tl_optimized;

% Also try a direct fitting approach for comparison
primary_estimate_tf = tfest(masked_data, 1, 1);

% Estimate a higher-order transfer function model
% This uses 4th order denominator (poles) and 2nd order numerator (zeros)
high_order_model = tfest(data, 4, 2);

% Display fitted parameters
fprintf('Optimized Transmission Line Parameters:\n');
fprintf('R_prime: %.6e Ohm/m\n', R_prime_opt);
fprintf('L_prime: %.6e H/m\n', L_prime_opt);
fprintf('G_prime: %.6e S/m\n', G_prime_opt);
fprintf('C_prime: %.6e F/m\n', C_prime_opt);
fprintf('Total C: %.6e F\n', C_prime_opt * wire_length_Rx);

% Generate frequency vector
w = 2 * pi * primary_frequency;

% Compute frequency responses
[mag_rlc, phase_rlc] = bode(rx_primary_tf_rlc, w);
[mag_tl, phase_tl] = bode(rx_primary_tf_tl, w);
[mag_est, phase_est] = bode(primary_estimate_tf * H_rlc, w);
[mag_high, phase_high] = bode(high_order_model, w);

% Squeeze 3D arrays to vectors
mag_rlc = squeeze(mag_rlc);
phase_rlc = squeeze(phase_rlc);
mag_tl = squeeze(mag_tl);
phase_tl = squeeze(phase_tl);
mag_est = squeeze(mag_est);
phase_est = squeeze(phase_est);
mag_high = squeeze(mag_high);
phase_high = squeeze(phase_high);

% Convert magnitude to dB
mag_rlc_db = 20*log10(mag_rlc);
mag_tl_db = 20*log10(mag_tl);
mag_est_db = 20*log10(mag_est);
mag_high_db = 20*log10(mag_high);

% Create figure for model comparison
figure;

% Magnitude plot
subplot(2,1,1);
semilogx(primary_frequency, plot_magnitude, 'r', 'LineWidth', 2, 'DisplayName', 'Measured Data');
hold on;
semilogx(primary_frequency, mag_rlc_db, 'b', 'LineWidth', 1.5, 'DisplayName', 'RLC Model');
semilogx(primary_frequency, mag_tl_db, 'g', 'LineWidth', 1.5, 'DisplayName', 'Transmission Line Model');
semilogx(primary_frequency, mag_high_db, 'm--', 'LineWidth', 1.5, 'DisplayName', 'High-Order TF Model');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Model Comparison - Magnitude');
legend('show');
grid on;
xlim([min(primary_frequency), max(primary_frequency)]);

% Phase plot
subplot(2,1,2);
semilogx(primary_frequency, plot_phase, 'r', 'LineWidth', 2, 'DisplayName', 'Measured Data');
hold on;
semilogx(primary_frequency, phase_rlc, 'b', 'LineWidth', 1.5, 'DisplayName', 'RLC Model');
semilogx(primary_frequency, phase_tl, 'g', 'LineWidth', 1.5, 'DisplayName', 'Transmission Line Model');
semilogx(primary_frequency, phase_high, 'm--', 'LineWidth', 1.5, 'DisplayName', 'High-Order TF Model');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Model Comparison - Phase');
legend('show');
grid on;
xlim([min(primary_frequency), max(primary_frequency)]);

% Adjust layout
sgtitle('RX Coil Model Comparison');

% Save the optimized models
save('rx_coil_models.mat', 'H_rlc', 'H_tl_optimized', 'high_order_model', 'R_prime_opt', 'L_prime_opt', 'G_prime_opt', 'C_prime_opt');

% Function to create ladder network model for transmission line
function [num, den] = create_ladder_network(R_prime, L_prime, G_prime, C_prime, length, segments)
    % Calculate segment parameters
    segment_length = length / segments;
    R_seg = R_prime * segment_length;
    L_seg = L_prime * segment_length;
    G_seg = G_prime * segment_length;
    C_seg = C_prime * segment_length;
    
    % For voltage transfer function, start with load impedance (open circuit)
    Z_load = tf([1], [0 0 1]); % Represents infinite impedance
    
    % Build the ladder network from right (load) to left (source)
    Z_in = Z_load;
    for i = 1:segments
        % Series impedance of segment
        Z_series = tf([L_seg, R_seg], [1, 0]);
        
        % Shunt admittance of segment
        Y_shunt = tf([C_seg, G_seg], [1, 0]);
        
        % Update input impedance (voltage divider with previous Z_in)
        Z_in = Z_series + 1/(1/Z_in + Y_shunt);
    end
    
    % Convert to transfer function
    H = 1/Z_in; % Current to voltage transfer function
    
    % Extract numerator and denominator
    [num, den] = tfdata(H, 'v');
end

% Function to calculate error between model and measurements
function error = transmission_line_fit_error(params, frequencies, measured_response, length, segments)
    % Extract parameters
    R_prime = params(1);
    L_prime = params(2);
    G_prime = params(3);
    C_prime = params(4);
    
    % Create model
    [num, den] = create_ladder_network(R_prime, L_prime, G_prime, C_prime, length, segments);
    model_tf = tf(num, den);
    
    % Calculate model response at measurement frequencies
    w = 2*pi*frequencies;
    [mag, phase] = bode(model_tf, w);
    model_response = squeeze(mag) .* exp(1j * deg2rad(squeeze(phase)));
    
    % Calculate error (complex difference)
    error_complex = measured_response - model_response;
    
    % Return sum of squared errors
    error = sum(abs(error_complex).^2);
end