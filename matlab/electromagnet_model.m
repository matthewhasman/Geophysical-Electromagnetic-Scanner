% Electromagnet Parameters of Tx coil
r = 0.12;         % Radius of coil (m)
N = 40;         % Number of turns
wireGauge = 26;   % AWG wire gauge
rho = 1.68e-8;    % Resistivity of copper (ohm·m)
driver_voltage = 1;

% Electromagnet Parameters of Rx coil
r2 = 0.05;         % Radius of coil (m)
N2 = 200;         % Number of turns

% Physical system parameters
coil_distance = 0.86;  % Intercoil distance (m)

bucking_constant = 0.01; % 1 percent for testing

% Wire properties based on gauge
wireProperties = struct(...
    'diameter', 2*sqrt((0.012668 * 92^((36 - wireGauge)/19.5) * 1e-6)/pi), ...  % Wire diameter (m) for AWG 22
    'area', 0.012668 * 92^((36 - wireGauge)/19.5) * 1e-6 ... % Wire cross-sectional area (m^2)
);

% Calculate resistance
wireLength = 2 * pi * r * N;  % Total wire length
R = rho * wireLength / wireProperties.area;

% Calculate inductance (approximate formula for long solenoid)
mu0 = 4*pi*1e-7;  % Permeability of free space
A = pi * r^2;     % Cross-sectional area of coil
A2 = pi * r2^2;
L = (mu0 * N^2 * A) * (log(16*A / wireProperties.diameter) - 1.75);

% Using values found from measured coil
%L = 0.00099;
%R = 4.584;

tf_num = 1;
tf_den = [L R];

% Curve fit data from simpeg to estimate a simple transfer function
load('data_real_imag.mat')
response_hs_hp = data_real + 1j * data_imag;

data = idfrd(response_hs_hp, frequencies, 0, 'FrequencyUnit', 'Hz'); % '0' indicates zero delay

% system parameters 
% Define the order of the system: 1st order (1 numerator, 1 denominator)
numPoles = 5; % Number of poles
numZeros = 5; % Number of zeros for a first-order system (adjust if necessary)

% Estimate the transfer function
secondary_response_tf = tfest(data, numPoles, numZeros);

% Estimage background noise levels
magnetic_snd = 0.2e-9; % 0.2 nT / hz

% Display calculated values
fprintf('Calculated Resistance: %.2f ohms\n', R);
fprintf('Calculated Inductance: %.6f H\n', L);
fprintf('Calculated Cutoff frequency: %.6f Hz\n', R/(2*pi*L));
% Optional: Run simulation and plot frequency response
tx_tf = tf(tf_num, tf_den) * driver_voltage;
w = logspace(2, 7, 1000);
[current, phase] = bode(tx_tf, w);
current = squeeze(current);

fprintf('Peak Current: %.6f Amps\n', max(current));

% calculate noise floor seen at the Rx coil
noise_floor = 10^(-100/20) * ones(1,length(w)); % Measured value maxxing out at -72 dbV

% calculated divergence factor of between coils and dipoles for
% configuration
geometric_factor = 1.0;

rx_primary_tf = tf([ -N*N2*A*A2,0],coil_distance^3) * tx_tf * 1e-7 * geometric_factor;
[voltage_primary, phase_primary] = bode(rx_primary_tf, w);
voltage_primary = squeeze(voltage_primary);
phase_primary = squeeze(phase_primary);

rx_secondary_tf = rx_primary_tf * secondary_response_tf;
[voltage_secondary, phase_secondary] = bode(rx_secondary_tf, w);
voltage_secondary = squeeze(voltage_secondary);

[voltage_combined, phase_combined] = bode(rx_secondary_tf + (rx_primary_tf  * bucking_constant), w);
voltage_combined = squeeze(voltage_combined);

[voltage_bucked, phase_bucked] = bode(rx_primary_tf  * bucking_constant, w);
voltage_bucked = squeeze(voltage_combined);


[hs_hp, phase] = bode(secondary_response_tf, w);
hs_hp = squeeze(hs_hp);

figure;
% subplot(3,1,1);
% semilogx(w/(2*pi), current * N * A, 'LineWidth', 2);
% grid on;
% title('Frequency Response of Electromagnet');
% xlabel('Frequency (Hz)');
% ylabel('Dipole Moment (A m^2)');

subplot(2,1,1);
semilogx(w/(2*pi),20*log10(voltage_primary), 'LineWidth', 2);
hold on
semilogx(w/(2*pi),20*log10(voltage_secondary), 'LineWidth', 2);
semilogx(w/(2*pi),20*log10(voltage_combined), 'LineWidth', 2);

hold off
grid on;
title('System Frequency Response');
xlabel('Frequency (Hz)');
ylabel('Power (Db)');
legend('primary', 'secondary', 'combined bucked')

subplot(2,1,2);
semilogx(w/(2*pi),squeeze(phase_primary), 'LineWidth', 2);
hold on
semilogx(w/(2*pi),squeeze(phase_secondary), 'LineWidth', 2);
semilogx(w/(2*pi),squeeze(phase_combined), 'LineWidth', 2);
title('Phase shift');
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
legend('primary', 'secondary', 'combined bucked')
