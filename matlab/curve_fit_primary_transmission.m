[primary_magnitude, primary_phase, primary_frequency] = LoadADProMeasure("NewDevicePrimary4.csv", 1);
primary_phase = mod(primary_phase, 360);
% TX coil parameters (Initial Guess)
res_Tx = 4; % Ohms
L_Tx = 0.000892; % Henries
N_Tx = 40; % Number of turns
r_Tx = 0.114; % Radius (m)
a_Tx = r_Tx^2 * pi;
% Electromagnet Parameters of Rx coil
r_Rx = 0.05; % Radius of coil (m)
N_Rx = 250; % Number of turns
coil_distance = 0.9; % Intercoil distance (m)
a_Rx = r_Rx^2 * pi;
tx_tf = tf(1, [L_Tx res_Tx]);
rx_primary_tf = tf([ -N_Tx*N_Rx*a_Tx*a_Rx,0],coil_distance^3) * tx_tf * 1e-7;
% Convert primary measurement to linear units
primary_linear = 10.^(primary_magnitude ./ 20);
primary_complex = primary_linear .* exp(1j * deg2rad(primary_phase));
% Set frequency cutoff
freq_cutoff = 1500; % 20 kHz cutoff - adjust as needed
% Filter out high frequencies

freq_mask = primary_frequency <= freq_cutoff;
masked_frequency = primary_frequency(freq_mask);
primary_complex = primary_complex(freq_mask);
% Estimate the transfer function
data = idfrd(primary_complex, masked_frequency, 0, 'FrequencyUnit', 'Hz'); % '0' indicates zero delay
primary_estimate_tf = tfest(data, 1, 1);
% Generate frequency vector in rad/s
plot_frequency = primary_frequency;
plot_magnitude = primary_magnitude;
plot_phase = primary_phase;
w = 2 * pi * plot_frequency;
% Compute frequency responses
[mag1, phase1] = bode(rx_primary_tf, w);
[mag2, phase2] = bode(primary_estimate_tf, w);
% Squeeze 3D arrays to vectors
mag1 = squeeze(mag1);
phase1 = squeeze(phase1);
mag2 = squeeze(mag2);
phase2 = squeeze(phase2);
% Convert magnitude to dB
mag1_db = 20*log10(mag1);
mag2_db = 20*log10(mag2);

% Calculate isolated_rx
isolated_rx = 10.^((plot_magnitude - mag2_db)/20.0) .* exp(1j*deg2rad(plot_phase - phase2));

freq_mask2 = primary_frequency > 1000;
masked_frequency2 = primary_frequency(freq_mask2);
isolated_rx_cutoff = isolated_rx(freq_mask2);

rxdata = idfrd(isolated_rx_cutoff, masked_frequency2, 0, 'FrequencyUnit', 'Hz'); % '0' indicates zero delay
rx_estimate = tfest(rxdata, 6);
[mag3, phase3] = bode(rx_estimate, w);
mag3 = squeeze(mag3);
mag3_db = 20*log10(mag3);
phase3 = squeeze(phase3);

% Get the Total Fitted Model
total_model = rx_estimate * primary_estimate_tf;
[mag4, phase4] = bode(total_model, w);
mag4 = squeeze(mag4);
mag4_db = 20*log10(mag4);
phase4 = squeeze(phase4);



% Create figure
figure;
% Magnitude plot
subplot(4,1,1);
semilogx(plot_frequency, mag1_db, 'b', 'LineWidth', 1.5);
hold on;
semilogx(plot_frequency, mag2_db, 'k--', 'LineWidth', 1.5);
semilogx(plot_frequency, plot_magnitude, 'r', 'LineWidth', 1.5);
semilogx(plot_frequency, mag4_db, 'r--', 'LineWidth', 1.5);

xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Transfer Function Comparison');
legend('Theoretical Model', 'Estimated Model', 'Recorded Values');
grid on;
xlim([min(plot_frequency), max(plot_frequency)]);

% Phase plot
subplot(4,1,2);
semilogx(plot_frequency, phase1, 'b', 'LineWidth', 1.5);
hold on;
semilogx(plot_frequency, phase2, 'k--', 'LineWidth', 1.5);
semilogx(plot_frequency, plot_phase, 'r', 'LineWidth', 1.5);
semilogx(plot_frequency, phase4, 'r', 'LineWidth', 1.5);

xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
legend('Theoretical Model', 'Estimated Model', 'Recorded Values');
grid on;
xlim([min(plot_frequency), max(plot_frequency)]);

% Residual magnitude plot
subplot(4,1,3);
semilogx(plot_frequency, 20*log10(abs(isolated_rx)), 'r', 'LineWidth', 1.5);
hold on;
semilogx(plot_frequency, mag3_db, 'k--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Transmission Line Model Fit - Magnitude');
legend('Isolated Response', 'Transmission Line Model');
grid on;
xlim([min(plot_frequency), max(plot_frequency)]);

% Residual phase plot
subplot(4,1,4);
semilogx(plot_frequency, rad2deg(angle(isolated_rx)), 'r', 'LineWidth', 1.5);
hold on;
semilogx(plot_frequency, phase3, 'k--', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Transmission Line Model Fit - Phase');
legend('Isolated Response', 'Transmission Line Model');
grid on;
xlim([min(plot_frequency), max(plot_frequency)]);

final_model = rx_estimate * primary_estimate_tf;
save('primary_curve_fit.mat', "primary_estimate_tf");
num = final_model.Numerator;
den = final_model.Denominator;
save('primary_curve_fit_python.mat', 'num', 'den');