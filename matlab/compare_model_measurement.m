% Load the data from CSV file
data = readtable('NetworkAnalyzerOutput-Jan12-Long.csv', 'FileType', 'delimitedtext', ...
    'HeaderLines', 19, 'NumHeaderLines', 0);
data = rmmissing(data);
% Extract frequency and measurements
frequencies_measured = data.x_DigilentWaveFormsNetworkAnalyzer_Bode;  % First column is frequency
tx_magnitude = data.Var2; % Channel 1 magnitude (Tx voltage)
rx_magnitude = data.Var3; % Channel 2 magnitude (Rx voltage)
rx_phase = data.Var4 + 360;     % Channel 2 phase

% Convert from dB to linear scale and normalize to 1V source
tx_linear = 10.^(tx_magnitude/20);
rx_linear = 10.^(rx_magnitude/20);

% Calculate transfer function (normalize to 1V source)
transfer_magnitude = rx_linear ./ tx_linear;
transfer_magnitude_db = 20*log10(transfer_magnitude);

% Load electromagnet_model parameters and run simulation
electromagnet_model;

% Get model transfer function data
w = logspace(2, 7, 1000);
rx_secondary_tf_response = squeeze(bode(rx_secondary_tf, w));
rx_secondary_tf_db = 20*log10(rx_secondary_tf_response);

rx_primary_tf_response = squeeze(bode(rx_primary_tf, w));
rx_primary_tf_db = 20*log10(rx_primary_tf_response);

% Create figure for comparison
figure;

% Plot magnitude response
subplot(3,1,1);
semilogx(frequencies_measured, transfer_magnitude_db, 'b.', 'DisplayName', 'Measured');
hold on;
semilogx(w/(2*pi), rx_secondary_tf_db, 'r-', 'DisplayName', 'Model');
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Measured vs Model Transfer Function');
legend;

% Plot phase response
subplot(3,1,2);
semilogx(frequencies_measured, rx_phase, 'b.', 'DisplayName', 'Measured');
hold on;
[~, phase_model] = bode(rx_secondary_tf, w);
[~, primary_phase] = bode(rx_primary_tf, w);

phase_model = squeeze(phase_model);
semilogx(w/(2*pi), phase_model, 'r-', 'DisplayName', 'Model');
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
legend;
% Set frequency cutoff
freq_cutoff = 20000; % 100 kHz cutoff - adjust as needed

% Filter out high frequencies
freq_mask = frequencies_measured <= freq_cutoff;
frequencies_measured = frequencies_measured(freq_mask);
transfer_magnitude = transfer_magnitude(freq_mask);
rx_phase = rx_phase(freq_mask);
tx_magnitude = tx_magnitude(freq_mask);
rx_magnitude = rx_magnitude(freq_mask);

% Interpolate theoretical response to match measured frequencies
theoretical_magnitude = interp1(w/(2*pi), rx_secondary_tf_db, frequencies_measured, 'linear', 'extrap');
theoretical_phase = interp1(w/(2*pi), phase_model, frequencies_measured, 'linear', 'extrap');

% Convert from dB to linear scale for theoretical response
theoretical_magnitude_linear = 10.^(theoretical_magnitude/20);

% Convert magnitude and phase to complex numbers
measured_complex = transfer_magnitude .* exp(1j * deg2rad(rx_phase));
theoretical_complex = theoretical_magnitude_linear .* exp(1j * deg2rad(theoretical_phase));

% Get primary field response interpolated to same frequencies
primary_magnitude = interp1(w/(2*pi), rx_primary_tf_db, frequencies_measured, 'linear', 'extrap');
primary_phase = interp1(w/(2*pi), squeeze(primary_phase), frequencies_measured, 'linear', 'extrap');
primary_complex = 10.^(primary_magnitude/20) .* exp(1j * deg2rad(primary_phase));

% Calculate secondary/primary ratio for both measured and theoretical
measured_ratio = measured_complex ./ primary_complex;
theoretical_ratio = theoretical_complex ./ primary_complex;

% Plot the results
subplot(3,1,3);
semilogx(frequencies_measured, real(measured_ratio) * 1e6, 'r.', 'DisplayName', 'Real');
hold on;
semilogx(frequencies_measured, imag(measured_ratio)* 1e6, 'b.', 'DisplayName', 'Imaginary');
grid on;
xlabel('Frequency (Hz)');
ylabel('H_s / H_p (PPM)');
title('Secondary to Primary Field Ratio');
legend;