primary = load('clean_primary_data_243.mat');
frequencies = primary.frequencies;
primary_real = primary.data_real;
primary_imag = primary.data_imag;

[secondary_magnitude, secondary_phase, secondary_frequency] = LoadADProMeasure("MetalSide.csv", 20);
secondary_linear = 10.^(secondary_magnitude ./ 20);
secondary_complex = secondary_linear .* exp(1j * deg2rad(secondary_phase));

% Interpolate secondary data to match primary frequencies
primary_real = interp1(frequencies, primary_real, secondary_frequency, 'pchip');
primary_imag = interp1(frequencies, primary_imag, secondary_frequency, 'pchip');

% Reconstruct interpolated complex secondary data
secondary_interpolated = secondary_complex;

% Calculate primary complex data
primary_complex = primary_real + 1j * primary_imag;

% Calculate ratio (transfer function)
transfer_function = secondary_interpolated ./ primary_complex;

% Create figure for real component ratio
figure('Position', [100, 100, 800, 600]);

% Plot real component
subplot(2,1,1);
semilogx(secondary_frequency, real(transfer_function), 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Real Component Ratio');
title('Real Component of Secondary/Primary Ratio');

% Plot imaginary component
subplot(2,1,2);
semilogx(secondary_frequency, imag(transfer_function), 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Imaginary Component Ratio');
title('Imaginary Component of Secondary/Primary Ratio');

% Adjust spacing between subplots
spacing = 0.1;
set(gcf, 'Units', 'normalized');
set(gcf, 'Position', [0.1, 0.1, 0.8, 0.8]);