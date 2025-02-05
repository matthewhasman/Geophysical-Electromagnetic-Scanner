[primary_magnitude, primary_phase, primary_frequency] = LoadADProMeasure("NetworkAnalyzerOutput-Jan26-Primary-Fine.csv", 20);
[secondary_magnitude, secondary_phase, secondary_frequency] = LoadADProMeasure("NetworkAnalyzerOutput-Jan26-Secondary-Fine.csv", 20);

%% Procedure is as follows 
% Measure response of system with bucking coil (secondary + bucking error)
% Measure system without bucking coil (primary + secondary)
% Convert both measurements to linear imaginary units
% Subtract secondary from primary to get (primary - bucking error)
% Assuming bucking error << primary then we have ideal primary curve 


primary_linear = 10.^(primary_magnitude ./ 20);
secondary_linear = 10.^(secondary_magnitude ./ 20);

primary_complex = primary_linear .* exp(1j * deg2rad(primary_phase));
secondary_complex = secondary_linear .* exp(1j * deg2rad(secondary_phase));

%% Interpolate the two values to same frequency basis

% Set frequency cutoff
freq_cutoff = 100000; % 100 kHz cutoff - adjust as needed

% Filter out high frequencies
freq_mask = primary_frequency <= freq_cutoff;
primary_frequency = primary_frequency(freq_mask);
secondary_frequency = secondary_frequency(freq_mask);

primary_complex = primary_complex(freq_mask);
secondary_complex = secondary_complex(freq_mask);



% First decide which frequency basis to use - let's use primary as reference
target_frequency = primary_frequency;

% Interpolate secondary complex values to match primary frequency points
%secondary_complex_interp = interp1(secondary_frequency, secondary_complex, target_frequency, 'linear');


clean_primary = primary_complex - secondary_complex;

% Create a figure with subplots for complex response and ratios
figure('Position', [100, 100, 1000, 800]);

% Calculate ratios (using interpolated secondary)
measured_ratio = secondary_complex ./ primary_complex;
clean_ratio = secondary_complex ./ clean_primary;

%% Save Clean Primary to .mat file
% Prepare the data to save
window_size = 20;
data_real = smoothdata(real(clean_primary));
data_imag = smoothdata(imag(clean_primary));
frequencies = target_frequency;

% Create timestamp for filename
timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
filename = sprintf('clean_primary_data_%s.mat', timestamp);

filename2 = sprintf('dirty_primary_data_%s.mat', timestamp);

% Save the data
% save(filename, 'data_real', 'data_imag', 'frequencies');

data_real = real(primary_complex);
data_imag = imag(primary_complex);

save(filename2, 'data_real', 'data_imag', 'frequencies');


% Use a colorblind-friendly, professional palette
color1 = [0.8500, 0.3250, 0.0980];  % Deep orange - measured real
color2 = [0.4940, 0.1840, 0.5560];  % Purple - measured imaginary
color3 = [0.4660, 0.6740, 0.1880];  % Green - clean real
color4 = [0.3010, 0.7450, 0.9330];  % Light blue - clean imaginary

measured_db = 20 * log10(abs(primary_complex));
clean_db = 20 * log10(abs(clean_primary));

semilogx(target_frequency, measured_db, '-', 'Color', color1, 'LineWidth', 2, 'DisplayName', 'Measured Primary Magnitude');
hold on;
semilogx(target_frequency, clean_db, '-.', 'Color', color3, 'LineWidth', 2, 'DisplayName', 'Clean Primary Magnitude');
hold off

grid on;
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Magnitude Ratio', 'FontSize', 12);
title('Subtracted Secondary Response from  Primary', 'FontSize', 14);
legend('Location', 'best');
set(gca, 'FontSize', 12);
%ylim([0, max(max(abs(primary_complex)), max(abs(clean_primary)))*1.1]);  % Set y-axis limits with 10% padding

% Adjust spacing between subplots
set(gcf, 'Color', 'white');
set(findall(gcf,'-property','FontName'), 'FontName', 'Arial');
