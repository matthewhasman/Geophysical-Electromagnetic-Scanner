function [rx_magnitiude,rx_phase, frequency] = LoadADProMeasure(path,normalization)

% Load the data from CSV file
data = readtable(path, 'FileType', 'delimitedtext', ...
    'HeaderLines', 19, 'NumHeaderLines', 0);
data = rmmissing(data);
% Extract frequency and measurements
frequency = data.x_DigilentWaveFormsNetworkAnalyzer_Bode;  % First column is frequency
rx_magnitude = data.Var3; % Channel 2 magnitude (Rx voltage)
rx_phase = data.Var4 + 360;     % Channel 2 phase

% Convert from dB to linear scale and normalize to 1V source
rx_linear = 10.^(rx_magnitude/20);

% Calculate transfer function (normalize to 1V source)
transfer_magnitude = rx_linear ./ normalization;
rx_magnitiude = 20*log10(transfer_magnitude);

end