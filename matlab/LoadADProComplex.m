function [complex, frequency] = LoadADProComplex(path,normalization)

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
complex = transfer_magnitude .* exp(1j * deg2rad(rx_phase));

end