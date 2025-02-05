primary_est = load('primary_curve_fit.mat');
path = "NetworkAnalyzerOutput-Jan26-Secondary-Fine.csv";
frequency_cutoff = 70000;
[secondary_complex, frequency] = LoadADProComplex(path, 20);
primary_est = primary_est.primary_estimate_tf;
w = 2 * pi * frequency;
[mag, phase] = bode(primary_est, w);
mag = squeeze(mag);
phase = squeeze(phase);
primary_complex = mag .* exp(1j * deg2rad(phase));

% Filter out high frequencies
freq_mask = frequency <= frequency_cutoff;
masked_frequency = frequency(freq_mask);
primary_complex = primary_complex(freq_mask);
secondary_complex = secondary_complex(freq_mask);

ratio_complex = secondary_complex ./ primary_complex;

figure;
semilogx(masked_frequency, real(secondary_complex./primary_complex), LineWidth=2.0)
hold on
semilogx(masked_frequency, imag(secondary_complex./primary_complex), LineWidth=2.0)
hold off
legend("Real", "Imaginary", Location="southeast")
ylabel("Hs/Hp ratio")
xlabel("Frequency (hz)")
xlim([min(masked_frequency), max(masked_frequency)]);
title("measured Hs/Hp ratio")
grid on;
ax = gca;
copygraphics(ax)



