load('clean_primary_data_243.mat')
far_real = data_real;
far_imag = data_imag;

far_mag = log10(abs(far_real + 1j * far_imag)) * 20;

load('clean_primary_data_19.mat')

close_real = data_real;
close_imag = data_imag;

close_mag = log10(abs(close_real + 1j * close_imag)) * 20;

figure()
subplot(2,1,1)
semilogx(frequencies, log10(far_real + 1j * far_imag -  close_real + 1j * close_imag) * 20)

hold on
semilogx(frequencies, far_mag)
semilogx(frequencies, close_mag)
hold off

subplot(2,1,2)
semilogx(frequencies, angle(far_real + 1j * far_imag -  close_real + 1j * close_imag))
hold on
semilogx(frequencies, angle(far_real + 1j * far_imag))
semilogx(frequencies, angle(close_real + 1j * close_imag))

