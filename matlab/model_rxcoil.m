R_adc = 1e6;
L_coil = 7.12e-3;
R_coil = 3.12;
C_shunt = 1e-12;
transfer = tf(1, [R_coil * C_shunt  R_coil/R_adc  R_coil/L_coil] + 1);
bode(transfer)