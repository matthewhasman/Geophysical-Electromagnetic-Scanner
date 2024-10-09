clear
radiusTx = 0.12; % radius fo tx coil
widthTx = 0.01; % width of tx coil, defined as radius as the outer edge;
heightTx = 0.01; % height of tx coil in meters (approximating as solid square of current)
numTurnsTx = 80;

radiusBucking = 0.03;
widthBucking = 0.01;
heightBucking = 0.01;
numTurnsBucking = 5;

current = 0.6; % amps
% Geometry: Create two circular coils with a radius of 0.1 m, spaced 1 m apart
coilTxRadius = [radiusTx, radiusTx - widthTx]; % Radius of the coils (in meters)
coilBuckingRadius = [radiusBucking, radiusBucking - widthBucking];
distance = 1; % Distance between coils (in meters)

% Geometry: Create two circular coils with a radius of 0.1 m, spaced 1 m apart
coilTxRadius = [radiusTx, radiusTx - widthTx]; % Radius of the coils (in meters)
coilBuckingRadius = [radiusBucking, radiusBucking - widthBucking];
distance = 1; % Distance between coils (in meters)

% Define the centers of the coils
centerTx = [0, distance/2]; % Center of the first coil
centerBucking = [0, -distance/2 + 0.25]; % Center of the second coil

% Create the circular loops (using a cylindrical representation)
coilTx = multicylinder(coilTxRadius, heightTx); % Thickness of 0.01 m
coilTx = translate(coilTx, centerTx);

coilBucking = multicylinder(coilBuckingRadius, heightBucking);
coilBucking = translate(coilBucking, centerBucking);

air = multicuboid(1, 2, 1, 'Zoffset', -0.5);