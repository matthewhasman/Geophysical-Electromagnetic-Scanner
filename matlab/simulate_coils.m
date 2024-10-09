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

% Define the centers of the coils
centerTx = [0, distance/2, 0]; % Center of the first coil
centerBucking = [0, -distance/2 + 0.25, 0]; % Center of the second coil

% Create the circular loops (using a cylindrical representation)
coilTx = multicylinder(coilTxRadius, heightTx); % Thickness of 0.01 m
coilTx = translate(coilTx, centerTx);

coilBucking = multicylinder(coilBuckingRadius, heightBucking);
coilBucking = translate(coilBucking, centerBucking);

air = multicuboid(1, 2, 1, 'Zoffset', -0.5);

gm = addCell(air, coilTx);
gm = addCell(gm, coilBucking);

model = femodel(AnalysisType="magnetostatic", ...
                Geometry=gm);
model.VacuumPermeability = 1.2566370614E-6;

model.MaterialProperties = ...
    materialProperties(RelativePermeability=1);

function f3D = coilWinding(center, width, height, numTurns, current)
    % center: Center of the coil [x, y, z]
    % radius: Radius of the coil (m)
    % numTurns: Number of turns in the coil
    % current: Current through the coil (A)
    
    % Calculate the cross-sectional area of the coil
    area = width*height;
    
    % Compute the surface current density J (A/m)
    Jsurface = (current * numTurns) / area;
    
    % Define the function for current density using an anonymous function
    f3D = @(region,~) localTxCoilWinding(region, center, Jsurface);
end

function currentDensity = localTxCoilWinding(region, center, Jsurface)
    % Calculate the current density based on the center of the coil and surface current density
    [TH, ~, ~] = cart2pol(region.x, region.y - center(2), region.z - center(3));
    % Create the current density vector using cylindrical coordinates (only in the xy plane)
    currentDensity = Jsurface * [-sin(TH); cos(TH); zeros(size(TH))];
end



model.CellLoad(2) = ...
    cellLoad(CurrentDensity=coilWinding(centerTx, widthTx, heightTx, numTurnsTx, current));

model.CellLoad(4) = ...
    cellLoad(CurrentDensity=coilWinding(centerBucking, widthBucking, heightBucking, numTurnsBucking, -current));


pdegplot(model.Geometry,FaceAlpha=0.5, ...
                        FaceLabels="on")

model.FaceBC(1:6) = faceBC(MagneticPotential=[0;0;0]);

internalFaces = cellFaces(model.Geometry,2:5);
model = generateMesh(model,Hface={internalFaces,0.007});

R = solve(model);

x = -0.5:0.1:0.5;
z = -0.5:0.1:0.5;
y = -1:0.1:1;
[X,Y,Z] = meshgrid(x,y,z);
intrpBcore = R.interpolateMagneticFlux(X,Y,Z); 

Bx = reshape(intrpBcore.Bx,size(X));
By = reshape(intrpBcore.By,size(Y));
Bz = reshape(intrpBcore.Bz,size(Z));

B_mag = sqrt(Bx.^2 + By.^2 + Bz.^2);
logB_mag = log10(B_mag + 1e-12);

quiver3(X,Y,Z,Bx./B_mag.*logB_mag,By./B_mag.*logB_mag,Bz.*logB_mag./B_mag,Color="r")
hold on
pdegplot(gm,FaceAlpha=0.2);

% Create a contour plot of Bz in the XY plane
figure;
contourf(X(:,:,1), Y(:,:,1), Bz(:,:,1), 'LineColor', 'none');
colorbar;
xlabel('X (m)');
ylabel('Y (m)');
title('Intensity of B_z in the XY plane at Z = 0');

