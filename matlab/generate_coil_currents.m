function current_sources = generate_coil_currents(center, radius, turns, resolution, current_magnitude)
    % Generates current sources arranged in a coil pattern
    % 
    % Parameters:
    % center - [x, y] coordinates for the center of the coil
    % radius - Radius of the coil (meters)
    % turns - Number of turns in the coil
    % resolution - Number of current points per turn
    % current_magnitude - Magnitude of the current in each segment (Amps)
    %
    % Returns:
    % current_sources - Array of current source points in the format [x, y, I_x, I_y]

    % Preallocate the array for current sources
    total_points = turns * resolution;
    current_sources = zeros(total_points, 5);

    theta_step = 2 * pi / resolution; % Angular step for each point
    
    % Generate the coil
    idx = 1; % Index for storing current source points
    for turn = 1:turns
        for n = 0:(resolution - 1)
            theta = (n + (turn - 1) * resolution) * theta_step; % Angle for current segment

            % Calculate current source position on the coil
            x = center(1) + radius * cos(theta);
            y = center(2) + radius * sin(theta);

            % Calculate direction of the current (tangential)
            I_x = -current_magnitude * sin(theta); % x-component of current
            I_y = current_magnitude * cos(theta);  % y-component of current

            % Store in the array
            dL = theta_step*radius;
            current_sources(idx, :) = [x, y, I_x, I_y, dL];
            idx = idx + 1;
        end
    end
end
