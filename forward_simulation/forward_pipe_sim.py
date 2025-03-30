import numpy as np

def mind(x, y, z, dincl, ddecl, x0, y0, z0, aincl, adecl):
    """
    Calculate the magnetic field of a magnetic dipole at a given location
    
    Parameters
    ----------
    x : float
        x location of the observation point
    y : float
        y location of the observation point
    z : float
        z location of the observation point
    dincl : float
        inclination of the dipole moment (degrees)
    ddecl : float
        declination of the dipole moment (degrees)
    x0 : float
        x location of the dipole
    y0 : float
        y location of the dipole
    z0 : float
        z location of the dipole
    aincl : float
        inclination of the inducing field (degrees)
    adecl : float
        declination of the inducing field (degrees)
    
    Returns
    -------
    float
        The magnetic field at the observation point due to the dipole
    """

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    x0 = np.array(x0, dtype=float)
    y0 = np.array(y0, dtype=float)
    z0 = np.array(z0, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    aincl = np.array(aincl, dtype=float)
    adecl = np.array(adecl, dtype=float)

    di = np.pi * dincl / 180.0
    dd = np.pi * ddecl / 180.0

    cx = np.cos(di) * np.cos(dd)
    cy = np.cos(di) * np.sin(dd)
    cz = np.sin(di)

    ai = np.pi * aincl / 180.0
    ad = np.pi * adecl / 180.0

    ax = np.cos(ai) * np.cos(ad)
    ay = np.cos(ai) * np.sin(ad)
    az = np.sin(ai)

    # begin the calculation
    a = x - x0
    b = y - y0
    h = z - z0

    rt = np.sqrt(a ** 2.0 + b ** 2.0 + h ** 2.0) ** 5.0

    txy = 3.0 * a * b / rt
    txz = 3.0 * a * h / rt
    tyz = 3.0 * b * h / rt

    txx = (2.0 * a ** 2.0 - b ** 2.0 - h ** 2.0) / rt
    tyy = (2.0 * b ** 2.0 - a ** 2.0 - h ** 2.0) / rt
    tzz = -(txx + tyy)

    bx = txx * cx + txy * cy + txz * cz
    by = txy * cx + tyy * cy + tyz * cz
    bz = txz * cx + tyz * cy + tzz * cz

    return bx * ax + by * ay + bz * az

def rotate_coordinates(coords, incl, decl):
    """
    Rotate coordinates based on inclination and declination angles.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (3, N) containing the coordinates to be rotated.
    incl : float
        Inclination angle in degrees.
    decl : float
        Declination angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated coordinates of the same shape as input.
    """
    # convert to radians
    incl = np.deg2rad(incl)
    decl = np.deg2rad(decl)

    # Rotation matrix for inclination (around y-axis)
    R_incl = np.array([
        [np.cos(incl), 0, np.sin(incl)],
        [0, 1, 0],
        [-np.sin(incl), 0, np.cos(incl)]
    ])
    
    # Rotation matrix for declination (around z-axis)
    R_decl = np.array([
        [np.cos(decl), -np.sin(decl), 0],
        [np.sin(decl), np.cos(decl), 0],
        [0, 0, 1]
    ])
    
    # Apply rotation
    rotated_coords = R_decl @ (R_incl @ coords)
    
    return rotated_coords

def fem_pipe(
        sigma, mu, xc, yc, zc, dincl, ddecl, dipole_dist, sensor_ht, f, xmin, xmax, dx, N_loops, length, rad, 
):
    """
    Forward simulation of a pipe using the finite element method.
    
    Parameters
    ----------
    sigma : float or array-like
        Conductivity of the medium.
    mu : float or array-like
        Permeability of the medium.
    xc : float or array-like
        x-coordinate of the center of the pipe.
    yc : float or array-like
        y-coordinate of the center of the pipe.
    zc : float or array-like
        z-coordinate of the center of the pipe.
    dincl : float or array-like
        Inclination of the dipole moment (degrees).
    ddecl : float or array-like
        Declination of the dipole moment (degrees).
    dipole_dist : float or array-like
        Distance between the transmitter and receiver dipoles.
    sensor_ht : float or array-like
        Height of the sensor above the ground.
    f : float or array-like
        Frequency of the electromagnetic field.
    xmin : float
        Minimum x-coordinate of the grid.
    xmax : float
        Maximum x-coordinate of the grid.
    dx : float
        Grid spacing in the x-direction.
    N_loops : int
        Number of loops in the pipe.
    length : float
        Length of the pipe.
    rad : float
        Radius of the pipe.

    Returns
    -------
    np.ndarray
        The computed response of the pipe at the specified frequencies.
    """
    
    sigma = np.array(sigma, dtype=float)
    mu = np.array(mu, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    dipole_dist = np.array(dipole_dist, dtype=float)
    sensor_ht = np.array(sensor_ht, dtype=float)
    f = np.array(f, dtype=float)
    dx = np.array(dx, dtype=float)
    N_loops = np.array(N_loops, dtype=int)
    length = np.array(length, dtype=float)
    rad = np.array(rad, dtype=float)

    skin_depth = np.sqrt(1 / (np.pi * mu * f * sigma))
    R = N_loops * 2 * np.pi * rad / (sigma * length * skin_depth)
    L = mu * N_loops * np.pi * rad**2 / length

    ymin = xmin
    ymax = xmax
    dy = dx

    # generate the grid
    xp = np.arange(xmin, xmax, dx)
    yp = np.arange(ymin, ymax, dy)
    [y, x] = np.meshgrid(yp, xp)
    z = 0.0 * x - sensor_ht

    # frequency characteristics
    alpha = 2.0 * np.pi * f * L / R
    f_factor = (alpha ** 2.0 + 1j * alpha) / (1 + alpha ** 2.0)

    # compute distances
    y_tx = y - dipole_dist / 2.0
    y_rx = y + dipole_dist / 2.0

    # define the cylinder
    x_centers = np.linspace(-length / 2, length / 2, N_loops) + xc
    loop_centers = np.c_[x_centers, np.zeros(N_loops) + yc, np.zeros(N_loops) + zc]
    loop_centers = rotate_coordinates(loop_centers.T, dincl, ddecl).T
    
    # compute the response
    MTR = mind(0.0, -dipole_dist / 2.0, 0.0, 90.0, 0.0, 0.0, dipole_dist / 2.0, 0.0, 90.0, 0.0)

    mut_ind = 0
    for i in range(N_loops):
        xci, yci, zci = loop_centers[i, 0], loop_centers[i, 1], loop_centers[i, 2]
        MTk = mind(x, y_tx, z, 90.0, 0.0, xci, yci, zci, dincl, ddecl)
        MkR = mind(xci, yci, zci, dincl, ddecl, x, y_rx, z, 90.0, 0.0)
        mut_ind += MTk * MkR

    freq_dependent = f_factor * L
    freq_channels = freq_dependent.shape[0]
    c_response = -mut_ind[:,:,np.newaxis] * freq_dependent.reshape(1,1,freq_channels)/ (MTR)
    return c_response

def fem_pipe_coupled(
        sigma, mu, xc, yc, zc, dincl, ddecl, dipole_dist, sensor_ht, f, xmin, xmax, dx, N_loops, length, rad, 
):
    """"
    Forward simulation of a pipe using the finite element method with coupled response."
    
    Parameters
    ----------
    sigma : float or array-like
        Conductivity of the medium.
    mu : float or array-like
        Permeability of the medium.
    xc : float or array-like
        x-coordinate of the center of the pipe.
    yc : float or array-like
        y-coordinate of the center of the pipe.
    zc : float or array-like
        z-coordinate of the center of the pipe.
    dincl : float or array-like
        Inclination of the dipole moment (degrees).
    ddecl : float or array-like
        Declination of the dipole moment (degrees).
    dipole_dist : float or array-like
        Distance between the transmitter and receiver dipoles.
    sensor_ht : float or array-like
        Height of the sensor above the ground.
    f : float or array-like
        Frequency of the electromagnetic field.
    xmin : float
        Minimum x-coordinate of the grid.
    xmax : float
        Maximum x-coordinate of the grid.
    dx : float
        Grid spacing in the x-direction.
    N_loops : int
        Number of loops in the pipe.
    length : float
        Length of the pipe.
    rad : float
        Radius of the pipe.

    Returns
    -------
    np.ndarray
        The computed response of the pipe at the specified frequencies.
    """
    
    sigma = np.array(sigma, dtype=float)
    mu = np.array(mu, dtype=float)
    xc = np.array(xc, dtype=float)
    yc = np.array(yc, dtype=float)
    zc = np.array(zc, dtype=float)
    dincl = np.array(dincl, dtype=float)
    ddecl = np.array(ddecl, dtype=float)
    dipole_dist = np.array(dipole_dist, dtype=float)
    sensor_ht = np.array(sensor_ht, dtype=float)
    freq = np.array(f, dtype=float)
    dx = np.array(dx, dtype=float)
    N_loops = np.array(N_loops, dtype=int)
    length = np.array(length, dtype=float)
    rad = np.array(rad, dtype=float)

    skin_depth = np.sqrt(1 / (np.pi * mu * f * sigma))
    R = N_loops * 2 * np.pi * rad / (sigma * length * skin_depth)
    L = mu * N_loops * np.pi * rad**2 / length

    ymin = xmin
    ymax = xmax
    dy = dx

    # generate the grid
    xp = np.arange(xmin, xmax, dx)
    yp = np.arange(ymin, ymax, dy)
    [y, x] = np.meshgrid(yp, xp)
    z = 0.0 * x - sensor_ht

    # define the cylinder
    x_centers = np.linspace(-length / 2, length / 2, N_loops) + xc
    loop_centers = np.c_[x_centers, np.zeros(N_loops) + yc, np.zeros(N_loops) + zc]
    loop_centers = rotate_coordinates(loop_centers.T, dincl, ddecl).T
    
    # compute the response
    MTR = mind(0.0, -dipole_dist / 2.0, 0.0, 90.0, 0.0, 0.0, dipole_dist / 2.0, 0.0, 90.0, 0.0)

    def solve_hshp(f, x_val, y_val, z_val):
        # compute distances
        y_tx = y_val - dipole_dist / 2.0
        y_rx = y_val + dipole_dist / 2.0

        current_matrix = np.zeros((N_loops, N_loops), dtype=complex)
        transmit_vector = np.zeros((N_loops,1), dtype=complex)
        reciever_vector = np.zeros(N_loops, dtype=complex)
        for i in range(N_loops):
            xci, yci, zci = loop_centers[i, 0], loop_centers[i, 1], loop_centers[i, 2]
            MTi = mind(x_val, y_tx, z_val, 90.0, 0.0, xci, yci, zci, dincl, ddecl)
            MiR = mind(xci, yci, zci, dincl, ddecl, x_val, y_rx, z_val, 90.0, 0.0)
            transmit_vector[i,0] = 1j*2*np.pi*f*MTi
            reciever_vector[i] = MiR
            for j in range(N_loops):
                xcj, ycj, zcj = loop_centers[j, 0], loop_centers[j, 1], loop_centers[j, 2]
                if i == j:
                    current_matrix[i, j] = R + 1j*2*np.pi*f*L
                else:
                    Mij = mind(xci, yci, zci, dincl, ddecl, xcj, ycj, zcj, dincl, ddecl)
                    current_matrix[i, j] = -1j*2*np.pi*f*Mij
        
        # invert the matrix to solve for the currents
        currents = np.linalg.solve(current_matrix, transmit_vector)
        
        return np.sum(reciever_vector * currents) / MTR
    
    def hshp_for_all_pts(f):
        c_response = np.zeros(x.shape, dtype=complex)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                c_response[i,j] = solve_hshp(f, x[i,j], y[i,j], z[i,j])
        return c_response

    if freq.size == 1:
        c_response = hshp_for_all_pts(freq)
    else:
        c_response = np.zeros(freq.size, dtype=complex)
        for nf, f in enumerate(freq):
            c_response[nf] = hshp_for_all_pts(f)

    return c_response