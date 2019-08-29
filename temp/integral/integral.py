import numpy as np
import sys

sys.path.append('../..')

import myfilemanager_sixtracklib as mfm
import matplotlib.pyplot as plt
import pickle

##################
# misc functions #
##################

def merge(filelist):
    t0_all   = np.array([])
    J1_all   = np.array([])
    J2_all   = np.array([])
    phi1_all = np.array([])
    phi2_all = np.array([])

    for myfile in filelist:
        print(' ')
        print(myfile)
        t0, J1_init, J2_init, phi1, phi2, epsg_x, epsg_y = initialization(myfile)
        t0_all   = np.append(t0_all, t0)
        J1_all   = np.append(J1_all, J1_init)
        J2_all   = np.append(J2_all, J2_init)
        phi1_all = np.append(phi1_all, phi1)
        phi2_all = np.append(phi2_all, phi2)

    turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)

    myinput = {'t0'    : t0_all,
               'J1'    : J1_all,
               'J2'    : J2_all,
               'phi1'  : phi1_all,
               'phi2'  : phi2_all,
               'turns' : turns
             }
    
    return myinput

def normalize(X, part_on_CO, invW):
    X_norm = np.zeros_like(X)
    X_norm = np.tensordot(X - part_on_CO, invW, axes = (-1,1))
    return X_norm

def K(t, t0):
    return 1 - np.heaviside(t - t0, 0)

def J(a, b):
    J = 0.5 * ( np.multiply(a, a)
              + np.multiply(b, b ))
    return J

def phi(a, b):
    phi = np.arctan(b/a)
    return phi

##################
# initialization #
##################

def initialization(myfile):
    
    # Open files
    print('Importing the files ...')
    data           = mfm.h5_to_dict(myfile, group = 'output')
    in_data        = mfm.h5_to_dict(myfile, group = 'input')
    optics         = mfm.h5_to_dict(myfile, group = 'beam-optics')
    particle_on_CO = mfm.h5_to_dict(myfile, group = 'closed-orbit')

    # Extract constants
    print('Extracting constants and variables...')
    epsg_x = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
    epsg_y = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])
    invW = optics['invW']

    # Get X
    X          = np.array([data['x_tbt_first'].T,
                           data['px_tbt_first'].T,
                           data['y_tbt_first'].T,
                           data['py_tbt_first'].T,
                           data['zeta_tbt_first'].T,
                           data['delta_tbt_first'].T]).T
    part_on_CO = np.array([particle_on_CO['x'],
                           particle_on_CO['px'],
                           particle_on_CO['y'],
                           particle_on_CO['py'],
                           particle_on_CO['zeta'],
                           particle_on_CO['delta']]).T
    X_init     = np.array([in_data['init_x'],
                           in_data['init_px'],
                           in_data['init_y'],
                           in_data['init_py'],
                           in_data['init_zeta'],
                           in_data['init_delta']]).T
    # Normalize X
    print('Normalizing...')
    X_norm      = normalize(X, part_on_CO, invW)
    X_init_norm = normalize(X_init, np.zeros_like(part_on_CO), invW)

    # Get invariants of motion J
    print('Getting invariants of motion J...')
    J1_init     =         J(X_init_norm[:, 0], X_init_norm[:, 1])
    J2_init     =         J(X_init_norm[:, 2], X_init_norm[:, 3])

    # Get angle phi
    phi1        =         phi(X_init_norm[:, 0], X_init_norm[:, 1])
    phi2        =         phi(X_init_norm[:, 2], X_init_norm[:, 3])

    # Define t0
    T = 1          # time 1 turn takes (convert n_turns to t0)
    t0 = T * data['at_turn_last']

    return t0.flatten(), J1_init, J2_init, phi1, phi2, epsg_x, epsg_y

##################
#    integral    #
##################

# get Q
def integral(t, t0, J1, J2, sigma1, sigma2):
    V_1   = 1/2 * np.pi**2 * sigma1**4 /16
    V_2   = 1/2 * np.pi**2 * sigma2**4 /16  # volume of hypersphere    
    V     = V_2 - V_1
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1        # total number of particles  ~ 10^11
    T     = 1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-4**2/2) * (1 + 4**2/2))
    g_fct = np.exp(-J1-J2)

    n     = t / T # convert time t to number of turns n

    Q = V/M * g_cst  * np.sum(np.multiply(K(t, t0), g_fct))

    return Q

# get Q^2
def integral2(t, t0, J1, J2, sigma1, sigma2):
    V_1   = 1/2 * np.pi**2 * sigma1**4 /16
    V_2   = 1/2 * np.pi**2 * sigma2**4 /16  # volume of hypersphere    
    V     = V_2 - V_1
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1        # total number of particles  ~ 10^11
    T     = 1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2))
    g_fct = np.exp(-2 * (J1 + J2))

    n     = t / T # convert time t to number of turns n

    Q2 = V**2/M * g_cst**2 * np.sum(np.multiply(K(t, t0), g_fct))

    return Q2
