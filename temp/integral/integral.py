import numpy as np
import sys

sys.path.append('../..')

import myfilemanager_sixtracklib as mfm
import matplotlib.pyplot as plt
import pickle

myfile = 'test_delta0_10000parts.h5'


##################
# misc functions #
##################

def normalize(X, part_on_CO, invW):
    X_norm = np.zeros_like(X)
    X_norm = np.tensordot(X - part_on_CO, invW, axes = (-1,1))
    return X_norm

def K(t, t0):
    K = np.zeros_like(t0)
    for i in range(t0.shape[0]):
        if t < t0[i]:
            K[i] = 1
        else:
            K[i] = 0
    return K

def J(a, b):
    J = 0.5 * ( np.multiply(a, a)
              + np.multiply(b, b ))
    return J


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
    J1_avg      = np.mean(J(X_norm[:, :, 0],   X_norm[:, :, 1]),  axis = 0)
    J2_avg      = np.mean(J(X_norm[:, :, 2],   X_norm[:, :, 3]),  axis = 0)

    # Define t0
    T = 1          # time 1 turn takes (convert n_turns to t0)
    t0 = T * data['at_turn_last']

    return t0.flatten(), J1_init, J1_avg, J2_init, J2_avg, epsg_x, epsg_y

t0, J1_init, J1_avg, J2_init, J2_avg, epsg_x, epsg_y = initialization(myfile)

##################
#    integral    #
##################

def integral(t, t0, J1, J2, epsg_x, epsg_y):
    V     = 1    # volume of hypersphere     ~ 1/2 pi^2 R^4
    M     = 1    # number of samples         ~ 50,000
    N     = 1    # total number of particles ~ 10^11
    T     = 1

    g_cst = N/(2 * np.pi)**2 / (epsg_x * epsg_y)
    g_fct = np.exp(-J1/(epsg_x)) * np.exp(-J2/(epsg_y))

    g     = g_cst * g_fct

    n     = t / T # convert time t to number of turns n
    
    Q = V/M * np.sum(np.multiply(K(t, t0), g))
    
    return Q

