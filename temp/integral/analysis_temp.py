import numpy as np
import matplotlib.pyplot as plt
from integral import *

optics = mfm.h5_to_dict('../data/losses_sixtracklib.3724052.0.h5', group = 'beam-optics')

mydict = mfm.h5_to_dict('losses_sixtracklib.h5')
J1_all = mydict['J1']
J2_all = mydict['J2']
t0_all = mydict['t0']
epsg_x = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])

import time
start = time.time()

######################################################################

def integral(t, t0, J1, J2, epsg_x, epsg_y):
    R     = 4        # radius of hypersphere
    V     = 1/2 * np.pi**2 * R**4 /16  # volume of hypersphere    
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1        # total number of particles  ~ 10^11
    T     = 1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2))
    g_fct = np.exp(-J1/(epsg_x)) * np.exp(-J2/(epsg_y))

    g     = g_fct

    n     = t / T # convert time t to number of turns n

    Q = V/M * g_cst  * np.sum(np.multiply(K(t, t0), g))

    return Q

def integral2(t, t0, J1, J2, epsg_x, epsg_y):
    R     = 4        # radius of hypersphere
    V     = 1/2 * np.pi**2 * R**4 /16  # volume of hypersphere
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1        # total number of particles  ~ 10^11
    T     = 1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2))
    g_fct = np.exp(-2 * J1/(epsg_x)) * np.exp(-2 * J2/(epsg_y))

    g2     = g_fct

    n     = t / T # convert time t to number of turns n

    Q2 = V**2/M * g_cst**2 * np.sum(np.multiply(K(t, t0), g2))

    return Q2


######################################################################

sigma1 = 3
sigma2 = 4

# from 0 to sigma1
f1 = ((1 - np.exp(-sigma1**2/2) * (1 + sigma1**2/2)) /
      (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2)))
# from sigma1 to sigma2


epsg_x = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])


# separate the particles in two regions 1 & 2
mask1 = (J1_all / epsg_x + J2_all / epsg_y < sigma1**2 /2)
mask2 = (J1_all / epsg_x + J2_all / epsg_y >= sigma1**2 /2)

t0_1, J1_1, J2_1 = t0_all[mask1], J1_all[mask1], J2_all[mask1]
t0_2, J1_2, J2_2 = t0_all[mask2], J1_all[mask2], J2_all[mask2]



Q  = np.zeros([20000,])
Q2 = np.zeros([20000,])

for i in range(0,20000):
    Q[i] = integral((1000*i), t0_2, J1_2, J2_2, epsg_x, epsg_y)
    Q2[i] = integral2((1000*i), t0_2, J1_2, J2_2, epsg_x, epsg_y)
    if i % 200 == 0:
        print(i/200, '%')

errorQ = np.sqrt((Q2 - Q**2)/100000)

end = time.time()
print(end - start)
