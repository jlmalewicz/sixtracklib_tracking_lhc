import numpy as np
import random

sigma = 4
N     = 0.5
dim   = 2

def g_n(x_n, px_n, y_n, py_n):
    J1_n = J(x_n, px_n) / epsn_x
    J2_n = J(y_n. py_n) / epsn_y
    g_n = 1 / max_g * np.exp(-J1_n-J2_n)
    return g_n


# since only J is relevant for G, can I get away with only 
# generating J1_n and J2_n in \omega?

def accept_reject(sigma, nr_particles, dim, seed = 0):
    np.random.seed(seed)
    np.random.uniform((1,10000))
    i = 0
    sphere = np.zeros((nr_particles, dim))
    while i < nr_particles:
        point = np.random.uniform(low=0, high=sigma**2/2, size=(dim,))
        N_n   = random.random() 
        if np.linalg.norm(point) <= sigma**2/2 and g_n(point) >= N_n:
            sphere[i] = point
            Q[i]      = g_n(point)
            i += 1
    return sphere, Q


