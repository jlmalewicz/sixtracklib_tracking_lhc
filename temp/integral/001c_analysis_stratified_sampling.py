import numpy as np
import matplotlib.pyplot as plt
from integral import *


input_file = 'losses_sixtracklib_20000_27.h5'
# input_file = 'losses_sixtracklib_200000_097.h5'

######################
#   initialization   #
######################

mydict = mfm.h5_to_dict(input_file, group = 'input')
optics = mfm.h5_to_dict(input_file, group = 'optics')

J1_all   = mydict['J1']
J2_all   = mydict['J2']
t0_all   = mydict['t0']
phi1_all = mydict['phi1']
phi2_all = mydict['phi2']
epsg_x   = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y   = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])

sigma1 = 2.5
sigma2 = 4

###########################
#   stratified sampling   #
###########################

N_strat = 1# number of slices
strat   = np.linspace(sigma1, sigma2, N_strat+1)
low_lim = strat[:-1]
up_lim  = strat[1:]

J_norm  = J1_all/epsg_x + J2_all/epsg_y

masks   = []
for i in range(N_strat):
    mask = np.logical_and(low_lim[i] ** 2 / 2 < J_norm, 
                           up_lim[i] ** 2 / 2 >= J_norm)
    masks.append(mask)


# from 0 to sigma1
f1 = ((1 - np.exp(-sigma1**2/2) * (1 + sigma1**2/2)) /
      (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2)))

# from sigma1 to sigma2
Q      = np.zeros([20000,])
Q2     = np.zeros([20000,])
Q_tot  = np.zeros([20000,])
errorQ = np.zeros([20000,])

turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)

for ii in range(N_strat):
    print('* * * slice %s out of %s * * *'%(ii+1, N_strat))
    mask = masks[ii]
    t0, J1, J2 = t0_all[mask], J1_all[mask], J2_all[mask]
    for i,t in enumerate(turns):
        Q[i] = integral(t, t0, J1/epsg_x, J2/epsg_y,low_lim[ii], up_lim[ii])
        Q2[i] = integral2(t, t0, J1/epsg_x, J2/epsg_y, low_lim[ii], up_lim[ii])
        Q_tot[i] += Q[i]
        if i % 2000 == 0:
            print(i/200, '%')
    errorQ += (Q2 - Q**2)/t0.shape[0]
errorQ = np.sqrt(errorQ)

##################
# intensity plot #
##################

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = 'blue'
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


# Intensity plot
def intensity_plot(c):
    if c is None:
        c = 'blue'

    plt.title(r'Losses, stratified sampling, $\delta_{init}$ = %s'%(mydict['init_delta']), loc = 'right')
    plt.plot(turns, Q_tot+f1, c, label = '%s slices'%N_strat)
    plt.xlabel('Number of turns')
    plt.ylabel('Intensity')

#    errorfill(turns, Q_tot+f1, yerr=errorQ, color = c)

    plt.legend(loc = 'lower left')

    plt.show()

