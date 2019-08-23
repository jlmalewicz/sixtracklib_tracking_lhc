import numpy as np
import matplotlib.pyplot as plt
from integral import *


mydict = mfm.h5_to_dict('mydict.h5')
optics = mfm.h5_to_dict('../data/losses_sixtracklib.3724052.0.h5', group = 'beam-optics')

J1_all   = mydict['J1']
J2_all   = mydict['J2']
t0_all   = mydict['t0']
phi1_all = mydict['phi1']
phi2_all = mydict['phi2'] 
epsg_x   = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y   = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])

######################
# define sigma 1 & 2 #
######################

sigma1 = 3.5
sigma2 = 4

######################
#  lost particles ?  #
######################


# J1_lost = np.multiply((1 - K(max(t0_all) - 1, t0_all)), J1_all)[J1_lost != 0]
# J2_lost = np.multiply((1 - K(max(t0_all) - 1, t0_all)), J2_all)[J2_lost != 0]
# 
# phi1_lost = np.multiply((1-K(max(t0_all)-1, t0_all)), phi1_all)[phi1_lost != 0]
# phi2_lost = np.multiply((1-K(max(t0_all)-1, t0_all)), phi2_all)[phi2_lost != 0]
# 
# A1_lost = np.sqrt(2 * J1_lost / epsg_x)
# A2_lost = np.sqrt(2 * J2_lost / epsg_y)

######################
#   get integrals    #
######################

# from 0 to sigma1
f1 = ((1 - np.exp(-sigma1**2/2) * (1 + sigma1**2/2)) /
      (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2)))
# from sigma1 to sigma2


# separate the particles in two regions 1 & 2
mask1 = (J1_all / epsg_x + J2_all / epsg_y < sigma1**2 /2)
mask2 = (J1_all / epsg_x + J2_all / epsg_y >= sigma1**2 /2)

t0_1, J1_1, J2_1 = t0_all[mask1], J1_all[mask1], J2_all[mask1]
t0_2, J1_2, J2_2 = t0_all[mask2], J1_all[mask2], J2_all[mask2]

Q  = np.zeros([20000,])
Q2 = np.zeros([20000,])

turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)

for i,t in enumerate(turns):
    Q[i] = integral(t, t0_2, J1_2/epsg_x, J2_2/epsg_y, sigma1, sigma2)
    Q2[i] = integral2(t, t0_2, J1_2/epsg_x, J2_2/epsg_y, sigma1, sigma2)
    if i % 200 == 0:
        print(i/200, '%')

errorQ = np.sqrt((Q2 - Q**2)/t0_all.shape[0])


#############
# get plots #
#############

plt.style.use('kostas')

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
def intensity_plot():
    plt.title(r'Losses, $\delta_{init} = 9.7 \times 10^{-5}$', loc = 'right')
    plt.plot(turns, Q+f1)
    plt.xlabel('Number of turns')
    plt.ylabel('Intensity')

    errorfill(turns, Q+f1, yerr=errorQ,)
    plt.show()

    return

# Lost particles
# J1 vs. J2
def lostparticle_J_plot():
    plt.plot(J1_lost/epsg_x, J2_lost/epsg_y, 'r.', label = 'Lost particles')
    plt.legend(loc = 'upper right')
    plt.title(r'Invariants of motion for lost particles')
    plt.xlabel(r'$J_1 / \epsilon_{1}$')
    plt.ylabel(r'$J_2 / \epsilon_{2}$')
    for i in np.arange(0, 4.5, 0.5):
        x = np.linspace(0, (i**2)/2, num = 100)
        y = (i**2)/2 - x
        plt.plot(x, y, 'k')
    return

# A1 vs. A2
def lostparticle_A_plot():
    plt.xlabel(r'$A_1 [\sigma]$')
    plt.ylabel(r'$A_2 [\sigma]$')
    plt.plot(A1_lost, A2_lost, 'r.')
    for i in np.arange(0, 4.5, 0.5):
        x = np.linspace(0, i, num = 100)
        y = np.sqrt((i)**2 - x**2)
        plt.plot(x, y, 'k')
    plt.title(r'Amplitudes for lost particles')

def lostparticle_phi_plot():
    plt.plot(phi1_lost, phi2_lost, 'r.', label = 'Lost particles')
    plt.legend(loc = 'upper right')
    plt.title(r'Transverse angles for lost particles')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(r'$\phi_2$')
    return


