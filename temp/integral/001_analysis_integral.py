import numpy as np
import matplotlib.pyplot as plt
from integral import *


input_file = 'losses_sixtracklib_20000_27.h5'

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

######################
#   get integrals    #
######################

# separate the particles in two regions 1 & 2
mask_1 = (J1_all / epsg_x + J2_all / epsg_y <  sigma1**2 /2) #       0 < (1) < sigma_1
mask_2 = (J1_all / epsg_x + J2_all / epsg_y >= sigma1**2 /2) # sigma_1 < (2) < sigma_2

t0_1, J1_1, J2_1 = t0_all[mask_1], J1_all[mask_1], J2_all[mask_1]
t0_2, J1_2, J2_2 = t0_all[mask_2], J1_all[mask_2], J2_all[mask_2]

# from 0 to sigma1
f1 = ((1 - np.exp(-sigma1**2/2) * (1 + sigma1**2/2)) /
      (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2)))
# from sigma1 to sigma2
Q  = np.zeros([20000,])
Q2 = np.zeros([20000,])

turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)

# calculate the integral for (2)
for i,t in enumerate(turns):
    Q[i] = integral(t, t0_2, J1_2/epsg_x, J2_2/epsg_y, sigma1, sigma2)
    Q2[i] = integral2(t, t0_2, J1_2/epsg_x, J2_2/epsg_y, sigma1, sigma2)
    if i % 1000 == 0:
        print(i/200, '%')

errorQ = np.sqrt((Q2 - Q**2)/t0_all.shape[0])

######################
#  lost particles ?  #
######################

mask_lost    = (K(max(t0_all)-1, t0_all)  == 0)

J1_lost,   J2_lost   = J1_all[mask_lost],   J2_all[mask_lost]
phi1_lost, phi2_lost = phi1_all[mask_lost], phi2_all[mask_lost]

A1_lost = np.sqrt(2 * J1_lost / epsg_x)
A2_lost = np.sqrt(2 * J2_lost / epsg_y)

################
#   save dict  #
################

analysis = {'sigma1'  :  sigma1,
            'sigma2'  :  sigma2,
            'n_part_in_integral' : t0_2.shape[0],
            'n_part_lost'        : J1_lost.shape[0]
            }

mfm.dict_to_h5(analysis, input_file, group='analysis', readwrite_opts='a')

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
    plt.title(r'Losses, $\delta_{init}$ = %s'%(mydict['init_delta']), loc = 'right')
    plt.plot(turns, Q+f1)
    plt.xlabel('Number of turns')
    plt.ylabel('Intensity')

    errorfill(turns, Q+f1, yerr=errorQ)

    plt.plot([], [], 'b.', label = '%s particles in integration'%(t0_2.shape[0]))
    plt.legend(loc = 'lower left')
    
    plt.show()

# Lost particles
# J1 vs. J2

def lostparticles_heat_plot():
    from scipy.stats import kde
    data = np.array([A1_lost, A2_lost]).T
    x, y = data.T
    k = kde.gaussian_kde(data.T)
    nbins = 300
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape), cmap= cm.magma)
    plt.xlabel(r'$A_1 [\sigma]$')
    plt.ylabel(r'$A_2 [\sigma]$')
    plt.title(r'Spatial distribution of Losses for $\delta_{init}$ = %s'%(mydict['init_delta']))

def lostparticle_J_plot():
    plt.plot(J1_lost/epsg_x, J2_lost/epsg_y, 'r.', label = '%s lost particles'%(J1_lost.shape[0]))
    plt.legend(loc = 'upper right')
    plt.title(r'Invariants of motion for lost particles')
    plt.xlabel(r'$J_1 / \epsilon_{1}$')
    plt.ylabel(r'$J_2 / \epsilon_{2}$')
    for i in np.arange(0, 4.5, 0.5):
        x = np.linspace(0, (i**2)/2, num = 100)
        y = (i**2)/2 - x
        plt.plot(x, y, 'k')

# A1 vs. A2
def lostparticle_A_plot():
    plt.xlabel(r'$A_1 [\sigma]$')
    plt.ylabel(r'$A_2 [\sigma]$')
    plt.plot(A1_lost, A2_lost, 'r.', label = '%s lost particles'%(J1_lost.shape[0]))
    plt.legend(loc = 'upper right')
    for i in np.arange(0, 4.5, 0.5):
        x = np.linspace(0, i, num = 100)
        y = np.sqrt((i)**2 - x**2)
        plt.plot(x, y, 'k')
    plt.title(r'Amplitudes for lost particles')

# phi1 vs. phi2
def lostparticle_phi_plot():
    plt.plot(phi1_lost, phi2_lost, 'r.', label = '%s lost particles'%(J1_lost.shape[0]))
    plt.legend(loc = 'upper right')
    plt.title(r'Transverse angles for lost particles')
    plt.xlabel(r'$\phi_1$')
    plt.ylabel(r'$\phi_2$')


plt.figure(1)
lostparticle_J_plot()

plt.figure(2)
lostparticle_A_plot()

plt.figure(3)
lostparticle_phi_plot()
