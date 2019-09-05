import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../..')

import myfilemanager_sixtracklib as mfm



n_turns  = 2e7
n_points = 20

t_i      = np.linspace(0, n_turns, n_points + 1) 

input_file = 'losses_sixtracklib_20000_27.h5' 

myinput    = mfm.h5_to_dict(input_file, group = 'input')
optics     = mfm.h5_to_dict(input_file, group = 'optics')
analysis   = mfm.h5_to_dict(input_file, group = 'analysis')

e1              = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
e2              = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])
sigma1, sigma2  = analysis['sigma1'], analysis['sigma2']
t0, J1, J2      = myinput['t0'] + 1, myinput['J1']/e1, myinput['J2']/e2


def K(t0, t1, t2):
    return 1 * np.logical_and(t0 > t1, t0 < t2)
    
def losses(t0, J1, J2, sigma1, sigma2):
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1              # total number of particles  ~ 10^11
    T     = 1
    V_1   = 1/2 * np.pi**2 * sigma1**4 /16
    V_2   = 1/2 * np.pi**2 * sigma2**4 /16  # volume of hypersphere    
    V     = V_2 - V_1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-4**2/2) * (1 + 4**2/2))
    g_fct  = np.exp(-J1-J2)

    losses = np.zeros(n_points,)
    
    for i in range(n_points):
        losses[i] = np.sum(np.multiply(K(t0, t_i[i], t_i[i+1]), g_fct))
    
    return V/M * g_cst * losses

def losses_squared(t0, J1, J2, sigma1, sigma2):
    M     = t0.shape[0]    # number of samples          ~ 50,000
    N     = 1              # total number of particles  ~ 10^11
    T     = 1
    V_1   = 1/2 * np.pi**2 * sigma1**4 /16
    V_2   = 1/2 * np.pi**2 * sigma2**4 /16  # volume of hypersphere    
    V     = V_2 - V_1

    g_cst = N * 4 / np.pi**2 / (1 - np.exp(-sigma2**2/2) * (1 + sigma2**2/2))
    g_fct = np.exp(-2 * (J1 + J2))

    losses_squared = np.zeros(n_points,)

    for i in range(n_points):
        losses_squared[i] = np.sum(np.multiply(K(t0, t_i[i], t_i[i+1]), g_fct))

    return V**2/M * g_cst**2 * losses_squared

def variance_losses(t0, J1, J2, sigma1, sigma2):
    losses_squared_0 = losses_squared(t0, J1, J2, sigma1, sigma2)
    losses_0         = losses(t0, J1, J2, sigma1, sigma2)
    variance_losses  = (losses_squared_0 - np.multiply(losses_0, losses_0)) / t0.shape[0]
    return variance_losses




def intensity(i, low, high, t0, J1, J2): # with i in [0, n_points] 
    losses_total    = losses(t0, J1, J2, low, high)
    losses_at_t     = np.sum(losses_total[:i])
    intensity_at_t  = 1 - losses_at_t
    return intensity_at_t

def variance_intensity(i, low, high, t0, J1, J2):
    variance_losses_total    = variance_losses(t0, J1, J2, low, high)
    variance_intensity_at_t  = np.sum(variance_losses_total[:i])
    return variance_intensity_at_t

###########################
#   stratified sampling   #
###########################

n_strat = 5 # number of slices
strat   = np.linspace(sigma1, sigma2, n_strat+1)
low_lim = strat[:-1]
up_lim  = strat[1:]

masks   = []
for i in range(n_strat):
    mask = np.logical_and(low_lim[i] ** 2 / 2 <  J1 + J2,
                           up_lim[i] ** 2 / 2 >= J1 + J2)
    masks.append(mask)

I        = np.zeros(n_points + 1,)
I_tot    = np.zeros(n_points + 1,)
error_I  = np.zeros(n_points + 1,)

######################
# Intensity integral #
######################

for ii in range(n_strat):
    print('Slice %s out of %s'%(ii+1, n_strat))
    mask       = masks[ii]
    t0_ii, J1_ii, J2_ii = t0[mask], J1[mask], J2[mask]

    for i in range(n_points + 1):
        I[i]        = intensity(i, low_lim[ii], up_lim[ii], t0_ii, J1_ii, J2_ii)
        error_I[i]  = variance_intensity(i, low_lim[ii], up_lim[ii], t0_ii, J1_ii, J2_ii)
        I_tot[i]   += I[i] - 1

    error_I[i] += error_I[i] / t0_ii.shape[0]
error_I = np.sqrt(error_I)
I_tot += 1        

# WITHOUT STRATIFICATION 1.0 nor 2.0
# for i in range(n_points):
#     intensity_tot[i]       = intensity(i, sigma1, sigma2)
#     error_intensity_tot[i] = np.sqrt(variance_intensity(i, sigma1, sigma2))

t_c = (t_i[1:] + t_i[:-1]) / 2

#############
### plots ###
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

def intensity_plot():
    errorfill(t_i, I_tot, yerr = error_I, color = 'lightgray')
    plt.plot(t_i, I_tot, '.-')
    plt.title(r'Losses,  $\delta_{init}$ = %s'%(myinput['init_delta']), loc = 'right')
    plt.xlabel('Number of turns')
    plt.ylabel(r'Intensity $I(t) = 1 - \sum_{i=0} L( i \Delta t, (i+1) \Delta t)$')
    
    plt.show()

def losses_plot():
    plt.errorbar(t_c, losses(t0, J1, J2, sigma1, sigma2), 
                 yerr = np.sqrt(variance_losses(t0, J1, J2, sigma1, sigma2)),
                 fmt = 'rs', ecolor = 'lightgray', capsize = 1 )
    plt.title(r'Losses,  $\delta_{init}$ = %s'%(myinput['init_delta']), loc = 'right')
    plt.xlabel('Number of turns')
    plt.ylabel(r'Integrated Losses $L(t)$ over $10^6$ turns each')
    
    plt.show()


plt.figure(1)
intensity_plot()

plt.figure(2)
losses_plot()

################
#   save dict  #
################

intensity = {'Intensity'            : I_tot,
             'Intensity errors'     : error_I,
             'Turn interval limits' : t_i,
             'Losses'               : losses(t0, J1, J2, sigma1, sigma2),
             'Losses errors'        : np.sqrt(variance_losses(t0, J1, J2, sigma1, sigma2))
             }

# mfm.dict_to_h5(intensity, input_file, group='intensity', readwrite_opts='a')


