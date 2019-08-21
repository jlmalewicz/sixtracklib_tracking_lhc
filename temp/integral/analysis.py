import numpy as np
import matplotlib.pyplot as plt
from integral import *

#################
#     MERGE     #
#################

filelist = ['../data/losses_sixtracklib.3724052.0.h5',
            '../data/losses_sixtracklib.3724052.1.h5',
            '../data/losses_sixtracklib.3724052.2.h5',
            '../data/losses_sixtracklib.3724052.3.h5',
            '../data/losses_sixtracklib.3724052.4.h5',

            '../data/losses_sixtracklib.3724053.0.h5',
            '../data/losses_sixtracklib.3724053.1.h5',
            '../data/losses_sixtracklib.3724053.2.h5',
            '../data/losses_sixtracklib.3724053.3.h5',
            '../data/losses_sixtracklib.3724053.4.h5']


t0_all = np.array([])
J1_all = np.array([])
J2_all = np.array([])

for myfile in filelist:
    print(' ')
    print(myfile)
    t0, J1_init, a, J2_init, b, epsg_x, epsg_y = initialization(myfile)
    t0_all = np.append(t0_all, t0)
    J1_all = np.append(J1_all, J1_init)
    J2_all = np.append(J2_all, J2_init)

############
# Integral #
############

print(' ')
print(' - - - - - - - - - - - - - - - - - - - - - ')
print('Now calculating the intensity integral...')

Q = np.zeros([20000,])
for i in range(0,20000):
    Q[i] = integral((1000*i), t0_all, J1_all, J2_all, epsg_x, epsg_y)
    if i % 200 == 0:
        print(i/200, '%')
Q2 = np.zeros([20000,])
for i in range(0,20000):
    Q2[i] = integral2((1000*i), t0_all, J1_all, J2_all, epsg_x, epsg_y)
    if i % 200 == 0:
        print(i/200, '%')

errorQ = np.sqrt((Q2 - Q**2)/100000)

turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)


mydict = {'t0'    : t0_all,
          'J1'    : J1_all,
          'J2'    : J2_all,
          'Q'     : Q,
          'Q2'    : Q2
          'varQ'  : varQ
          'turns' : turns
         }

mfm.dict_to_h5(mydict, 'losses_sixtracklib.h5')

#############
# LOST PART #
#############

J1_lost = np.multiply((1 - K(max(t0_all) - 1, t0_all)), J1_all)
J2_lost = np.multiply((1 - K(max(t0_all) - 1, t0_all)), J2_all)

J1_lost = J1_lost[J1_lost != 0]
J2_lost = J2_lost[J2_lost != 0]


A1 = np.sqrt(2 * J1_lost / epsg_x)
A2 = np.sqrt(2 * J2_lost / epsg_y)


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

plt.title(r'Losses, $\delta_{init} = 9.7 \times 10^{-5}$', loc = 'right')
plt.plot(turns, Q)
plt.xlabel('Number of turns')
plt.ylabel('Intensity')

plt.savefig('losses_plot.png')
plt.show()

errorfill(turns, Q, yerr=errorQ,)


#### Lost particles #####
# J1 vs. J2
plt.plot(J1_lost/epsg_x, J2_lost/epsg_y, 'r.', label = 'Lost particles') 
plt.legend(loc = 'upper right')
plt.title(r'Invariants of motion for lost particles')
plt.xlabel(r'$J_1 / \epsilon_{1}$')
plt.ylabel(r'$J_2 / \epsilon_{2}$')
for i in np.arange(0, 4.5, 0.5):
    x = np.linspace(0, (i**2)/2, num = 100)
    y = (i**2)/2 - x
    plt.plot(x, y, 'k')

# A1 vs. A2
plt.xlabel(r'$A_1 [\sigma]$')
plt.ylabel(r'$A_2 [\sigma]$')
plt.plot(A1, A2, 'r.')
for i in np.arange(0, 4.5, 0.5):
    x = np.linspace(0, i, num = 100)
    y = np.sqrt((i)**2 - x**2)
    plt.plot(x, y, 'k')
plt.title(r'Amplitudes for lost particles')

#############
# save data #
#############

mydict = {'t0'    : t0_all,
          'J1'    : J1_all,
          'J2'    : J2_all,
          'Q'     : Q,
          'Q2'    : Q2
          'varQ'  : varQ
          'turns' : turns
         }

mfm.dict_to_h5(mydict, 'losses_sixtracklib.h5')

