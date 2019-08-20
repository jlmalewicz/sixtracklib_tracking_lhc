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

turns = np.array([range(0, 20000000, 1000)]).reshape(20000,)
QQ = Q / Q[0] * 100

mydict = {'t0'    : t0_all,
          'J1'    : J1_all,
          'J2'    : J2_all,
          'Q'     : Q,
          'turns' : turns
         }

mfm.dict_to_h5(mydict, 'losses_sixtracklib.h5')



#############
# get plots #
#############

plt.style.use('kostas')

# plt.plot(J1_all[::100], J2_all[::100], '.', label = r'$J_{init}$')
# plt.legend(loc = 'upper right')
# plt.title(r'Invariants of motion $J_1$ vs. $J_2$')

plt.title(r'Losses, $\delta_{init} = 9.7 \times 10^{-5}$', loc = 'right')
plt.plot(turns, QQ)
plt.xlabel('Number of turns')
plt.ylabel('Intensity [$\%$ of initial]')

plt.savefig('losses_plot.png')
plt.show()
