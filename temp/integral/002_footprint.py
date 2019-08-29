import NAFFlib
import numpy as np
import matplotlib.pyplot as plt
from integral import *

filelist = ['../data/losses_sixtracklib.3729580.0.h5',
            '../data/losses_sixtracklib.3731612.1.h5']

t0_all   = np.zeros([20000, 1000])
x_norm   = np.zeros([20000, 1000])
px_norm  = np.zeros([20000, 1000])
y_norm   = np.zeros([20000, 1000])
py_norm  = np.zeros([20000, 1000])


myfile = filelist[0]
data           = mfm.h5_to_dict(myfile, group = 'output')
in_data        = mfm.h5_to_dict(myfile, group = 'input')
optics         = mfm.h5_to_dict(myfile, group = 'beam-optics')
particle_on_CO = mfm.h5_to_dict(myfile, group = 'closed-orbit')

epsg_x = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])
invW = optics['invW']

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
X_norm      = normalize(X, part_on_CO, invW)
t0 = data['at_turn_last']

for i in range(1000):
    t0_all[0:10000,i]  = t0.flatten()
x_norm[0:10000,:]  = X_norm[:,:,0].T
px_norm[0:10000,:] = X_norm[:,:,1].T
y_norm[0:10000,:]  = X_norm[:,:,2].T
py_norm[0:10000,:] = X_norm[:,:,3].T

#################################

myfile = filelist[1]
data           = mfm.h5_to_dict(myfile, group = 'output')
in_data        = mfm.h5_to_dict(myfile, group = 'input')
optics         = mfm.h5_to_dict(myfile, group = 'beam-optics')
particle_on_CO = mfm.h5_to_dict(myfile, group = 'closed-orbit')

epsg_x = optics['epsn_x'] / (optics['beta0'] * optics['gamma0'])
epsg_y = optics['epsn_y'] / (optics['beta0'] * optics['gamma0'])
invW = optics['invW']

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
X_norm      = normalize(X, part_on_CO, invW)
t0 = data['at_turn_last']

for i in range(1000):
    t0_all[10000:20000, i]  = t0.flatten()
x_norm[10000:20000, :]  = X_norm[:,:,0].T
px_norm[10000:20000,:]  = X_norm[:,:,1].T
y_norm[10000:20000, :]  = X_norm[:,:,2].T
py_norm[10000:20000,:]  = X_norm[:,:,3].T

################################

mask_lost    = (K(max(t0_all[:,0])-1, t0_all[:,0])  == 0)

mask_kept    = (K(max(t0_all[:,0])-1, t0_all[:,0])  == 1)



# q1_lost = NAFFlib.multiparticle_tunes(x_norm[mask_lost] +
#                                  0.j*px_norm[mask_lost], 0)
# q2_lost = NAFFlib.multiparticle_tunes(y_norm[mask_lost] +
#                                  0.j*py_norm[mask_lost], 0)
# 
# q1_kept = NAFFlib.multiparticle_tunes(x_norm[mask_kept] +
#                                  0.j*px_norm[mask_kept], 0)
# q2_kept = NAFFlib.multiparticle_tunes(y_norm[mask_kept] +
#                                  0.j*py_norm[mask_kept], 0)


q1_lost = np.array([NAFFlib.multiparticle_tunes(x_norm[mask_lost][:,:200],    0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_lost][:,200:400], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_lost][:,400:600], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_lost][:,600:800], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_lost][:,800:1000],0)])
                     
q2_lost = np.array([NAFFlib.multiparticle_tunes(y_norm[mask_lost][:,:200],    0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_lost][:,200:400], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_lost][:,400:600], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_lost][:,600:800], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_lost][:,800:1000],0)])

q1_kept = np.array([NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,:200],    0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,200:400], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,400:600], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,600:800], 0),
                    NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,800:1000],0)])

q2_kept = np.array([NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,:200],    0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,200:400], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,400:600], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,600:800], 0),
                    NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,800:1000],0)])





                    
q1_kept = NAFFlib.multiparticle_tunes(x_norm[mask_kept][:,:200], 0)
q2_kept = NAFFlib.multiparticle_tunes(y_norm[mask_kept][:,:200], 0)

plt.plot(q1_lost[i,:], q2_lost[i,:], 'r.', label = 'lost')
plt.plot(q1_kept[i,:], q2_kept[i,:], '.',  label = 'kept')
plt.title('Footprint')
plt.legend(loc = 'upper right')
plt.show()

