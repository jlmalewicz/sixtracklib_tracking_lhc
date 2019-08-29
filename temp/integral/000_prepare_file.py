import numpy as np
import matplotlib.pyplot as plt
from integral import *

# input
# delta init = 0.000097

#filelist = ['../data/losses_sixtracklib.3724052.0.h5',
#            '../data/losses_sixtracklib.3724052.1.h5',
#            '../data/losses_sixtracklib.3724052.2.h5',
#            '../data/losses_sixtracklib.3724052.3.h5',
#            '../data/losses_sixtracklib.3724052.4.h5',
#
#            '../data/losses_sixtracklib.3724053.0.h5',
#            '../data/losses_sixtracklib.3724053.1.h5',
#            '../data/losses_sixtracklib.3724053.2.h5',
#            '../data/losses_sixtracklib.3724053.3.h5',
#            '../data/losses_sixtracklib.3724053.4.h5',
# 
#            '../data/losses_sixtracklib.3724054.0.h5',
#            '../data/losses_sixtracklib.3724054.1.h5',
#            '../data/losses_sixtracklib.3724054.2.h5',
#            '../data/losses_sixtracklib.3724054.3.h5',
#            '../data/losses_sixtracklib.3724054.4.h5',
# 
#            '../data/losses_sixtracklib.3724055.0.h5',
#            '../data/losses_sixtracklib.3724055.1.h5',
#            '../data/losses_sixtracklib.3724055.2.h5',
#            '../data/losses_sixtracklib.3724055.3.h5',
#            '../data/losses_sixtracklib.3724055.4.h5']

# delta init = 0.00027
filelist = ['../data/losses_sixtracklib.3729580.0.h5',
            '../data/losses_sixtracklib.3731612.1.h5']

optics  = mfm.h5_to_dict(filelist[0], group = 'beam-optics')
in_file = merge(filelist)
init_delta = (str(mfm.h5_to_dict(filelist[0], group = 'input')['init_delta_wrt_CO']*10000).replace('.','')) 
in_file['init_delta'] = mfm.h5_to_dict(filelist[0], group = 'input')['init_delta_wrt_CO']

# save output (nr of particles, delta_init)
output_fname = 'losses_sixtracklib_%s_%s.h5'%(in_file['t0'].shape[0], 
                                              init_delta)

mfm.dict_to_h5(in_file, output_fname, group='input',  readwrite_opts='w')
mfm.dict_to_h5(optics,  output_fname, group='optics', readwrite_opts='a')


