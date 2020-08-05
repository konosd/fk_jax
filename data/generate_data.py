import os
import random
import h5py
import math
import matplotlib.pyplot as plt
import jax
import jax.numpy as np
import fk

import os
from fk import params
from fk import model
from fk import convert
from fk import stimulus
from fk import plot
from fk import diffusivity
import matplotlib.pyplot as plt
import jax
import jax.numpy as np

import fk
import time

import sys
import numpy as onp


# #########################################################
# simulation inputs (real values)
# root = "data/"
field_size = (12, 12)  # cm
d = 0.001  # (cm^2/ms)
cell_parameters = fk.params.params5()

# infinitesimals
dx = 0.01  # (cm/units) - Fenton 1998 recommends ~200, 300 micron/gridunit (~0.02, 0.03), smaller dx means finer grid
dt = 0.01  # (ms) - Fenton 1998 recommends few hundreds of ms (~0.01, 0.04)

# diffusivity 
d = 0.001  # cm^2/ms
shape = fk.convert.realsize_to_shape(field_size, dx)

diffusivity_0 = np.ones(shape) * d

diffusivity_field_1 = diffusivity.rectangular(shape, (2*shape[1]/3, 2*shape[0]/3), (int(shape[0]/3), int(shape[0]/3)), d, 0.7)

diffusivity_field_2 = diffusivity.rectangular(shape, (2*shape[1]/3, 1*shape[0]/3), (int(shape[0]/3), int(shape[0]/3)), d, 0.3)
diffusivity_field_3 = diffusivity.rectangular(shape, (1*shape[1]/3, 1*shape[0]/3), (int(shape[0]/3), int(shape[0]/3)), d, 0.5)

diffusivity_fields = [diffusivity_0, diffusivity_1, diffusivity_2, diffusivity_3]

# #########################################################
# times
start = 0  # ms
stop = 2000  # ms
# #########################################################

stripe_size = int(shape[0] / 100)
dot_size = int(shape[0]/20)


# ################################## Simple Simulations #############################################


# S1 Stimulus - planar from bottom
protocol1 = stimulus.protocol(start=0, duration=50, period=50000, current = True)
s1 = stimulus.rectangular(shape, (shape[1]-stripe_size/2, shape[0]/2), (stripe_size, shape[0]), 1.0, protocol1)

# # S1 Stimulus - planar from middle horizontal
# protocol1 = stimulus.protocol(start=0, duration=50, period=20000, current = True)
# s1_m = stimulus.rectangular(shape, (shape[1]-stripe_size/2-int(shape[0] / 2), shape[0]/2), (stripe_size, shape[0]), 1.0, protocol1)

# S2 stimulus - planar from right
protocol2 = stimulus.protocol(start=40000, duration=50, period=0, current = True)
s2 = stimulus.rectangular(shape, (shape[0]/2,shape[1]-stripe_size/2), (shape[0], stripe_size), 1.0, protocol2)

# S3 - point stimulus
protocol3 = stimulus.protocol(start=0, duration=50, period=60000, current = True)
s3 = stimulus.rectangular(shape, (1*shape[1]/3, 1*shape[0]/3), (dot_size, dot_size), 1.0, protocol3)


# ################################## Complex Simulations #############################################

# The heartbeat
protocol4 = stimulus.protocol(start=0, duration=50, period=14000, current = True)
s4 = stimulus.triangular(shape, 'left', 30, 0.2, 1.0, protocol4)

protocol5 = stimulus.protocol(start=35000, duration=50, period=20000, current = True)
s5 = stimulus.rectangular(shape, (1*shape[1]/3, 1*shape[0]/3), (dot_size, dot_size), 1., protocol5)


# The cross-sectional case
protocol6 = stimulus.protocol(start=0, duration=50, period=0, current = True)
s6 = stimulus.triangular(shape, 'left', 30, 0.3, 1.0, protocol6)

protocol7 = stimulus.protocol(start=40000, duration=50, period=0, current = True)
s7 = stimulus.triangular(shape, 'down', 30, 0.1, 1.0, protocol7)


# The point stimulus case
protocol8 = stimulus.protocol(start=0, duration=50, period=0, current = True)
s8 = stimulus.rectangular(shape, (1*shape[1]/6, 1*shape[0]/6), (dot_size, dot_size), 1., protocol8)

protocol9 = stimulus.protocol(start=46500, duration=50, period=0, current = True)
s9 = stimulus.rectangular(shape, (5*shape[1]/12, 5*shape[0]/12), (dot_size, dot_size), 1., protocol9)

protocol10 = stimulus.protocol(start=51000, duration=50, period=0, current = True)
s10 = stimulus.rectangular(shape, (7.5*shape[1]/12, 2.5*shape[0]/12), (dot_size, dot_size), 1., protocol10)




all_stimuli = [ [s1], [s3], [s1, s2], [s4, s5], [s6, s7], [s8,s9,s10] ]

diff_names = ['dif0','dif1','dif2','dif3']
all_stimuli_filenames = [str(s) for s in range(len(all_stimuli))]
root = '/rds/general/user/kn119/home/data/fk_sims/train_dev_set/set5/'
filenames = [root + 'set5_' + s + "_" + diff_name + ".hdf5" for s in all_stimuli_filenames for diff_name in diff_names]



start = 0  # ms
stop = 2000  # ms




for jd, diff in enumerate(diffusivity_fields):
	for i, stimuli in enumerate(all_stimuli):
	    t0= time.clock()
	    print("Starting set "+ all_stimuli_filenames[i])

	    fk.data.generate(start=fk.convert.ms_to_units(start, dt),
	                    stop=fk.convert.ms_to_units(stop, dt),
	                    dt=dt, dx=dx,
	                    cell_parameters=cell_parameters,
	                    diffusivity=diff,
	                    stimuli=stimuli,
	                    filename= root + 'set5_' + i + "_" + diff_names[jd] + ".hdf5",#    filenames[i],
	                    reshape=None)
	    t1 = time.clock() - t0
	    print("Time elapsed:{} min".format(t1/60)) # CPU seconds elapsed (floating point)








