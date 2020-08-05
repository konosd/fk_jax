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
cell_parameters = fk.params.params1a()

# infinitesimals
dx = 0.01  # (cm/units) - Fenton 1998 recommends ~200, 300 micron/gridunit (~0.02, 0.03), smaller dx means finer grid
dt = 0.01  # (ms) - Fenton 1998 recommends few hundreds of ms (~0.01, 0.04)

# diffusivity 
d = 0.001  # cm^2/ms
shape = fk.convert.realsize_to_shape(field_size, dx)
diffusivity = np.ones(shape) * d

# #########################################################
# times
start = 0  # ms
stop = 2000  # ms
# #########################################################

stripe_size = int(shape[0] / 100)
dot_size = int(shape[0]/50)


# S1 Stimulus - planar from bottom
protocol1 = stimulus.protocol(start=0, duration=50, period=50000, current = True)
s1 = stimulus.rectangular(shape, (shape[1]-stripe_size/2, shape[0]/2), (stripe_size, shape[0]), 1.0, protocol1)


# S2 stimulus - planar from right
protocol2 = stimulus.protocol(start=10000, duration=50, period=0, current = True)
s2 = stimulus.rectangular(shape, (shape[0]/2,shape[1]-stripe_size/2), (shape[0], stripe_size), 1.0, protocol2)


all_stimuli = [ [s1], [s1, s2]]
all_stimuli_filenames = ['set_1a_'+str(s) for s in range(len(all_stimuli))]
root = '/rds/general/user/kn119/home/data/fk_sims/train_dev_set/set1a/'
filenames = [root + s + "_debug.hdf5" for s in all_stimuli_filenames]


start = 0  # ms
stop = 2  # ms

for i, stimuli in enumerate(all_stimuli):
    t0= time.clock()
    print("Starting set "+ all_stimuli_filenames[i])

    fk.data.generate(start=fk.convert.ms_to_units(start, dt),
                    stop=fk.convert.ms_to_units(stop, dt),
                    dt=dt, dx=dx,
                    cell_parameters=cell_parameters,
                    diffusivity=np.ones(shape) * d,
                    stimuli=stimuli,
                    filename=filenames[i],
                    reshape=(256,256))
    t1 = time.clock() - t0
    print("Time elapsed:{} min".format(t1/60)) # CPU seconds elapsed (floating point)


