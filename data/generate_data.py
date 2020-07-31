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

# S1 Stimulus - planar from middle horizontal
protocol1 = stimulus.protocol(start=0, duration=50, period=20000, current = True)
s1_m = stimulus.rectangular(shape, (shape[1]-stripe_size/2-int(shape[0] / 2), shape[0]/2), (stripe_size, shape[0]), 1.0, protocol1)

# S2 stimulus - planar from right
protocol2 = stimulus.protocol(start=10000, duration=50, period=0, current = True)
s2 = stimulus.rectangular(shape, (shape[0]/2,shape[1]-stripe_size/2), (shape[0], stripe_size), 1.0, protocol2)

# S2 stimulus - planar from middle vertical
protocol2 = stimulus.protocol(start=15000, duration=50, period=0, current = True)
s2_m = stimulus.rectangular(shape, (shape[0]/2,shape[1]-stripe_size/2-int(shape[0] / 2)), (shape[0], stripe_size), 1.0, protocol2)

# S3 - point stimulus
protocol3 = stimulus.protocol(start=0, duration=50, period=60000, current = True)
s3 = stimulus.rectangular(shape, (1*shape[1]/3, 1*shape[0]/3), (dot_size, dot_size), 1.0, protocol3)

# S4 - point stimulus
protocol4 = stimulus.protocol(start=15000, duration=50, period=0, current = True)
s4 = stimulus.rectangular(shape, (1*shape[1]/3, 1.8*shape[0]/3), (dot_size, dot_size), 1.0, protocol4)
    
# S5 - point stimulus
protocol5 = stimulus.protocol(start=20000, duration=50, period=15000, current = True)
s5 = stimulus.rectangular(shape, (2*shape[1]/3, 2*shape[0]/3), (dot_size, dot_size), 1.0, protocol5)

all_stimuli = [ [s1], [s1_m], [s1, s2], [s1_m, s2], [s1_m, s2_m], [s1, s3], [s3,s4, s5], [s1, s2_m, s3, s4], [s_2m, s3, s4], [s3] ]

all_stimuli_filenames = [str(s) for s in range(len(all_stimuli))]
root = '/rds/general/user/kn119/home/data/fk_sims/train_dev_set/set1a/'
filenames = [root + s + "_resized_256.hdf5" for s in all_stimuli_filenames]



start = 0  # ms
stop = 2000  # ms

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








