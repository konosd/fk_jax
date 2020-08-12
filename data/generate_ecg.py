import os 
import fk
from fk import electrogram

# root = 'data/fk_sims/train_dev_set/set5/'
root = '/rds/general/user/kn119/ephemeral/data/fk_sims/train_dev_set/set5/'
filenames = os.listdir(root)
filenames = [f for f in filenames if '128.hdf5' not in f ]
filenames = [f for f in fileanmes if 'hdf5' in f]

for f in filenames:
	electrogram.steady_probes(root+f, dt=1, dx=0.01, real_size = 12, scaled_size = 1200)