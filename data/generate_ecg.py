import os 
import fk
from fk import electrogram

root = 'data/fk_sims/train_dev_set/set5/'
filenames = os.listdir(root)

for f in filenames:
	electrogram.calc_egm(root+f, electrogram.hd_grid_probes(6.0), dt=1, dx=0.01, real_size = 12, scaled_size = 1200)