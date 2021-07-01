import glob
import os
import subprocess
import shutil

filenames = glob.glob('../a_mdt_data/HR_model_data/nemo_mnthly/*.dat')
filenames = [os.path.split(fname)[-1] for fname in filenames]
filenames = [fname[:len(fname)-4] for fname in filenames]
for fname in filenames:
    with open('cs_params.txt', 'w+') as f:
        f.write(fname)
        f.write('\n12')
    subprocess.run('./band10_shmdt')
    shutil.move("../a_mdt_data/computations/currents/" + fname + '_cs.dat', "../a_mdt_data/HR_model_data/nemo_currents/" + fname + '_cs.dat')