import numpy as np
import os







cwd = os.path.dirname(os.path.realpath(__file__))

beta = 10

ppc = 200
#noises = [1, 1.1, 1.2, 1.3, 1.4, 1.5]



prfix = "/raw_results/abl_extra/"

folders = ['k-means++_drop0_wagg0/', 'k-means++_drop0_wagg1/', 'k-means++_drop1_wagg0/', 'k-means++_drop1_wagg1/', 'random_drop0_wagg0/', 'random_drop0_wagg1/', 'random_drop1_wagg0/', 'random_drop1_wagg1/']


local_iters = [1, 10, 20, 40, 80, 100]
name = "iter_local1_out.txt"


nruns = 200
crounds = 100
out = np.zeros((nruns, crounds))

for folder in folders:
    for local_iter in local_iters:
        filename = f'iter_local{local_iter}_out.txt'
        full_path = cwd + prfix + folder + filename

        with open (full_path ) as f:
            for nr in range(nruns):
                for cr in range(crounds):
                    line = f.readline()
                    val = line.replace("array(", "").replace("[", "").replace("]", "").replace(",", "").replace(")", "").split()
                    out[nr,cr] = val[0]
            
        with open(f'{cwd}/processed_results/abl_extra/{folder}local_iter{local_iter}.npy', "wb") as f:
            np.save(f, out)