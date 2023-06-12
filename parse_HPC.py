import numpy as np
import os







cwd = os.path.dirname(os.path.realpath(__file__))

beta = 10

ppc = 200
#noises = [1, 1.1, 1.2, 1.3, 1.4, 1.5]



prfix = "/HPC_results/FEMNIST/"

name = "out.txt"

nruns = 200
crounds = 100
out = np.zeros((nruns, crounds))

#for n, noise in enumerate(noises):
with open (cwd + prfix + name ) as f:
    for nr in range(nruns):
        for cr in range(crounds):
            line = f.readline()
            val = line.replace("array(", "").replace("[", "").replace("]", "").replace(",", "").replace(")", "").split()
            out[nr,cr] = val[0]
        

with open("cluster_wise.npy", "wb") as f:
    np.save(f, out)