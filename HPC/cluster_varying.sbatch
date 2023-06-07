#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:01:00
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --mail-type=END

dset="cluster_wise"


srun -n 200 python main.py -d ${dset} > out${SLURM_PROCID} & 