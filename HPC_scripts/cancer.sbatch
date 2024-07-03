#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END


dset="cancer_iid"
nruns=1
ppc=50
crounds=100
beta=0.1
noise=1
k=4



#SBATCH -o "results/cluster_wise/slurm-%j.out"

module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/flower

prefix="raw_results/cancer/"



srun -n 200 python ../main.py -d ${dset} -r ${nruns} -k ${k}  > ${prefix}${dset}_out.txt &
wait 
