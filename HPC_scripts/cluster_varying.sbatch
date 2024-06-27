#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=124
#SBATCH --mail-type=END


dset="cluster_wise"
nruns=1



#SBATCH -o "results/cluster_wise/slurm-%j.out"

module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/envs/pytorch2


srun -n 1 python main.py -d ${dset} -r ${nruns} --pr True > out2.txt &

srun -n 9 python main.py -d ${dset} -r ${nruns} --pr False > out2.txt &
wait 
