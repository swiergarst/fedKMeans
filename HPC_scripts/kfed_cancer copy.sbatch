#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=END


dset="cancer_iid"
k=4
kl=4




#SBATCH -o "results/cluster_wise/slurm-%j.out"

module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/flower

prefix="raw_results/cancer/dennis/"


srun -n 200 python ../run_kfed.py -d ${dset} -p ${ppc} -n ${noise} -b ${beta} -k ${k} -l ${kl} > ${prefix}${dset}.txt &
wait 
