#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:10:00
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=124
#SBATCH --mail-type=END


dset="abl"
nruns=1
ppc=50
crounds=100
beta=1
noise=1



while true;
do
	case "$1" in
		-p) ppc=$2; shift 2;;
		-c) crounds=$2; shift 2;;
		-b) beta=$2; shift 2;;
		-n) noise=$2; shift 2;;
		--) shift; break ;;
		*) break ;;
	esac		
done

#SBATCH -o "results/cluster_wise/slurm-%j.out"

module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/flower

prefix="results/abl/beta${beta}/ppc${ppc}/noise_${noise}_"

srun -n 200 python main.py -d ${dset} -r ${nruns} -p ${ppc} -c ${crounds} -n ${noise} -b ${beta}  > ${prefix}out.txt &
wait 
