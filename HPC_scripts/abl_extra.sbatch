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
beta=0.1
noise=1

init='k-means++'
iter_local=1
drop=1
weighted_agg=1


while true;
do
	case "$1" in
		-i) init=$2; shift 2;;
		-l) iter_local=$2; shift 2;;
		-d) drop=$2; shift 2;;
		-w) weighted_agg=$2; shift 2;;
		--) shift; break ;;
		*) break ;;
	esac		
done

#SBATCH -o "results/cluster_wise/slurm-%j.out"

module use /opt/insy/modulefiles
#module load cuda/10.0 cudnn/10.0-7.4.2.24
module load miniconda/3.9
conda activate /tudelft.net/staff-bulk/ewi/insy/DBL/swiergarst/flower

prefix="raw_results/abl_extra/${init}_drop${drop}_wagg${weighted_agg}/iter_local${iter_local}_"



srun -n 200 python ../main.py -d ${dset} -r ${nruns} -p ${ppc} -c ${crounds} -n ${noise} -b ${beta} -i ${init} -l ${iter_local} -e ${drop} -w ${weighted_agg} > ${prefix}out.txt &
wait 
