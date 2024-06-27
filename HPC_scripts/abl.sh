#!/bin/sh



for beta in 0.1 1 10
do
	for ppc in 50 100 200 
	do
		for noise in 1 1.1 1.2 1.3 1.4 1.5
		do
			sbatch abl.sbatch -p ${ppc} -b ${beta} -n ${noise}  
		done
	done
done
