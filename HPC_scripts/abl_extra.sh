#!/bin/sh



for init in "kmeans++" "random"
do
	for weighted in 0 1 
	do
		for drop in 0 1
		do
			for local_iter in 1 10 20 40 80 100
			do
			sbatch abl_extra.sbatch -i ${init} -l ${local_iter} -w ${weighted} -d ${drop} 
			done 
		done
	done
done
