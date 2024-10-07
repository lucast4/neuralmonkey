#!/bin/bash
date_list=(221031 221122 220814)
animal="Pancho"
for date in "${date_list[@]}"; do
	echo here
	/home/danhan/code/neuralmonkey/neuralmonkey/scripts/load_and_save_locally_good.sh ${animal} ${date}
done
