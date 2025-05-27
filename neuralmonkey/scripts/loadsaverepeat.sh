#!/bin/bash
set -e
date_list=(231114)
animal="Pancho"
for date in "${date_list[@]}"; do
	/home/danhan/code/neuralmonkey/neuralmonkey/scripts/load_and_save_locally_good.sh ${animal} ${date} ${date}
done
