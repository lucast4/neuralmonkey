#!/bin/bash
set -e
date_list=(230913 231118)
animal="Diego"
for date in "${date_list[@]}"; do
	/home/danhan/code/neuralmonkey/neuralmonkey/scripts/load_and_save_locally_good.sh ${animal} ${date} ${date}
done
