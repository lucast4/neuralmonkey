#!/bin/bash
set -e
date_list=(230122 230112 230117 230120 230125 230118)
animal="Pancho"
for date in "${date_list[@]}"; do
	logfile="/home/danhan/code/emre/save_local_logs/log_${date}_${animal}.txt"
	touch ${logfile}  
	echo ${logfile}
	time /home/danhan/code/neuralmonkey/neuralmonkey/scripts/load_and_save_locally_good.sh ${animal} ${date} ${date} 2>&1 | tee ${logfile}
done
