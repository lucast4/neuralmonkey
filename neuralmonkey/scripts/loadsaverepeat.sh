#!/bin/bash
set -e
# date_list=(220915 230104 230119 230112 230126 221031 221220 220908 220907 230122 221218 220901 221217 230105 230117 230124 221015 230120 230125 221024 230103 220902 230118 220918)
# date_list=(230112 230117 230118 230119 230120 230122 230125 230126)
animal="Pancho"
for date in "${date_list[@]}"; do
	logfile="/home/danhan/code/emre/save_local_logs/log_${date}_${animal}.txt"
	touch ${logfile}  
	echo ${logfile}
	time /home/danhan/code/neuralmonkey/neuralmonkey/scripts/load_and_save_locally_good.sh ${animal} ${date} ${date} 2>&1 | tee ${logfile}
done
