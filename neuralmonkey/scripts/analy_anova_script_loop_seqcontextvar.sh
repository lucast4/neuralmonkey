#!/bin/bash -e

ANALY_VER="seqcontextvar"

sleep 4h

# PANCHO
datelist=(230614)
animal="Pancho"
which_level="trial"

for date1 in "${datelist[@]}"
do
    logfile="../logs/log_analy_anova_script_loop_${date1}_${ANALY_VER}_${which_level}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${animal} ${date1} ${which_level} ${ANALY_VER} 2>&1 | tee ${logfile} &

	sleep 30s
done


