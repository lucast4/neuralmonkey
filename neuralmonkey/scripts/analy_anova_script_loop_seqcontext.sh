#!/bin/bash -e

# PRIMS IN GRID (SEQUENCE)
datelist=(220709 220714 220727 220731 220805 230105 230106)
ANALY_VER="seqcontext"

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh Pancho ${date1} trial ${ANALY_VER} 2>&1 | tee ${logfile} &
done


