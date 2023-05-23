#!/bin/bash -e

SCRIPT_SUFFIX="ERROR"
datelist=( 220814 220815 220816 220827 220913 220921 220928 220929 220930 221001 221014 221020 221021 221031 221102 221107 221112 221114 221119)
#datelist=(221020 221021 221031 221102 221107 221112 221114 221119)

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${SCRIPT_SUFFIX}.txt"
	touch ${logfile}
    echo ${logfile}
	#taskset --cpu-list 0,1,2,3,4,5,6,7 bash ./_analy_anova_script.sh Pancho ${date1} trial rulesw |& tee -a ${logfile} &
	taskset --cpu-list 0,1,2,3,4,5,6,7 bash ./_analy_anova_script.sh Pancho ${date1} trial ruleswERROR |& tee -a ${logfile} &
done


