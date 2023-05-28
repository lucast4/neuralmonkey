#!/bin/bash -e

#### RULES
# SCRIPT_SUFFIX="ERROR"
# SCRIPT_SUFFIX=""
# datelist=( 220814 220815 220816 220827 220913 220921 220928 220929 220930 221001 221014 221020 221021 221031 221102 221107 221112 221114 221119)
#datelist=(221020 221021 221031 221102 221107 221112 221114 221119)

# Just the good expts, based on visually inspecting reuslts
# datelist=( 220930 221014 221020 221031 221102 221107 221114 221121 221125)
# datelist=( 221031 )

# 5/24/23 - trying to analyze all dates
datelist=( 221024 221002)
ANALY_VER="rulesw"
# ANALY_VER="ruleswERROR"

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh Pancho ${date1} trial ${ANALY_VER} |& tee -a ${logfile} &
done


sleep 2h


datelist=( 221024 221002)
# ANALY_VER="rulesw"
ANALY_VER="ruleswERROR"

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh Pancho ${date1} trial ${ANALY_VER} |& tee -a ${logfile} &
done
