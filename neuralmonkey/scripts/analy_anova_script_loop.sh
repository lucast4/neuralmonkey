#!/bin/bash -e

#### RULES
# SCRIPT_SUFFIX="ERROR"
# SCRIPT_SUFFIX=""
# datelist=( 220814 220815 220816 220827 220913 220921 220928 220929 220930 221001 221014 221020 221021 221031 221102 221107 221112 221114 221119) # all
#datelist=(221020 221021 221031 221102 221107 221112 221114 221119)

# Just the good expts, based on visually inspecting reuslts
datelist=( 220930 221014 221020 221031 221102 221107 221114 221121 221125)

# # 5/24/23 - trying to analyze all dates, adding more
# datelist=( 221118 221113 221023 221024 221002)
# ANALY_VER="rulesw"
# # ANALY_VER="ruleswERROR"


sleep 2h

# 5/25/23 - trying to analyze all dates
# datelist=( 221002 221014 221020 221021 221023 221024 )
# datelist=(220930 221014 221020 221021 221031 221102 221107 221114 221121 221125 221118 221113 221023 221024 221002)
# datelist=(221020)
ANALY_VER="rulesw"
# ANALY_VER="ruleswALLDATA"
# ANALY_VER="ruleswERROR"

for date1 in "${datelist[@]}"
do
    logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh Pancho ${date1} trial ${ANALY_VER} 2>&1 | tee ${logfile} &
done


# sleep 3h


# # datelist=(220930 221014 221020 221021 221031 221102 221107 221114 221121 221125 221118 221113 221023 221024 221002)
# datelist=(221002)
# ANALY_VER="ruleswALLDATA"
# for date1 in "${datelist[@]}"
# do
#     logfile="log_analy_anova_script_loop_${date1}_${ANALY_VER}.txt"
# 	touch ${logfile}
#     echo ${logfile}
# 	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh Pancho ${date1} trial ${ANALY_VER} 2>&1 | tee ${logfile} &
# done
