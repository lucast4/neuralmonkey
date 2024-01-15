#!/bin/bash -e

# sleep 1h

#### RULES
# animal=Pancho
# # SCRIPT_SUFFIX="ERROR"
# # SCRIPT_SUFFIX=""
# # datelist=( 220814 220815 220816 220827 220913 220921 220928 220929 220930 221001 221014 221020 221021 221031 221102 221107 221112 221114 221119) # all
# #datelist=(221020 221021 221031 221102 221107 221112 221114 221119)

# # Just the good expts, based on visually inspecting reuslts
# datelist=( 220930 221014 221020 221031 221102 221107 221114 221121 221125)

# # # 5/24/23 - trying to analyze all dates, adding more
# # datelist=( 221118 221113 221023 221024 221002)
# # ANALY_VER="rulesw"
# # # ANALY_VER="ruleswERROR"

# # 5/25/23 - trying to analyze all dates
# # datelist=( 221002 221014 221020 221021 221023 221024 )
# # datelist=(220930 221014 221020 221021 221031 221102 221107 221114 221121 221125 221118 221113 221023 221024 221002)
# # datelist=(221020)

animal=Diego
# datelist=( 230701 230702 230704 230706 )
# datelist=( 230703 230705 230707 230713 230717 230719 ) # dirshape
datelist=( 230802 ) # dirgrammar4

ANALY_VER="rulesw"
# ANALY_VER="ruleswALLDATA"
# ANALY_VER="ruleswERROR"
which_level=trial
for date1 in "${datelist[@]}"
do
    logfile="../logs/log_analy_anova_script_loop_${date1}_${ANALY_VER}_${which_level}_${animal}.txt"
	touch ${logfile}
    echo ${logfile}
	taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${animal} ${date1} ${which_level} ${ANALY_VER} 2>&1 | tee ${logfile} &

	sleep 5s
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
