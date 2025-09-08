#!/bin/bash -e


##################
animal=Diego
run=20

question=RULE_ANBMCK_STROKE
datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118 240822 240827 250319 250321 250416 250417) # ALL (confirmed)
# SUBSETS
# datelist=(230726 230815 230816 230817 230913 230915 231118) # Good subset of dates, just for testing
# datelist=(230726 230816 230913 231118) # smaller subset, just for testing
# datelist=(240822 240827 250321 250416 250417) # Two shapes
# datelist=(240822 240827) # Two shapes (not crossed)
# datelist=(230726 230815 230816 230817 230913 230915 231118) # SP vs PIG
# datelist=(230726 230815 230913 231118) # SP vs PIG (subset)

# question=RULESW_ANY_SEQSUP_STROKE
# datelist=(230920 230921 230922 230924 230925 250320) # ALL (confirmed)
# # datelist=(250320) # ALL (confirmed)

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}_${run}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} ${run} 2>&1 | tee ${logfile} &
  sleep 10s
done

sleep 3h
# sleep 1m

# ##################
animal=Pancho
run=20

question=RULE_ANBMCK_STROKE
datelist=(220901 220902 220906 220907 220908 220909 230810 230811 230824 230826 230829 231114 231116 240830 250322) # ALL (confirmed)
# datelist=(220906 220909 230829 240830 250322) # 8/26/25 - Missed
# SUBSETS
# datelist=(220902 220909 230810 230824 230826 230829 231114 231116 240830 250322) # Good subset for debugging
# datelist=(220909 230824 230829 231116 250322) # Smaller subset, good subset for debugging
# datelist=(220902 220906 220907 220908 220909 250322) # All with 2 shape sets
# datelist=(230810 230811 230826 230829 231114 231116 240830 250322) # SP vs PIG (Very sure these are the only ones.)
# datelist=(231114 231116) # [Testing]
# datelist=(220902 220909 231114 231116 250322) # missed

# datelist=(230920 230921 230923 231019 231020 240828 240829 250324 250325) # ALL (confirmed)
# # datelist=(250324 250325) # ALL (confirmed)
# question=RULESW_ANY_SEQSUP_STROKE

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}_${run}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} ${run} 2>&1 | tee ${logfile} &
  sleep 10s
done

# sleep 10m
# sleep 1m