#!/bin/bash -e


# ##################
animal=Pancho
datelist=(220831 220901 220902 220906 220907 220908 220909 230810 230811 230824 230826 230829 230830 231114 231116 240830 250321 250322) # ALL (confirmed)
# datelist=(220901 250321 250322)
question=RULE_ANBMCK_STROKE

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 30s
done

sleep 20m

##################
animal=Diego
datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118 240822 240827 250319 250321) # ALL (confirmed)
# datelist=(230728 230817 250319 250321) # ALL (confirmed)
# datelist=(250319) # ALL (confirmed)
question=RULE_ANBMCK_STROKE

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
  sleep 1m
done

# sleep 10m

# sleep 2h

# # ##################
# animal=Diego
# datelist=(230920 230921 230922 230924 230925 250320) # ALL (confirmed)
# # datelist=(250320) # ALL (confirmed)
# question=RULESW_ANY_SEQSUP_STROKE

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#   sleep 1m
# done

# sleep 30m

# # # ##################
# animal=Pancho
# datelist=(230920 230921 230923 231019 231020 240828 240829 250324 250325) # ALL (confirmed)
# # datelist=(250324 250325) # ALL (confirmed)
# question=RULESW_ANY_SEQSUP_STROKE

# for date1 in "${datelist[@]}"
# do
#   logfile="../logs/analy_syntax_good_eucl_state-${animal}_${date1}_${question}.txt"
#   touch ${logfile}
#   echo ${logfile}
#   python analy_syntax_good_eucl_state.py ${animal} ${date1} ${question} 2>&1 | tee ${logfile} &
#   sleep 1m
# done

# # sleep 2h

