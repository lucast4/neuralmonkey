#!/bin/bash -e

animal=$1
get_all_events=0

if [[ $animal == Diego ]]; then
  # datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118) # ALL singles
  datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118 240822 240827 250319 250321 250416 250417) # ALL single (same as neural)
#   datelist=(230816 230817 230913 230914)
elif [[ $animal == Pancho ]]; then
  # datelist=(220831 220901 230810 230811 230824 230826 230829 231114 231116 240830) # ALL single
  datelist=(220901 220902 220906 220907 220908 220909 230810 230811 230824 230826 230829 231114 231116 240830 250322) # ALL single (same as neural)
  # datelist=(220902 220906 220907 220908 220909) # ALL double
  # datelist=(220831 220901 240830) # selected
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

######
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_syntax_good_gap_durations-${animal}_${date1}.txt"
  touch ${logfile}  
  echo ${logfile}
  python analy_syntax_good_gap_durations.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done