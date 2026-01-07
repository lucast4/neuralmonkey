#!/bin/bash -e

animal=$1
get_all_events=0

if [[ $animal == Diego ]]; then
#   datelist=(230723 230724 230726 230727 230728 230730 230815 230816 230817 230913 230914 230915 231116 231118) # ALL singles
  # datelist=(230816 230724 230817 230913 230914) # gotten
  datelist=(230723 230726 230727 230728 230730 230815 230915 231116 231118) # to get

elif [[ $animal == Pancho ]]; then
#   datelist=(220831 220901 230810 230811 230824 230826 230829 231114 231116 240830) # ALL single
#   datelist=(220902 220906 220907 220908 220909) # ALL double
  # datelist=(220901 231114 220906 220908 220909) # gotten
  datelist=(220831 230810 230811 230824 230826 230829 231116 240830 220902 220907)  # not gotten (single and double)
else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

######
for date1 in "${datelist[@]}"
do
  logfile="../logs/extract_eyetracking_clusterfix_labels-${animal}_${date1}.txt"
  touch ${logfile}  
  echo ${logfile}
  python extract_eyetracking_clusterfix_labels.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done