#!/bin/bash -e


###########################
animal=Diego
question=RULEVSCOL_BASE_stroke
ver=stroke
#datelist=(230922 230920 230921) # SH vs DIR vs SEQSUP
#datelist=(230924 230925) # SH vs DIR vs SEQSUP vs COLOR_RANK
#datelist=(231001) # SH vs DIR vs COLOR_RANK
datelist=(230922 230920 230921 230924 230925 231001) # ALL
#datelist=(230922 230924 231001) # quick, test of all
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 30m

animal=Pancho
question=RULEVSCOL_BASE_stroke
ver=stroke
#datelist=(230923 230921 230920) # SH vs DIR vs SEQSUP
#datelist=(231019 231020) # SH vs DIR vs SEQSUP vs COLOR_RANK
datelist=(230923 230921 230920 231019 231020) # ALL
#datelist=(230923 231019) # quick, test of all
for date1 in "${datelist[@]}"
do
  logfile="../logs/log_analy_state_space_umap_${animal}_${date1}_${question}_${ver}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_state_space_umap_script.py ${animal} ${date1} 0 ${question} ${ver} 2>&1 | tee ${logfile} &
  sleep 1m
done
sleep 20m
