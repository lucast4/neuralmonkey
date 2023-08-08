#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt

# cd /gorilla1/code/neuralmonkey/neuralmonkey/scripts
# ./load_and_save_locally.sh 2>&1 | tee preprocess_log_221108.txt

# d=230612 # first date to check.
# date2=230621 # the final date to check

# Mount server
# server_mount
# freiwald_mount

git_pull_all

animal=$1
d=$2
date2=$3
date1=$d

# logfile=/gorilla1/code/neuralmonkey/neuralmonkey/logs/log_preprocess_${animal}_${d}_${date2}.txt
# touch ${logfile}

until [[ ${date1} > ${date2} ]]; do
  logfile=/gorilla1/code/neuralmonkey/neuralmonkey/logs/log_preprocess_good_${animal}_${date1}.txt
  touch ${logfile}
  echo ${date1}
  echo ${logfile}
  python -m neuralmonkey.scripts.load_and_save_locally ${date1} ${animal} 2>&1 | tee ${logfile}
  date1=$(date +%y%m%d -d "${date1} + 1 day")
done