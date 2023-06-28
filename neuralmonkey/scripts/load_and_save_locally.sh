#!/bin/bash

# Run this in base dir: /neuralmonkey
# bash scripts/load_and_save_locally.sh 2>&1 | tee preprocess_log.txt

# cd /gorilla1/code/neuralmonkey/neuralmonkey/scripts
# ./load_and_save_locally.sh 2>&1 | tee preprocess_log_221108.txt


# Mount server
server_mount

touch preprocess_log.txt

# ####### Version 1- give a list of dates
# datelist=( 220702 220703 220630 220628 220624 220616 220603 220609 )
# datelist=( 220704 )
# datelist=( 220606 220608 220609 220610 220715 220716 220717 )

# # 11/15/22
# datelist=( 221029 221031 221102 221112 221114 )

# 230205
# datelist=( 221029 221031 221102 221112 221114 )

# 230405
# datelist=( 220608 220616 220714 220715 220716 220624 220630 220730 220805 220816 220827 221002 221020 221107 221125 230106 230109 230126 230310 230320 )

# 230405_2
# datelist=( 230103 )

# # 230405_3
# datelist=( 220805 230106 230109 230126 230310 )

# 230413_3
# datelist=( 220621 220624 220703 220706 220805 220731 220927 220714 230105 230106 230727 220812 220814 220816 220823 220824 220827 220830 220831 220902 220908 220909 220913 220916 220921 220929 220928 220930 221001 221014 221015 221017 221020 221021 221031 221102 221107 221112 221114 221119 221121 230307 230320 230310 )

# 230413
# datelist=( 220715 )

# 230508
# datelist=( 220709 )

# 230517
# datelist=( 220930 221024 )

#230524
# datelist=( 221023 )
# datelist=( 221024 )
# datelist=( 221113 )
# datelist=( 220608 220610 220606 220609 )

# # 230612 - getting all the single prim expts
# datelist=( 220606 220718 220719 220724 220918 221217 221218 221220 230103 230104 )
# # datelist=( 220719 220724 220918 221217 221218 221220 230103 230104 )
# # 230612_2
# # datelist=( 220719 )


# for date1 in "${datelist[@]}"
# do
#     echo ${date1}
#     # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
#     python -m neuralmonkey.scripts.load_and_save_locally ${date1} |& tee -a preprocess_log.txt
# done



# ######## version 1 - iterate thru dates in order.
# date2=220815 # the final date to check

# d=220603 # first date to check.
# d=220707 # the final date to check

# d=220815 # first date to check.

# d=220526 # first date to check.
# date2=221004 # the final date to check

# # 221019 (#4)
# d=221004 # first date to check.
# date2=221018 # the final date to check

# # 221019 (#5)
# d=221014 # first date to check.
# date2=221022 # the final date to check

# 221019 (#6)
# d=220526 # first date to check.
# date2=221024 # the final date to check

# # 221103
# d=221020 # first date to check.
# date2=221103 # the final date to check

# 221108
# d=221027 # first date to check.
# date2=221108 # the final date to check

# 221114
# d=221027 # first date to check.
# date2=221112 # the final date to check

# 221114 - 2
# d=221027 # first date to check.
# date2=221027 # the final date to check

# # 221207
# d=221112 # first date to check.
# date2=221130 # the final date to check

#221212
# d=221029 # first date to check.
# date2=221029 # the final date to check

# # # 221212_2
# d=220526 # first date to check.
# date2=221201 # the final date to check

# # 221212_3
# d=221129 # first date to check.
# date2=221129 # the final date to check

# # # 221212_4
# d=220709 # first date to check.
# date2=220805 # the final date to check

# # 221212_5
# d=221014 # first date to check.
# date2=221202 # the final date to check

# # 221212_6
# d=220808 # first date to check.
# date2=221002 # the final date to check

# 221213_1
# d=220709 # first date to check.
# date2=220714 # the final date to check

# # 221213_2
# d=220805 # first date to check.
# date2=220805 # the final date to check

# # 221213_3
# d=221019 # first date to check.
# date2=221024 # the final date to check

# # 221213_4
# d=221029 # first date to check.
# date2=221107 # the final date to check

# # 221213_5
# d=221117 # first date to check.
# date2=221121 # the final date to check

# 230123_1
# d=221121 # first date to check.
# date2=230110 # the final date to check

# 230205_1
# d=220911 # first date to check.
# date2=231014 # the final date to check

# # 230205_2
# d=220709 # first date to check.
# date2=220724 # the final date to check

# # 230205_3
# d=220810 # first date to check.
# date2=220909 # the final date to check

# 230205_4
# d=221110 # first date to check.
# date2=221117 # the final date to check

# # 230205_5
# d=221122 # first date to check.
# date2=221125 # the final date to check

# # 230205_6
# d=221217 # first date to check.
# date2=230108 # the final date to check

# # 230205_7
# d=220719 # first date to check.
# date2=220719 # the final date to check

# 2300213
# d=221114 # first date to check.
# date2=221114 # the final date to check

# # 230326
# d=230127 # first date to check.
# date2=230319 # the final date to check

# 230326_2
# d=230320 # first date to check.
# date2=230324 # the final date to check

# 230403
# d=230324 # first date to check.
# date2=230403 # the final date to check

# 230405_4
# d=220526 # first date to check.
# date2=230403 # the final date to check

# 230525
# d=220526 # first date to check.
# date2=230403 # the final date to check

# 230621
d=230612 # first date to check.
date2=230621 # the final date to check

date1=$d
until [[ ${date1} > ${date2} ]]; do
  echo ${date1}
  # python -m neuralmonkey.scripts.load_and_save_locally ${date1} 2>>&1 | tee preprocess_log.txt
  python -m neuralmonkey.scripts.load_and_save_locally ${date1} |& tee -a preprocess_log.txt
  date1=$(date +%y%m%d -d "${date1} + 1 day")
done