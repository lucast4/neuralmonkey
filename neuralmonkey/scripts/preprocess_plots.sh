#!/bin/bash


#all
# datelist=( 220608 220616 220624 220630 220714 220715 220730 220805 220816 220827 221002 221020 221107 221125 230106 230109 230126 230310 230320 )

#good subset
# datelist=( 220608 220616 220730 220816 220827 221002 )

datelist=( 230603 )

# datelist=( 220608 230320 )
for date in "${datelist[@]}"
do
	echo "$date"
	python preprocess_plots.py ${date} n n y
done