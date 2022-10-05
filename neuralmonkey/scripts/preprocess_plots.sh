#!/bin/bash

datelist=( 220608 220714 220715 220716 )
for date in "${datelist[@]}"
do
	echo "$date"
	python preprocess_plots.py ${date}
done