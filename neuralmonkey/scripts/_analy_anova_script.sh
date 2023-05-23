#!/bin/bash -e

python analy_anova_extract.py $@ &&
python analy_anova_plot.py $@ n
#python analy_anova_plot.py $@ y
