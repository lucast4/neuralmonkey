#!/bin/bash -e

# params=( Pancho 220709 trial seqcontext )
# params=( Pancho 220709 trial seqcontext n)
# params=( Pancho 220709 trial seqcontext y)

# params=( Pancho 220714 trial seqcontext )

# params=( Pancho 220805 trial seqcontext )

# params=( Pancho 220814 trial rulesw )

# params=( Pancho 220815 trial rulesw )

# params=( Pancho 220814 trial rulesw )
# params=( Pancho 220814 trial ruleswERROR )

# params=( Pancho 220816 trial rulesw )
# params=( Pancho 220816 trial ruleswERROR )

# params=( Pancho 220827 trial rulesw )
# params=( Pancho 220827 trial ruleswERROR )

# params=( Pancho 220913 trial rulesw )
# params=( Pancho 220913 trial ruleswERROR )

# params=( Pancho 220921 trial rulesw )

# params=( Pancho 220928 trial rulesw )

# params=( Pancho 220929 trial rulesw )

# params=( Pancho 220930 trial rulesw )

# params=( Pancho 221001 trial rulesw )

# params=( Pancho 221014 trial rulesw )
# params=( Pancho 221014 trial ruleswERROR )

params=( Pancho 221020 stroke rulesw )

# params=( Pancho 221020 trial rulesw )
# params=( Pancho 221020 trial ruleswALLDATA )
# params=( Pancho 221020 trial rulesw n)
# params=( Pancho 221020 trial rulesw y)
# params=( Pancho 221020 trial ruleswERROR )

# params=( Pancho 221021 trial rulesw )

# params=( Pancho 221031 trial rulesw )
# params=( Pancho 221031 trial ruleswERROR )

# params=( Pancho 221102 trial rulesw )

# params=( Pancho 221107 trial rulesw )
# params=( Pancho 221107 trial ruleswERROR )

# params=( Pancho 221112 trial rulesw )

# params=( Pancho 221114 trial rulesw )
# params=( Pancho 221114 trial ruleswERROR )

# params=( Pancho 221119 trial rulesw )
# params=( Pancho 221119 trial ruleswERROR )

# params=( Pancho 221121 trial rulesw )

# params=( Pancho 221125 trial rulesw )
# params=( Pancho 221125 trial ruleswALLDATA )
# params=( Pancho 221125 trial ruleswERROR )

# params=( Pancho 230105 trial seqcontext )
# params=( Pancho 230105 trial seqcontext n)
# params=( Pancho 230105 trial seqcontext y)

# params=( Pancho 220715 trial singleprim )
# params=( Pancho 220715 trial singleprim n)

# params=( Pancho 220716 trial singleprim )

# python analy_anova_extract.py ${params[@]}
# python analy_anova_plot.py ${params[@]} n
# python analy_anova_plot.py ${params[@]} y

taskset --cpu-list 0,1,2,3,4,5,6 bash ./_analy_anova_script.sh ${params[@]}
