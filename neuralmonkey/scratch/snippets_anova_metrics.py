""" metrics for computing modulation, one and two-way
"""


## New metric, using running anova (2-way)
site = 314
var = "epoch"
vars_others = ["taskgroup"]
event ="02_samp"

dfthis, _, _ = SPall.dataextract_as_df_conjunction_vars(var, vars_others, site=site, event=event)

display(dfthis)

import pingouin as pg
display(pg.anova(data=dfthis, dv="fr_scalar", between=["vars_others", var], detailed=True, effsize="np2"))
display(pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="np2"))
display(pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="n2"))
aov = pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="np2")

# display(pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="n2", ss_type=1))
# display(pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="n2", ss_type=2))
# display(pg.anova(data=dfthis, dv="fr_scalar", between=[var, "vars_others"], detailed=True, effsize="n2", ss_type=3))


SPall.plotgood_rasters_smfr_each_level_combined(site, var, vars_others, event);

def _anova_running_calc_peta2(dict_results, source, ind):
    sse = dict_results[source][ind]
    ssr = dict_results["Residual"][ind]
    peta2 = sse/(sse + ssr)
    return peta2
    
    

pre_dur = 0.05
post_dur = 0.6

# time window of interest
SPall.globals_update(PRE_DUR_CALC=pre_dur, POST_DUR_CALC=post_dur)

MS = SPall._dataextract_as_metrics_scalar(dfthis, var=var)
PLOT = False
PLOT_RESULTS_STANDARD_ERROR = True
n_iter = 4
df_res = MS._anova_running_wrapper(MS.Data, var, n_iter=n_iter, PLOT=PLOT,
                                  PLOT_RESULTS_STANDARD_ERROR=PLOT_RESULTS_STANDARD_ERROR)
df_res

MS.modulationgood_wrapper_(var, version = "r2smfr_running_maxtime_oneway", return_as_score_zscore_tuple=True)

MS._anova_running_wrapper_inner(MS.Data, var=var, vars_others=None, ret)

this = MS._anova_running_compute(MS.Data, var, vars_others=None)
for source, dat in this.items():
    print(source, len(dat))







    