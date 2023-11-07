""" 
How consistent is activity aligned to specific temproal events.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from pythonlib.tools.plottools import savefig

def preprocess_and_plot(MS, SAVEDIR, session = 0, DEBUG=False):
    """
    All steps to extract and plot analyses related to modulation by events/
    """
    
    ## from neuralmonkey.classes.snippets import datasetstrokes_extract
    from neuralmonkey.analyses.site_anova import _dataset_extract_prune_rulesw, _dataset_extract_prune_sequence, params_database_extract
    import os
    import numpy as np
    import seaborn as sns
    from neuralmonkey.classes.snippets import Snippets, extraction_helper
    from pythonlib.tools.pandastools import pivot_table


    SN = MS.SessionsList[session]

    SP_trial = extraction_helper(SN, which_level="trial")
    SP_stroke = extraction_helper(SN, which_level="stroke")

    # 1) Concatenate dfscalars
    # (generate a common column, using goruping)
    grouping_variables = ["event_aligned"]
    df1 = SP_trial.dataextract_as_df(grouping_variables, "event_var_level")

    grouping_variables = ["stroke_index_semantic"]
    df2 = SP_stroke.dataextract_as_df(grouping_variables, "event_var_level")
    df2["event_var_level"] = "stroke-" + df2["event_var_level"]

    list_site_good = [site for site in SP_trial.Sites if site in SP_stroke.Sites]    

    if DEBUG:
        list_site_good = list_site_good[::10]

    # concatenate the df
    import pandas as pd
    dfscalar_all = pd.concat([df1, df2])
    dfscalar_all["event_aligned"] = dfscalar_all["event_var_level"]

    from neuralmonkey.metrics.scalar import MetricsScalar
    Mscal = MetricsScalar(dfscalar_all)


    ##### COMPUTE EVENT MODULATION
    res_all = []

    # Remove events from MScal which you dont want
    events_remove = ["stroke-both_fl"] # remove stroke-both_fl since is low dsample size and not relevant?
    Mscal.ListEventsUniqname = [ev for ev in Mscal.ListEventsUniqname if ev not in events_remove]

    # THis is the "stroke-ALL", which combines all strokes.
    event_stroke_all_tuple = ("stroke-last", "stroke-first", "stroke-both_fl", "stroke-middle")
    list_event_get = Mscal.ListEventsUniqname + [event_stroke_all_tuple]

    print("Running Mscal.modulationbytime_calc_this using these events:")
    for ev in list_event_get:
        print(ev)

    for site in list_site_good:

        if site%20==0:
            print(site)
            
        # 1) Compute matrix for trial-level data
        res_this = Mscal.modulationbytime_calc_this(site, list_event=list_event_get)
        res_all.extend(res_this)

    df_modtime = pd.DataFrame(res_all)
    df_modtime["event_var_level"] = df_modtime["event"]

    # rename strokes all
    from pythonlib.tools.pandastools import applyFunctionToAllRows
    def F(x):
        if x["event_var_level"] == event_stroke_all_tuple:
            return "stroke-ALL"
        else:
            return x["event_var_level"]
        
    df_modtime = applyFunctionToAllRows(df_modtime, F, "event_var_level")

    # give site_area names
    def F(x):
        return SP_trial.SN.sitegetter_summarytext(x["site"])
    df_modtime = applyFunctionToAllRows(df_modtime, F, newcolname="site_region")

    def F(x):
        return SP_trial.SN.sitegetter_map_site_to_region(x["site"])
    df_modtime = applyFunctionToAllRows(df_modtime, F, newcolname="region")


    # SAVE
    import pickle

    path = f"{SAVEDIR}/df_modtime.pkl"
    with open(path, "wb") as f:
        pickle.dump(df_modtime, f)
        
    path = f"{SAVEDIR}/Mscal.pkl"
    with open(path, "wb") as f:
        pickle.dump(Mscal, f)    

    if False:
        # too large, like 0.5-1G?
        path = f"{SAVEDIR}/SP_stroke.pkl"
        SP_stroke.save(savedir=SAVEDIR, name="SP_stroke.pkl")

    ##################### PLOTS
    plot_overview(df_modtime, SAVEDIR)

    ########## PLOT HEATMAP
    sdir = f"{SAVEDIR}/heatmaps_smfr-stroke"
    os.makedirs(sdir, exist_ok=True)
    SP_stroke.plotwrapper_heatmap_smfr(which_var="stroke_index_semantic", sdir=sdir)

    sdir = f"{SAVEDIR}/heatmaps_smfr-trial"
    os.makedirs(sdir, exist_ok=True)
    SP_trial.plotwrapper_heatmap_smfr(which_var="event_aligned", sdir=sdir)
    if False:
        import random
        site = random.sample(SP.Sites, 1)[0]
        SP.plot_smfr_average_each_level(site, list_var=["ind_taskstroke_orig"], list_events_uniqnames=["00_stroke"])
        sn.sitegetter_summarytext(site)

    return SP_trial, SP_stroke, Mscal, df_modtime


def kernel_compute(df_modtime, response="r2_time_minusmean", sdir=None, normalize_by_row = True):
    """
    Compute kernel score for each row in dfmodtime, and here also includes repositoru of dict holding events.

    """
    from pythonlib.tools.pandastools import pivot_table

    if "site_region" in df_modtime.columns:
        df_modtime_wide = pivot_table(df_modtime, ["site", "region", "site_region"], 
                                      "event_var_level", [response], flatten_col_names=True)
    else:
        df_modtime_wide = pivot_table(df_modtime, ["site", "region"], 
                                      "event_var_level", [response], flatten_col_names=True)


    # normalize
    if normalize_by_row:
        list_ev_var = df_modtime["event_var_level"].unique().tolist()
        list_ev_var = [f"{response}-{ev_var}" for ev_var in list_ev_var]
        df_modtime_wide_normrowsubtr = df_modtime_wide.loc[:, list_ev_var]
        df_modtime_wide_normrowsubtr = df_modtime_wide_normrowsubtr.subtract(df_modtime_wide_normrowsubtr.mean(axis=1), axis=0)
    else:
        df_modtime_wide_normrowsubtr = df_modtime_wide

    # get other cols
    for key in ["site", "region", "site_region"]:
        if key in df_modtime_wide.columns:
            df_modtime_wide_normrowsubtr[key] = df_modtime_wide[key]

    # normalize each weigths
    dict_kernels = {
        ("01_samp", "00_fix_touch", "03_first_raise", "stroke-first"):[3, -1, -1, -1], # visual vs. motor
        ("04_off_stroke_last", "00_fix_touch", "03_first_raise", "stroke-first", "stroke-middle"):[4, -1, -1,-1,-1], # decide done.
        ("05_doneb", "00_fix_touch", "03_first_raise", "stroke-first"):[3, -1, -1, -1], # done button vs. motor
        ("07_reward_all", "01_samp", "00_fix_touch", "03_first_raise", "stroke-first"):[4, -1, -1, -1,-1], # rew vs. rest
        # ("stroke-last", "00_fix_touch", "03_first_raise"):[3, -1, -1, -1], # stroke vs. motor in general
        ("stroke-middle", "stroke-last", "00_fix_touch", "05_doneb"):[1, 1, -1, -1], # stroke vs. motor in general
        ("stroke-ALL", "00_fix_touch", "05_doneb"):[2, -1, -1], # stroke vs. motor in general
        ("stroke-first", "stroke-middle", "stroke-last", "00_fix_touch", "05_doneb"):[1, 1, 1, -1.5, -1.5], # stroke vs. motor in general
    #     ("stroke-first", "stroke-middle", "stroke-last", "00_fix_touch", "02_go_cue"):[1, 1, 1, -1.5, -1.5], # stroke vs. motor in general
        # ("stroke-first", "stroke-last"):[1,-1], # stroke first vs. last
        # ("stroke-first", "stroke-last", "stroke-middle", "00_fix_touch", "03_first_raise"):[1.5, 1.5, -1, -1, -1], # stroke (first last) vs. rest
        ("stroke-first", "04_off_stroke_last", "stroke-middle", "stroke-last", "00_fix_touch", "03_first_raise"):[2, 2, -1, -1, -1, -1], # edges 
        ("02_go_cue", "00_fix_touch", "03_first_raise", "stroke-first"):[3,-1, -1, -1], # start (cognitive) vs. movement
    }
    for kernel, weights in dict_kernels.items():
        assert sum(weights)==0
        weights = weights/np.sum(np.abs(weights), keepdims=True)
        dict_kernels[kernel]=weights

    if sdir is not None:
        dict_kernels_stringy = {}
        for key, value in dict_kernels.items():
            dict_kernels_stringy[", ".join(key)] = np.array2string(value)
            
        # save the kernel info
        from pythonlib.tools.expttools import writeDictToYaml
        writeDictToYaml(dict_kernels_stringy, f"{sdir}/dict_kernels.yaml")

    # 1) compute scores
    res = []
    for i, (kernel, weights) in enumerate(dict_kernels.items()):
        
        scores = _kernel_compute_scores(df_modtime_wide_normrowsubtr, 
            kernel, weights, response=response)

        # # get dot product
        # columns = [f"{response}-{k}" for k in kernel]
        # xmat = df_modtime_wide_normrowsubtr.loc[:, columns].to_numpy()
        # scores = np.matmul(xmat, weights)
        
        # expand to each unit
        regions = df_modtime_wide_normrowsubtr.loc[:, "region"].tolist()
        sites = df_modtime_wide_normrowsubtr.loc[:, "site"].tolist()
            
        for s, r, scor in zip(sites, regions, scores):
            res.append({
                "site":s,
                "region":r,
                "kernel_score":scor,
                "kernel":kernel,
                "kernel_id":i,            
            })
    df_kernels = pd.DataFrame(res) 

    return df_kernels, df_modtime_wide, df_modtime_wide_normrowsubtr, dict_kernels

def _kernel_compute_scores(df_modtime_wide, kernel, weights, response="r2_time_minusmean"):
    """ 
    Compure scores (single vector) for this kernal and associated weights.
    PARAMS:
    - df_modtime_wide, where columns include the items in kernel
    - kernel, list-like, where each picks out a column
    - weights, list-like of weigghts, one for each k in kernel.
    RETURNS:
    - scores, array of size num rows of df_modtime_wide, for each row gets its
    weighted score
    """
    # get dot product
    assert len(weights)==len(kernel)
    columns = [f"{response}-{k}" for k in kernel]
    xmat = df_modtime_wide.loc[:, columns].to_numpy()
    scores = np.matmul(xmat, weights)
    return scores

def _kernel_compute_scores_pos_and_neg(df_modtime_wide, kernel, weights, response="r2_time_minusmean"):
    """
    Get the scores for the positive and negative components of the kernel.
    RETURNS:
    - scores_pos, scores_neg, both array-like, len same as df_modtime_wide nrows. Note:
    scores_neg is flipped so positive means more strong effect of the negative kernels
    """
    kernel_pos = []
    weights_pos = []
    kernel_neg = []
    weights_neg = []
    
    for k, w in zip(kernel, weights):
        if w>0:
            kernel_pos.append(k)
            weights_pos.append(w)
        elif w<0:
            kernel_neg.append(k)
            weights_neg.append(w)
        else:
            assert False
        
    # get scores
    scores_pos = _kernel_compute_scores(df_modtime_wide, kernel_pos, weights_pos, response)
    scores_neg = _kernel_compute_scores(df_modtime_wide, kernel_neg, weights_neg, response)

    # Flip sign of scores_neg, since the more negative they are, the more strongly they are present..
    scores_neg = -scores_neg    

    return scores_pos, scores_neg



def plot_overview(df_modtime, SAVEDIR, response = "r2_time_minusmean",
    PLOT_KERNELS=True):
    """ Helkper for the overall plots to look at event encoding
    """
    
    from pythonlib.tools.plottools import plotScatterXreduced, plotScatterOverlay
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    from pythonlib.tools.snstools import rotateLabel

    sdir = f"{SAVEDIR}/modulation"
    os.makedirs(sdir, exist_ok=True)

    for norm_method in [None, "row_sub"]:
        for val_name in ["r2_time", "r2_time_zscored", "r2_time_minusmean"]:
            
            # 1) Plot summary (mean over sites)
            fig, ax = plt.subplots(1,1, figsize=(8,8))
            convert_to_2d_dataframe(df_modtime, col1="region", col2="event_var_level", plot_heatmap=True, 
                                    agg_method="mean", val_name=val_name, annotate_heatmap=False, ax=ax, 
                                   norm_method=norm_method);
            savefig(fig, f"{sdir}/heatmapsummary-region_by_event-norm_{norm_method}-score_{val_name}.pdf")
                
            # 2) Plot each site. Only do this if site_region exists (i.e. combined datasets removes this
            # which is good, since too much data to plot)
            if "site_region" in df_modtime.columns:
                fig, ax = plt.subplots(1,1, figsize=(6,20))
                convert_to_2d_dataframe(df_modtime, col1="site_region", col2="event_var_level", plot_heatmap=True, 
                                        agg_method="mean", val_name=val_name, annotate_heatmap=False, ax=ax, 
                                       norm_method=norm_method, dosort_colnames=False,
                                       list_cat_2 = None);        
                savefig(fig, f"{sdir}/heatmapunits-region_by_event-norm_{norm_method}-score_{val_name}.pdf")    
            
            plt.close("all")

    # Bar plot and scatter summaries of scores
    for val_name in ["r2_time", "r2_time_zscored", "r2_time_minusmean"]:

        fig = sns.catplot(data=df_modtime, x="event_var_level", y=val_name, col="region", col_wrap=4, kind="bar", ci=68)
        rotateLabel(fig)
        savefig(fig, f"{sdir}/barsummary-region_by_event-score_{val_name}.pdf")

        fig = sns.catplot(data=df_modtime, x="event_var_level", y=val_name, col="region", col_wrap=4,
                         alpha=0.25, jitter=True)
        for ax in fig.axes.flatten():
            ax.axhline(0, color="k", alpha=0.2)
        rotateLabel(fig)
        savefig(fig, f"{sdir}/scattersummary-region_by_event-score_{val_name}.pdf")

        plt.close("all")

    ######### Compute kernels
    if PLOT_KERNELS:
        for row_norm in [False, True]:
            sdir_kernel = f"{sdir}/kernels-norm_by_rows_{row_norm}"
            os.makedirs(sdir_kernel, exist_ok=True)
     
            # list_ev_var = df_modtime["event_var_level"].unique().tolist()
            df_kernels, df_modtime_wide, df_modtime_wide_normrowsubtr, dict_kernels = kernel_compute(
                df_modtime, response=response, sdir=sdir_kernel, normalize_by_row=row_norm)

            ####### Plots
            fig = sns.catplot(data=df_kernels, x="kernel_score", y="region", col="kernel", col_wrap=3, aspect=1, alpha=0.35,
                         sharex=False)
            for ax in fig.axes.flatten():
                ax.axvline(0, color="k")
                
            savefig(fig, f"{sdir_kernel}/kernel_scores-scatter.pdf")


            fig = sns.catplot(data=df_kernels, x="kernel_score", y="region", col="kernel", col_wrap=3, aspect=1,
                             kind="bar", ci=68, sharex=False)
            for ax in fig.axes.flatten():
                ax.axvline(0, color="k")
            savefig(fig, f"{sdir_kernel}/kernel_scores-bar.pdf")        

            plt.close("all")

            from neuralmonkey.neuralplots.brainschematic import plot_scalar_values, plot_df
            sdirthis = f"{sdir_kernel}/brain_schematic"
            os.makedirs(sdirthis, exist_ok=True)

            # plot_scalar_values(regions, scores, diverge=False)
            plot_df(df_kernels, valname="kernel_score", subplot_var="kernel", savedir=sdirthis, diverge=False)
            plt.close("all")


            for kernel in dict_kernels.keys():
                dfthis = df_kernels[df_kernels["kernel"]==kernel].reset_index(drop=True)
                
                sdirthis = f"{sdir_kernel}/brain_schematic-kernel_{kernel}"
                os.makedirs(sdirthis, exist_ok=True)
                
                plot_df(dfthis, valname="kernel_score", savedir=sdirthis, diverge=True)

                plt.close("all")

            ############### PIARWISE PLOTS]
            sdirthis = f"{sdir_kernel}/kernel_all_pairwise"
            os.makedirs(sdirthis, exist_ok=True)        
            overlay_mean = False
            regions = df_modtime_wide_normrowsubtr["region"].tolist()
            text_to_plot = df_modtime_wide_normrowsubtr["site"].tolist()

            for i, (kernel, weights) in enumerate(dict_kernels.items()):
                col_positive = [k for k, w in zip(kernel, weights) if w>0]
                col_negative = [k for k, w in zip(kernel, weights) if w<0]
                
                
                for colp in col_positive:
                    for coln in col_negative:
                
                        keys = [f"{response}-{colp}", f"{response}-{coln}"]

                        X = df_modtime_wide_normrowsubtr.loc[:, keys].to_numpy()

                        for plot_text_over_examples in [True, False]:
                            fig, axes = plotScatterOverlay(X, labels=regions, alpha=0.5, ver="separate", downsample_auto=False, SIZE=3, 
                                               overlay_mean=overlay_mean, plot_text_over_examples=plot_text_over_examples, text_to_plot=text_to_plot)
                            # label
                            axes[0][0].set_xlabel(keys[0])
                            axes[0][0].set_ylabel(keys[1])

                            # cross lines
                            for ax in axes.flatten():
                                ax.axhline(0, alpha=0.25)
                                ax.axvline(0, alpha=0.25)

                            # save
                            savefig(fig, f"{sdirthis}/kernel_{i}-scatter-{keys[0]}-vs-{keys[1]}-text_{plot_text_over_examples}.pdf")

                            plt.close("all") 
                
                #### Compute scatter of negative vs. positive (i..e, first aggregate 
                # the features before plotting)
                kernel_pos = []
                weights_pos = []
                kernel_neg = []
                weights_neg = []
                
                for k, w in zip(kernel, weights):
                    if w>0:
                        kernel_pos.append(k)
                        weights_pos.append(w)
                    elif w<0:
                        kernel_neg.append(k)
                        weights_neg.append(w)
                    else:
                        assert False
                    
                # get scores
                scores_pos = _kernel_compute_scores(df_modtime_wide_normrowsubtr, kernel_pos, weights_pos, response)
                scores_neg = _kernel_compute_scores(df_modtime_wide_normrowsubtr, kernel_neg, weights_neg, response)

                # Flip sign of scores_neg, since the more negative they are, the more strongly they are present..
                scores_neg = -scores_neg

                # plot
                X = np.stack((scores_pos, scores_neg), axis=1)
                assert X.shape[0] == len(regions)
                        
                for plot_text_over_examples in [True, False]:

                    fig, axes = plotScatterOverlay(X, labels=regions, alpha=0.5, ver="separate", downsample_auto=False, SIZE=3, 
                                       overlay_mean=overlay_mean, plot_text_over_examples=plot_text_over_examples, text_to_plot=text_to_plot)
                    # label
                    axes[0][0].set_xlabel("positive features")
                    axes[0][0].set_ylabel("negative features")

                    # cross lines
                    for ax in axes.flatten():
                        ax.axhline(0, alpha=0.25)
                        ax.axvline(0, alpha=0.25)

                    # save
                    fig.savefig(f"{sdirthis}/kernel_{i}-scatter-neg_vs_pos_features-text_{plot_text_over_examples}.pdf")
                    plt.close("all") 

            ##### SUMMARIZE across kernels, for each brain region, its kernels.
            sdirthis = f"{sdir_kernel}/kernel_summary_each_region"
            os.makedirs(sdirthis, exist_ok=True)
            
            # 1) bar plots
            fig = sns.catplot(data=df_kernels, x="kernel_id", y="kernel_score", col="region", col_wrap=4, kind="bar", ci=68)
            fig.savefig(f"{sdirthis}/bar_each_region.pdf")

            # 2) heatmap
            fig, ax = plt.subplots(1,1)
            convert_to_2d_dataframe(df=df_kernels, col1="region", col2="kernel_id", plot_heatmap=True, agg_method="mean",
                                   val_name="kernel_score", diverge=True, annotate_heatmap=False, ax=ax);
            fig.savefig(f"{sdirthis}/heatmap_each_region-diverge.pdf")

            fig, ax = plt.subplots(1,1)
            convert_to_2d_dataframe(df=df_kernels, col1="region", col2="kernel_id", plot_heatmap=True, agg_method="mean",
                                   val_name="kernel_score", diverge=False, annotate_heatmap=False, ax=ax);
            fig.savefig(f"{sdirthis}/heatmap_each_region.pdf")

            plt.close("all")


