"""
Organizing good plots for syntax, gap duration analysis, inclding beh (draw and eye).

Ie evidence for use of grammar structure from beh (draw and eye).

NOTEBOOK: 250715_syntax_gap_durations.ipynb

"""

import sys
from pythonlib.tools.plottools import savefig
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pythonlib.tools.plottools import savefig
from pythonlib.dataset.dataset_analy.grammar import syntaxconcrete_extract_more_info, syntaxconcrete_extract_more_info_eye_fixation, syntaxconcrete_dfmod_postprocess

# def syntaxconcrete_extract_more_info(syntax_concrete, index_gap):
#     """
#     Given syntax_concrete and a gap index, get more useful information about
#     the current state (i.e, like the state of the drawing agent)
#     """

#     s_tuple = []
#     for chunk_rank, n_in_chunk in enumerate(syntax_concrete):
#         s_tuple.extend([chunk_rank for _ in range(n_in_chunk)])

#     if index_gap < 0:
#         # Then this is any time before first stroke
#         pre_chunk_rank_global = -1
#         s_post = s_tuple
#         s_pre = tuple([])
#     else:
#         assert index_gap>-1
#         s_post = s_tuple[(index_gap+1):]
#         s_pre = s_tuple[:(index_gap+1)]
#         pre_chunk_rank_global = s_pre[-1]
#     post_chunk_rank_global = s_post[0]

#     n_remain_in_chunk = sum([x==post_chunk_rank_global for x in s_post])
#     n_completed_in_chunk = sum([x==post_chunk_rank_global for x in s_pre])

#     n_in_chunk = n_completed_in_chunk + n_remain_in_chunk

#     post_chunk_within_rank = n_completed_in_chunk

#     # Remaining chunk ranks
#     s_post_not_yet_started = [cr for cr in s_post if not cr==pre_chunk_rank_global]
#     chunk_ranks_remain_not_yet_started = list(set(s_post_not_yet_started))

#     # Mapping: rank --> chunk_rank
#     map_rank_to_chunk_rank = {}
#     for rank, chunk_rank in enumerate(s_tuple):
#         map_rank_to_chunk_rank[rank] = chunk_rank    

#     # Mapping: rank --> rank_within_chunk
#     cr_current = -1
#     map_rank_to_rank_within_chunk = {}
#     for rank, cr in enumerate(s_tuple):
#         if cr!=cr_current:
#             # A new chunk_rank. Do reset.
#             cr_current = cr
#             counter_within_chunk_rank = 0
#         else:
#             counter_within_chunk_rank += 1
#         map_rank_to_rank_within_chunk[rank] = counter_within_chunk_rank
#     map_rank_to_rank_within_chunk.values()    

#     # Get the indices where transitions occur (ie.. AAABBC would return [2, 4])
#     n_strokes = len(s_tuple)
#     inds_transition = []
#     counter = -1
#     for x0 in syntax_concrete:
#         counter+=x0
#         if counter>-1 and counter<n_strokes-1:
#             inds_transition.append(counter)
#     inds_transition = sorted(set(inds_transition))


#     info = {
#         "syntax_concrete":syntax_concrete,
#         "index_gap":index_gap,
#         "index_gap_is_chunk_switch":post_chunk_rank_global>pre_chunk_rank_global,
#         "s_tuple":s_tuple,
#         "s_tuple_remain":s_post,
#         "chunk_ranks_remain_not_yet_started":chunk_ranks_remain_not_yet_started,
#         "map_rank_to_chunk_rank":map_rank_to_chunk_rank,
#         "map_rank_to_rank_within_chunk":map_rank_to_rank_within_chunk,
#         "pre_chunk_rank_global":pre_chunk_rank_global,
#         "post_chunk_rank_global":post_chunk_rank_global,
#         "n_remain_in_chunk":n_remain_in_chunk,
#         "n_completed_in_chunk":n_completed_in_chunk,
#         "n_in_chunk":n_in_chunk,
#         "post_chunk_within_rank":post_chunk_within_rank,
#         "inds_stroke_before_chunk_transition":inds_transition,
#     }
    
#     return info
 
# def syntaxconcrete_extract_more_info_eye_fixation(syntax_concrete, index_gap, tokens_correct_order, list_fixed_idx_task):
#                                                 #   dfgaps, trialcode):
#     """
#     Given syntax_concrete and a gap index, get more useful information about
#     the current state of the eye fixation, ie., the task shape that is currnetly
#     being looked at (fixated).

#     PARAMS:
#     - syntax_concrete, e.g, (0, 2, 3) or (2,2,1)
#     - index_gap, int, state of drwing, e.g, where 0 means just finished stroke 0. Note: -1 means the gap before first stroke, which
#     means any/all time before first stroke.
#     - tokens_correct_order, the tokens, in correct order of drwaing. This is required to map from idx_task to 
#     the actual token
#     - fixed_idx_task, int, where you are looking at, ie the index_task. This is the same as tok["ind_taskstroke_orig"]
#     where tok is an item in tokens_correct_order
#     """

#     assert index_gap>=-2
#     if index_gap==-2:
#         # This is the code for (between samp -- go)
#         index_gap = -1
    
#     ### Prep -- get information about this gap
#     # First, get gap information
#     if False:
#         dfrow = dfgaps[(dfgaps["trialcode"]==trialcode) & (dfgaps["index_gap"]==index_gap)]
#         assert len(dfrow)==1

#         # gap_chunk_rank_global
#         # diff_chunk_rank_global
#         chunk_pre = dfrow["pre_chunk_rank_global"].values[0]
#         chunk_post = dfrow["post_chunk_rank_global"].values[0]
#         assert dfrow["syntax_concrete"].values[0] == syntax_concrete
#         syntax_concrete = dfrow["syntax_concrete"].values[0]
    
#     info_gap = syntaxconcrete_extract_more_info(syntax_concrete, index_gap)
#     # assert chunk_pre == info_gap["pre_chunk_rank_global"]
#     # assert chunk_post == info_gap["post_chunk_rank_global"]
#     chunk_pre = info_gap["pre_chunk_rank_global"]
#     chunk_post = info_gap["post_chunk_rank_global"]
#     map_rank_to_chunk_rank = info_gap["map_rank_to_chunk_rank"]
#     map_rank_to_rank_within_chunk = info_gap["map_rank_to_rank_within_chunk"]

#     rank_pre = index_gap # ie index_gap = 0 means the gap between rank 0 and rank 1.
#     rank_post = index_gap + 1

#     ### For each fixation, get its information, within this gap
#     # Convert task index to rank, this is easier (map from token index, to its rank (in correct sequence))
#     map_taskidx_to_rank = {}
#     for rank, tok in enumerate(tokens_correct_order):
#         idx_orig = tok["ind_taskstroke_orig"]
#         map_taskidx_to_rank[idx_orig] = rank

#     list_info_fixation = []
#     for fixed_idx_task in list_fixed_idx_task:
#         info_fixation = {}

#         fixed_rank = map_taskidx_to_rank[fixed_idx_task] # the rank in sequence of strokes, for this fixated shape

#         # (1) Info related to the fixated chunk_rank
#         fixed_chunk_rank = map_rank_to_chunk_rank[fixed_rank]

#         # Number, summarizing
#         info_fixation["chunkrank_fixed"] = fixed_chunk_rank
#         info_fixation["chunkrank_fixed_minus_post"] = fixed_chunk_rank - chunk_post
#         info_fixation["chunkrank_fixed_minus_pre"] = fixed_chunk_rank - chunk_pre

#         # Semantic label, related to donness of chunk
#         if fixed_chunk_rank < chunk_pre :
#             assert fixed_chunk_rank < chunk_post, "logically not possible"
#             info_fixation["chunkrank_fixed_status"] = "completed_before_last_stroke"
#         elif (fixed_chunk_rank == chunk_pre) and (fixed_chunk_rank < chunk_post):
#             info_fixation["chunkrank_fixed_status"] = "completed_by_last_stroke"
#         elif (fixed_chunk_rank == chunk_pre) and (fixed_chunk_rank == chunk_post):
#             info_fixation["chunkrank_fixed_status"] = "ongoing"
#         elif (fixed_chunk_rank > chunk_pre) and (fixed_chunk_rank == chunk_post):
#             info_fixation["chunkrank_fixed_status"] = "start_in_next_stroke"
#         elif (fixed_chunk_rank > chunk_pre) and (fixed_chunk_rank > chunk_post):
#             info_fixation["chunkrank_fixed_status"] = "start_after_next_stroke"
#         else:
#             print(fixed_chunk_rank, chunk_pre, chunk_post)
#             assert False

#         # Another semantic label, differentiating based on inner rank within chunk.
#         # This is differnet from above, in that it defines "next chunk" to be the one
#         # different from current, even when currently within a chunk. In contrast, above,
#         # next chunk would be the current chunk.
#         current_chunk = chunk_pre

#         # Get the actual chunk ranks rthat are upcoming on this trial
#         # Note: if this trial skips a chunk, then next chunk jumps also.
#         # next_chunk = chunk_pre + 1
#         next_chunk, nextnext_chunk, nextnextnext_chunk = None, None, None
#         chunk_ranks_remain_not_yet_started = info_gap["chunk_ranks_remain_not_yet_started"]
#         if len(chunk_ranks_remain_not_yet_started)>0:
#             next_chunk = chunk_ranks_remain_not_yet_started[0] # e.g, [1,3] means these are the following hcunk ranks
#         if len(chunk_ranks_remain_not_yet_started)>1:
#             nextnext_chunk = chunk_ranks_remain_not_yet_started[1] # e.g, [1,3] means these are the following hcunk ranks
#         if len(chunk_ranks_remain_not_yet_started)>2:
#             nextnextnext_chunk = chunk_ranks_remain_not_yet_started[2] # e.g, [1,3] means these are the following hcunk ranks
        
#         fixed_rank_within_chunk = map_rank_to_rank_within_chunk[fixed_rank]
#         info_fixation["rank_within_chunk_fixed"] = fixed_rank_within_chunk
#         assert fixed_chunk_rank == info_fixation["chunkrank_fixed"]

#         # if (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk==0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_first_stk"

#         # elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk>0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_inner_stk"

#         # elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk==0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_first_stk"

#         # elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk>0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_inner_stk"

#         # elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk==0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_first_stk"

#         # elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk>0):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_inner_stk"

#         # elif (fixed_chunk_rank == current_chunk):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk"

#         # elif (fixed_chunk_rank < current_chunk):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "any_chk_fini_b4_prev_stk"

#         # else:
#         #     print(fixed_chunk_rank, fixed_rank_within_chunk)
#         #     print(current_chunk, next_chunk)
#         #     print(info_fixation)
#         #     assert False

#         if (fixed_rank == rank_post):
#             # Then looking at next strokes. Important to have this (even thuogh it is the
#             # only one that cares about rank, not chunkrank) beucase it is "null model", what we
#             # expect during non-cr-transitioning gaps)
#             info_fixation["chunkrank_fixed_status_v2"] = "ntx_rank"

#         elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk==0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_first_stk"

#         elif (fixed_chunk_rank == nextnextnext_chunk) and (fixed_rank_within_chunk>0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxtnxtnxt_chk_inner_stk"

#         elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk==0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_first_stk"

#         elif (fixed_chunk_rank == nextnext_chunk) and (fixed_rank_within_chunk>0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxtnxt_chk_inner_stk"

#         elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk==0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_first_stk"

#         elif (fixed_chunk_rank == next_chunk) and (fixed_rank_within_chunk>0):
#             info_fixation["chunkrank_fixed_status_v2"] = "nxt_chnk_inner_stk"

#         elif (fixed_chunk_rank == current_chunk) and (fixed_rank < rank_post):
#             info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk_past"

#         elif (fixed_chunk_rank == current_chunk) and (fixed_rank >= rank_post):
#             info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk_future"

#         # elif (fixed_chunk_rank == current_chunk):
#         #     info_fixation["chunkrank_fixed_status_v2"] = "chk_prev_stk"

#         elif (fixed_chunk_rank < current_chunk):
#             info_fixation["chunkrank_fixed_status_v2"] = "any_chk_fini_b4_prev_stk"

#         else:
#             print(fixed_chunk_rank, fixed_rank_within_chunk)
#             print(current_chunk, next_chunk)
#             print(info_fixation)
#             assert False


#         # (2) Info related to the rank (ignoring chunk)
#         assert rank_pre == rank_post-1
#         # res["rank_fixed_minus_post"] = fixed_rank - rank_post
#         info_fixation["rank_fixed"] = fixed_rank
#         info_fixation["rank_fixed_minus_pre"] = fixed_rank - rank_pre # only need this, beucase rank_pre == rank_post-1
#         # # Is this stroke already done

#         # if fixed_rank < rank_pre :
#         #     res["rank_fixed_status"] = "completed_before_last_stroke"
#         # elif (fixed_rank == rank_pre):
#         #     res["rank_fixed_status"] = "completed_by_last_stroke"
#         # elif (fixed_rank == rank_post):
#         #     res["rank_fixed_status"] = "start_in_next_stroke"
#         # elif fixed_rank > rank_post:
#         #     res["rank_fixed_status"] = "start_after_next_stroke"
#         # else:
#         #     print(fixed_rank, rank_pre, rank_post)
#         #     assert False

#         list_info_fixation.append(info_fixation)

#     # convert to dataframe
#     df_fixations = pd.DataFrame(list_info_fixation)

#     return df_fixations, info_gap

# def syntaxconcrete_dfmod_postprocess(D, dfthis_long):
#     """
#     MOdifies df, where each row is trialcode, index_gap, index_shape
#     """
#     # Add more data to dflong
#     from neuralmonkey.scripts.analy_syntax_good_gap_durations import syntaxconcrete_extract_more_info

#     assert "syntax_concrete" in dfthis_long
#     assert "index_gap" in dfthis_long

#     map_shape_to_chunk_rank_global, map_chunk_rank_global_to_shape = D.grammarparses_rules_shape_AnBmCk_get_map_shape_to_chunk_rank()

#     tmp = []
#     list_pre_shape = []
#     list_post_shape = []
#     list_n_remain_in_chunk = []
#     for _, row in dfthis_long.iterrows():
#         info = syntaxconcrete_extract_more_info(row["syntax_concrete"], row["index_gap"])
#         tmp.append(info["index_gap_is_chunk_switch"])

#         if info["pre_chunk_rank_global"]>-1:
#             # This is during drwaing
#             list_pre_shape.append(map_chunk_rank_global_to_shape[info["pre_chunk_rank_global"]])
#         else:
#             list_pre_shape.append("none")

#         list_post_shape.append(map_chunk_rank_global_to_shape[info["post_chunk_rank_global"]])

#         list_n_remain_in_chunk.append(info["n_remain_in_chunk"])

#     dfthis_long["index_gap_is_chunk_switch"] = tmp
#     dfthis_long["pre_shape"] = list_pre_shape
#     dfthis_long["post_shape"] = list_post_shape
#     dfthis_long["n_remain_in_chunk"] = list_n_remain_in_chunk # remaining in the post chunk.

#     # Also recode rank_fixed to be relative to current index_gap
#     dfthis_long["rankfixed_min_idxgap"] = dfthis_long["rank_fixed"] - dfthis_long["index_gap"] # 1 means that you are looking at the next shape

#     # Note if looking at something already done
#     dfthis_long["looking_at_already_drawn"] = dfthis_long["rankfixed_min_idxgap"]<=0
#     dfthis_long["looking_at_already_drawn_earlier"] = dfthis_long["rankfixed_min_idxgap"]<=1


def preprocess_dataset_behavior(D, ANALY_VER, SAVEDIR):
    """
    Helper to preprocess behavioral dataset to prepare for anlaysis of gaps, with respect to grammar variables.

    PARAMS:
    - ANALY_VER, string, either "rulesingle" or "rulesw"
    """

    if False:
        from neuralmonkey.metadat.analy.anova_params import params_getter_dataset_preprocess

        params = params_getter_dataset_preprocess(ANALY_VER, animal, DATE)

        from neuralmonkey.classes.snippets import dataset_extract_prune_general_dataset

        D = dataset_extract_prune_general_dataset(D,
                                                    list_superv_keep=params["list_superv_keep"],
                                                    preprocess_steps_append=params["preprocess_steps_append"],
                                                    remove_aborts=params["remove_aborts"],
                                                    list_superv_keep_full=params["list_superv_keep_full"],
                                                    )

        # Minimalist, should be fine?
        D.grammarparses_successbinary_score_wrapper()  

        # Extract chunk variables from Dataset
        for i in range(len(D.Dat)):
            D.grammarparses_taskclass_tokens_assign_chunk_state_each_stroke(i)

        # Also extract "syntax parse" e.g., (3,1,1) for A3B1C1.
        # Also called "taskcat_by_rule"
        D.grammarparses_classify_tasks_categorize_based_on_rule()
        print("These are the SYNTAX PARSES (i.e., 'taskcat_by_rule'):")
        print(D.Dat["taskcat_by_rule"].value_counts())

        ###################################################
        ################# SNTAX-RELATED VARIABLES...
        # And extract syntax_concrete column
        D.grammarparses_syntax_concrete_append_column()

        D.grammarparses_rules_epochs_superv_summarize_wrapper(PRINT=True)

        # For each token, assign a new key called "syntax role" -- good.
        D.grammarparses_syntax_role_append_to_tokens()

        # Epochsets, figure out if is same or diff motor beahvior (quick and dirty)
        D.grammarparses_syntax_epochset_quick_classify_same_diff_motor()

        # Define separate epochsets based on matching motor beh within each epoch_orig
        D.epochset_extract_matching_motor_wrapper()
    else:
        # Do what do for neural analy
        from neuralmonkey.metadat.analy.anova_params import dataset_apply_params

        DS = None
        animal = D.animals(force_single=True)[0]
        date = D.dates(force_single=True)[0]
        SKIP_PLOTS = True
        D, _, params = dataset_apply_params(D, None, ANALY_VER, animal, date,
                                            SKIP_PLOTS=SKIP_PLOTS) # prune it

    import os
    import matplotlib.pyplot as plt
    plot_savedir = f"{SAVEDIR}/preprocess"
    os.makedirs(plot_savedir, exist_ok=True)
    PLOT = True
    # PLOT = True
    dfgaps = D.grammarparses_chunk_transitions_gaps_extract_batch(PLOT=PLOT, plot_savedir=plot_savedir, also_get_response_time=True)
    plt.close("all")

    ### Prep the dataset
    if ANALY_VER=="rulesingle":
        list_n0 = []
        list_n1 = []
        list_n2 = []
        for sc in dfgaps["syntax_concrete"]:
            list_n0.append(sc[0])
            list_n1.append(sc[1])
            if len(sc)>2:
                list_n2.append(sc[2])
            else:
                list_n2.append(0)

        dfgaps["n0"] = list_n0
        dfgaps["n1"] = list_n1
        dfgaps["n2"] = list_n2
        dfgaps["n2_str"] = [str(n2) for n2 in dfgaps["n2"]]
        # save the index of the gaps
        dfgaps["idx_gap_01"] = dfgaps["n0"] - 1
        dfgaps["idx_gap_12"] = dfgaps["n0"] + dfgaps["n1"] - 1

        dfgaps["pre_chunk_rank_global"] = [x[0] for x in dfgaps["gap_chunk_rank_global"]]
        dfgaps["post_chunk_rank_global"] = [x[1] for x in dfgaps["gap_chunk_rank_global"]]

        dfgaps["diff_chunk_rank_global"] = dfgaps["post_chunk_rank_global"] - dfgaps["pre_chunk_rank_global"]
        dfgaps["diff_chunk_rank_global"].value_counts()
        dfgaps_nojumps = dfgaps[dfgaps["diff_chunk_rank_global"].isin([0, 1])].reset_index(drop=True)

        # dfgaps["ntot"] = [sum(sc[:-1]) for sc in dfgaps["syntax_concrete"]] # OLD, but
        dfgaps["ntot"] = [sum(sc) for sc in dfgaps["syntax_concrete"]] 

        dfgaps["nremain"] = [row["ntot"] - row["index_gap"] - 1 for _, row in dfgaps.iterrows()]
        
        assert all(dfgaps["nremain"] >= 0)
        
        from pythonlib.tools.pandastools import grouping_print_n_samples
        grouping_print_n_samples(dfgaps, ["syntax_concrete", "idx_gap_01", "idx_gap_12"])
        # Given syntax concrete and gap index, return all the params

        PRINT = False

        tmp =[]
        for _, row in dfgaps.iterrows():
            
            # Get 
            post_chunk_within_rank = row["gap_chunk_within_rank"][1]
            
            sc = row["syntax_concrete"]
            post_chunk_rank_global = row["post_chunk_rank_global"]
            n_in_chunk = sc[post_chunk_rank_global]

            n_remain_in_chunk = n_in_chunk - post_chunk_within_rank

            if PRINT:
                print(sc, " -- index_gap: ", row["index_gap"], " -- post_chunk_rank_global:", row["post_chunk_rank_global"], " -- post_chunk_within_rank:", post_chunk_within_rank, " -- n_in_chunk:", n_in_chunk, " -- n_remain_in_chunk:", n_remain_in_chunk)

            info = syntaxconcrete_extract_more_info(sc, row["index_gap"])
            assert info["post_chunk_rank_global"] == post_chunk_rank_global
            assert info["n_remain_in_chunk"] == n_remain_in_chunk
            assert info["n_in_chunk"] == n_in_chunk
            assert info["post_chunk_within_rank"] == post_chunk_within_rank

            tmp.append(n_remain_in_chunk)
        dfgaps["nremain_in_chunk"] = tmp

        assert all(dfgaps["nremain"] >= dfgaps["nremain_in_chunk"])

    return D, dfgaps, dfgaps_nojumps

def gapdur_plots_general(dfgaps, savedir):
    """
    General plots (can't remember what that means...)
    """
    
    # (1) Hue = syntax_concrete
    fig = sns.relplot(data=dfgaps, x="index_gap", y="gap_dur", hue="syntax_concrete", col="idx_gap_12", row="idx_gap_01", 
                    kind="line", errorbar="se")

    for i, idx_gap_01 in enumerate(range(dfgaps["idx_gap_01"].min(), dfgaps["idx_gap_01"].max()+1)):
        for j, idx_gap_12 in enumerate(range(dfgaps["idx_gap_12"].min(), dfgaps["idx_gap_12"].max()+1)):
            ax = fig.axes[i][j]

            try:
                ax = fig.axes[i][j]

                sns.lineplot(dfgaps, x="index_gap", y="gap_dur", ax=ax, color="k", alpha=0.5)

                ax.axvline(idx_gap_01, color="b", alpha=0.4)
                ax.axvline(idx_gap_12, color="r", alpha=0.4)
            except IndexError as err:
                pass
            except Exception as err:
                raise err
    savefig(fig, f"{savedir}/relplot-all-1.pdf")

    # (2) Hue = n2
    fig = sns.relplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2_str", col="idx_gap_12", row="idx_gap_01", kind="line", errorbar="se")
    # fig = sns.relplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2_str", col="idx_gap_12", row="idx_gap_01", kind="scatter", alpha=0.4)

    for i, idx_gap_01 in enumerate(range(dfgaps["idx_gap_01"].min(), dfgaps["idx_gap_01"].max()+1)):
        for j, idx_gap_12 in enumerate(range(dfgaps["idx_gap_12"].min(), dfgaps["idx_gap_12"].max()+1)):
            ax = fig.axes[i][j]

            try:
                ax = fig.axes[i][j]

                sns.lineplot(dfgaps, x="index_gap", y="gap_dur", ax=ax, color="k", alpha=0.5)

                ax.axvline(idx_gap_01, color="b", alpha=0.4)
                ax.axvline(idx_gap_12, color="r", alpha=0.4)
            except IndexError as err:
                pass
            except Exception as err:
                raise err
    savefig(fig, f"{savedir}/relplot-all-2.pdf")

    # (3) Scatterplot
    # fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2_str", col="idx_gap_12", row="idx_gap_01", kind="point", errorbar="se")
    fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2_str", col="idx_gap_12", row="idx_gap_01",  alpha=0.4)
    max_sequence_length = 5

    for i, idx_gap_01 in enumerate(range(dfgaps["idx_gap_01"].min(), dfgaps["idx_gap_01"].max()+1)):
        for j, idx_gap_12 in enumerate(range(dfgaps["idx_gap_12"].min(), dfgaps["idx_gap_12"].max()+1)):
            
            try:
                ax = fig.axes[i][j]
                sns.pointplot(dfgaps, x="index_gap", y="gap_dur", ax=ax, errorbar="ci", color=[0.8, 0.8, 0.8])
            except IndexError as err:
                continue
            except Exception as err:
                raise err
            
    savefig(fig, f"{savedir}/catplot-all-1.pdf")

    plt.close("all")


    if False:
        # This not good becuase (i) x axis is categorical, so will not be algined and (ii) is skewed in sublot org
        fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2", col="n1", row="n0", kind="point", errorbar="se")
        # fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="n2", col="n1", row="n0", alpha=0.4)

        max_sequence_length = 5

        for i in range(4):
            for j in range(4):
                ax = fig.axes[i][j]
                sns.pointplot(dfgaps, x="index_gap", y="gap_dur", ax=ax, errorbar="ci", color=[0.8, 0.8, 0.8])

                ax.axvline(i, color="b", alpha=0.3)
                if i + j < max_sequence_length:
                    ax.axvline(i+j, color="r", alpha=0.3)


    from pythonlib.tools.pandastools import plot_subplots_heatmap

    for norm_method in [None, "col_sub", "row_sub"]:
        fig, _ = plot_subplots_heatmap(dfgaps, "syntax_concrete", "index_gap", "gap_dur", None, norm_method=norm_method)
        savefig(fig, f"{savedir}/heatmap-norm={norm_method}.pdf")

    if False:
        plot_subplots_heatmap(dfgaps, "syntax_concrete", "index_gap", "gap_dur", "idx_gap_12", norm_method=None, share_zlim=True)
        plot_subplots_heatmap(dfgaps, "syntax_concrete", "index_gap", "gap_dur", "idx_gap_01", norm_method=None, share_zlim=True)

    plt.close("all")

def gapdur_plots_hypothesis_1(dfgaps, SAVEDIR):
    """
    """
    if False:
        fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="syntax_concrete", col="n0", kind="point", errorbar="se")
        fig = sns.catplot(data=dfgaps, x="index_gap", y="gap_dur", hue="syntax_concrete", col="n1", kind="point", errorbar="se")

    list_sc = sorted(dfgaps["syntax_concrete"].unique().tolist())
    list_slots = list(range(len(list_sc[0])-1))
    savedir = f"{SAVEDIR}/questions/1_slow_if_skip_slot"
    os.makedirs(savedir, exist_ok=True)

    # Given a syntax concrete, and a slot you want to keep fixed, get all other syntax concretes that have
    import pandas as pd
    import matplotlib.pyplot as plt

    slot_test_fixed_value = 0 # ie 0 means look for 0 in sc. e.g., (0, 2, 1) has 0 in first slot.
    for slot_test in list_slots:
        list_sc_test = [sc for sc in list_sc if sc[slot_test]==slot_test_fixed_value]

        for sc_test in list_sc_test:
            print(sc_test)

            # try fixing each of the other slots
            list_sc_other = []
            for slot_to_fix in list_slots:
                slot_to_fix_value = sc_test[slot_to_fix]
                _list_sc_other = [sc for sc in list_sc if sc[slot_test]!=slot_test_fixed_value and sc[slot_to_fix]==slot_to_fix_value]
                list_sc_other.extend(_list_sc_other)

                print("----")
                print("sc_test: ", sc_test)
                print("slot_test:", slot_test, " = ", slot_test_fixed_value)
                
                print("slot_to_fix:", slot_to_fix)
                print("sc_other:", _list_sc_other)

            dfgaps_test = dfgaps[dfgaps["syntax_concrete"] == sc_test].reset_index(drop=True)
            dfgaps_other = dfgaps[dfgaps["syntax_concrete"].isin(list_sc_other)].reset_index(drop=True)

            dfgaps_test["test_condition"] = True
            dfgaps_other["test_condition"] = False

            dfgaps_this = pd.concat([dfgaps_test, dfgaps_other]).reset_index(drop=True)


            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="syntax_concrete", col="test_condition", kind="point", errorbar="se")
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-1.pdf")
            
            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", errorbar="se")
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-2.pdf")
            
            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", alpha=0.25, jitter=True)
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-3.pdf")

            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", col="n0", errorbar="se")
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-4.pdf")

            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", col="n1", errorbar="se")
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-5.pdf")

            fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", col="n2", errorbar="se")
            savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-6.pdf")

            if False:
                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", row="n0", col="n1", errorbar="se")
                savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-7.pdf")

                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", row="n1", col="n2", errorbar="se")
                savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-8.pdf")

                fig = sns.catplot(data=dfgaps_this, x="index_gap", y="gap_dur", hue="test_condition", kind="point", row="n0", col="n2", errorbar="se")
                savefig(fig, f"{savedir}/slot_test={slot_test}-sc_test={sc_test}-catplot-9.pdf")

            plt.close("all")

def gapdur_plots_hypothesis_2(dfgaps, dfgaps_nojumps, SAVEDIR):
    """
    """
    # For each idx gap, compare when that is between chunks 0 and 1 vs when it is not (i.e., it is within chunk 0 or 1).
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper

    savedir = f"{SAVEDIR}/questions/2_slow_between_chunks"
    os.makedirs(savedir, exist_ok=True)

    dfgaps_str = stringify_values(dfgaps)
    dfgaps_nojumps_str = stringify_values(dfgaps_nojumps)
    
    # Keep only those (prechunkrank, and idx) that have a transition
    dfgaps_nojumps_str_clean, _ = extract_with_levels_of_conjunction_vars_helper(dfgaps_nojumps_str, 
                        "gap_chunk_rank_global", ["pre_chunk_rank_global", "index_gap"], lenient_allow_data_if_has_n_levels=2)
    
    # Plot
    fig = sns.catplot(data=dfgaps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen")
    savefig(fig, f"{savedir}/catplot-1.pdf")

    fig = sns.catplot(data=dfgaps_nojumps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen")
    savefig(fig, f"{savedir}/catplot-2.pdf")

    fig = sns.catplot(data=dfgaps_nojumps_str_clean, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen")
    savefig(fig, f"{savedir}/catplot-3.pdf")

    fig = sns.catplot(data=dfgaps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen", col="pre_chunk_rank_global")
    savefig(fig, f"{savedir}/catplot-4.pdf")

    fig = sns.catplot(data=dfgaps_nojumps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen", col="pre_chunk_rank_global")
    savefig(fig, f"{savedir}/catplot-5.pdf")

    fig = sns.catplot(data=dfgaps_nojumps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen", col="pre_chunk_rank_global")
    savefig(fig, f"{savedir}/catplot-6.pdf")

    fig = sns.catplot(data=dfgaps_nojumps_str_clean, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen", col="pre_chunk_rank_global")
    savefig(fig, f"{savedir}/catplot-7.pdf")

    if False:
        sns.catplot(data=dfgaps_nojumps_str, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen", col="post_chunk_rank_global")

def gapdur_plots_hypothesis_3(dfgaps, SAVEDIR):
    """
    """
    from pythonlib.tools.pandastools import stringify_values
    dfgaps_str = stringify_values(dfgaps)

    # For each idx gap, compare when that is between chunks 0 and 1 vs when it is not (i.e., it is within chunk 0 or 1).

    savedir = f"{SAVEDIR}/questions/3_slower_if_upcoming_chunk_longer"
    os.makedirs(savedir, exist_ok=True)

    # Key: need to dissociate it from the total number of items left in the sequence.

    # NOTE: ignore the first chunk, as there is too much time during planning for this to have a noticable effect.
    # New variable, the number of remaining items in sequence

    fig = sns.catplot(data=dfgaps_str, x="nremain_in_chunk", y="gap_dur", col="gap_chunk_rank_global", kind="point", 
                hue="nremain", errorbar="se")
    savefig(fig, f"{savedir}/catplot-1.pdf")

    fig = sns.catplot(data=dfgaps_str, x="nremain_in_chunk", y="gap_dur", col="gap_chunk_rank_global", kind="point", 
                hue="index_gap", errorbar="se")
    savefig(fig, f"{savedir}/catplot-2.pdf")

    fig = sns.catplot(data=dfgaps_str, x="nremain_in_chunk", y="gap_dur", col="gap_chunk_rank_global", kind="point", 
                row="nremain", hue="index_gap", errorbar="se")
    savefig(fig, f"{savedir}/catplot-3.pdf")    


def gapdur_mult_load_dates(animal):
    """
    Helper to load and concat and preprocess all dates for this animal.
    """
    from neuralmonkey.scripts.analy_euclidian_dist_pop_script_MULT import load_preprocess_get_dates
    from glob import glob
    from pythonlib.tools.pandastools import grouping_print_n_samples
    from pythonlib.tools.pandastools import aggregGeneral
    from pythonlib.tools.pandastools import stringify_values
    from pythonlib.tools.pandastools import extract_with_levels_of_conjunction_vars_helper
    from pythonlib.tools.pandastools import append_col_with_grp_index

    ### Load all dates
    list_date, _, _, _ = load_preprocess_get_dates(animal, "AnBmCk_general")
    list_dfgaps = []
    for _date in list_date:

        searchstr = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_*/dfgaps.pkl"
        # path = f"/lemur2/lucas/analyses/main/syntax_gap_durations/{animal}_{_date}_dirgrammardiego3b/dfgaps.pkl"
        list_path = glob(searchstr)

        if len(list_path)==1:
            path = list_path[0]
            dfgaps = pd.read_pickle(path)
            dfgaps["date"] = _date
            dfgaps["animal"] = animal
            list_dfgaps.append(dfgaps)
        elif len(list_path)==0:
            print("Skipping date, didn't find data:", _date)
        elif len(list_path)>2:
            print(list_path)
            assert False, "why?"
    DFGAPS = pd.concat(list_dfgaps).reset_index(drop=True)    

    # Fix a potential bug in defining of ntot
    DFGAPS["ntot"] = [sum(sc) for sc in DFGAPS["syntax_concrete"]] 
    DFGAPS["nremain"] = [row["ntot"] - row["index_gap"] - 1 for _, row in DFGAPS.iterrows()]
    assert all(DFGAPS["nremain"] >= DFGAPS["nremain_in_chunk"])

    # By convention, make "start button" chunk_rank_global = -1
    if False: # Just sanity check
        grouping_print_n_samples(DFGAPS, ["diff_chunk_rank_global", "gap_chunk_rank_global"])
    assert all(DFGAPS[DFGAPS["pre_chunk_rank_global"].isna()]["index_gap"]==-1), "assumes this"
    DFGAPS.loc[DFGAPS["pre_chunk_rank_global"].isna(), "pre_chunk_rank_global"] = -1
    DFGAPS["diff_chunk_rank_global"] = DFGAPS["post_chunk_rank_global"] - DFGAPS["pre_chunk_rank_global"] 
    DFGAPS["diff_chunk_rank_global"] = DFGAPS["diff_chunk_rank_global"].astype(int)
    assert sum(DFGAPS["diff_chunk_rank_global"]<0)==0, "assumes this for the next line"
    DFGAPS["is_chunk_transition"] = DFGAPS["diff_chunk_rank_global"]>0
    DFGAPS = append_col_with_grp_index(DFGAPS, ["index_gap", "nremain"], "idx_nrem")
    DFGAPS = append_col_with_grp_index(DFGAPS, ["date", "epoch"], "date_epoch")

    ### Diff kinds of agg
    # Agg, datapt = (index_gap, gap_chunk_rank, gap_shape, syntax_concrete, etc)
    DFGAPS_AGG_1 = aggregGeneral(DFGAPS, ["date", "animal", "index_gap", "gap_chunk_within_rank", 
                        "gap_shape", "syntax_concrete", "epoch"], ["gap_dur"],
                        nonnumercols=["gap_chunk_rank_global", "ep_sy_gcr", "epoch_syntax", "ntot", 
                                        "nremain", "nremain_in_chunk", "pre_chunk_rank_global", "post_chunk_rank_global", 
                                        "diff_chunk_rank_global"])

    # Agg, averaging over many things. Datapt = ()
    # NOTE: Decided to exclude gap_chunk_rank from grouping var. The reason is that the transition between BC should be
    # considered the same whether its in AABBCC or BBBBBCC. 
    DFGAPS_AGG_2 = aggregGeneral(DFGAPS_AGG_1, ["date", "animal", "index_gap", "gap_shape", "epoch"], ["gap_dur"],
                        nonnumercols=["gap_chunk_rank_global", "pre_chunk_rank_global", "post_chunk_rank_global", 
                                        "diff_chunk_rank_global"])

    DFGAPS_NOJUMPS = DFGAPS[DFGAPS["diff_chunk_rank_global"].isin([0, 1])].reset_index(drop=True)
    DFGAPS_NOJUMPS_STR = stringify_values(DFGAPS_NOJUMPS)

    DFGAPS_AGG_NOJUMPS = DFGAPS_AGG_2[DFGAPS_AGG_2["diff_chunk_rank_global"].isin([0, 1])].reset_index(drop=True)
    DFGAPS_AGG_NOJUMPS_STR = stringify_values(DFGAPS_AGG_NOJUMPS)

    # NOTE: each index_gap WILL have both (and only both) within and across
    # Keep only those (prechunkrank, and idx) that have a transition
    DFGAPS_NOJUMPS_STR_CLEAN, _ = extract_with_levels_of_conjunction_vars_helper(DFGAPS_NOJUMPS_STR, 
                        "diff_chunk_rank_global", ["animal", "date", "epoch", "pre_chunk_rank_global", "index_gap"], 
                        levels_var=[0,1])

    DFGAPS_AGG_NOJUMPS_STR_CLEAN, _ = extract_with_levels_of_conjunction_vars_helper(DFGAPS_AGG_NOJUMPS_STR, 
                        "gap_chunk_rank_global", ["animal", "date", "epoch", "pre_chunk_rank_global", "index_gap"], lenient_allow_data_if_has_n_levels=2)

    return DFGAPS, DFGAPS_AGG_1, DFGAPS_AGG_2, DFGAPS_NOJUMPS, DFGAPS_AGG_NOJUMPS, DFGAPS_NOJUMPS_STR_CLEAN, DFGAPS_AGG_NOJUMPS_STR_CLEAN

def gapdur_mult_plotsummary_2_transitions(DFGAPS_NOJUMPS_STR_CLEAN, DFGAPS_AGG_NOJUMPS_STR_CLEAN, savedir,
                                          scatter_x_diffcr=0, scatter_y_diffcr=1):
    """
    Transitions between chunks slower than within?
    """
    import seaborn as sns
    from pythonlib.tools.pandastools import append_col_with_grp_index
    from pythonlib.tools.pandastools import plot_45scatter_means_flexible_grouping

    ### (1) Plot each date
    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="boxen",
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-1-alldates.pdf")

    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", kind="boxen",
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-1b-alldates.pdf")

    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", kind="point", errorbar="se",
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-2-alldates.pdf")

    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", kind="point", errorbar="se",
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-2-alldates.pdf")

    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="gap_chunk_rank_global", jitter=True, alpha=0.3,
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-3-alldates.pdf")

    fig = sns.catplot(data=DFGAPS_NOJUMPS_STR_CLEAN, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", jitter=True, alpha=0.3,
                    col="date", col_wrap=6)
    savefig(fig, f"{savedir}/catplot-3-alldates.pdf")

    for df, suff in [
        (DFGAPS_NOJUMPS_STR_CLEAN, "data=raw"), 
        (DFGAPS_AGG_NOJUMPS_STR_CLEAN, "data=agg"),
        ]:

        ### (2) Summary catplots, different levels of dtail. datpt = (date, epoch, pre_crglobal, index)
        fig = sns.catplot(data=df, x="pre_chunk_rank_global", y="gap_dur", hue="diff_chunk_rank_global", kind="boxen")
        savefig(fig, f"{savedir}/catplot-{suff}-1.pdf")

        fig = sns.catplot(data=df, x="pre_chunk_rank_global", y="gap_dur", hue="diff_chunk_rank_global", kind="bar", errorbar="se")
        savefig(fig, f"{savedir}/catplot-{suff}-1b.pdf")

        fig = sns.catplot(data=df, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", kind="boxen")
        savefig(fig, f"{savedir}/catplot-{suff}-2.pdf")

        fig = sns.catplot(data=df, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", kind="point", 
                        col="pre_chunk_rank_global", errorbar="se")
        savefig(fig, f"{savedir}/catplot-{suff}-3.pdf")

        fig = sns.catplot(data=df, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", kind="boxen", 
                        col="pre_chunk_rank_global")
        savefig(fig, f"{savedir}/catplot-{suff}-3.pdf")

        fig = sns.catplot(data=df, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", col="pre_chunk_rank_global", jitter=True, alpha=0.5)
        savefig(fig, f"{savedir}/catplot-{suff}-4.pdf")

        ### (3) Scatterplots
        df = append_col_with_grp_index(df, ["date", "epoch"], "date_epoch")
        df = append_col_with_grp_index(df, ["date", "epoch", "pre_chunk_rank_global"], "date_epoch_precrg")
        df = append_col_with_grp_index(df, ["date", "epoch", "index_gap"], "date_epoch_idx")
        df = append_col_with_grp_index(df, ["date", "epoch", "pre_chunk_rank_global", "index_gap"], "date_epoch_precrg_idx")

        if suff=="data=raw":
            df = append_col_with_grp_index(df, ["date", "epoch", "pre_chunk_rank_global", "behseq_locs", "index_gap"], 
                                           "d_e_precr_lo_ig")
            
        if False: # Too slow
            # Show each individual datapoint
            fig = sns.catplot(data=df, x="index_gap", y="gap_dur", hue="diff_chunk_rank_global", 
                            row="pre_chunk_rank_global", col="date_epoch")
            
        _, fig = plot_45scatter_means_flexible_grouping(df, "diff_chunk_rank_global", 
                                            scatter_x_diffcr, scatter_y_diffcr, "pre_chunk_rank_global", "gap_dur", "date_epoch", False, shareaxes=True);
        savefig(fig, f"{savedir}/scatter-{suff}-1.pdf")

        _, fig = plot_45scatter_means_flexible_grouping(df, "diff_chunk_rank_global", 
                                            scatter_x_diffcr, scatter_y_diffcr, "pre_chunk_rank_global", "gap_dur", "date_epoch_idx", False, shareaxes=True);
        savefig(fig, f"{savedir}/scatter-{suff}-2.pdf")

        if suff=="data=raw":
            # This is as low as dtapt goes (ie lower and you don't have mult levels of diff_chunk_rank_global per datapt)
            _, fig = plot_45scatter_means_flexible_grouping(df, "diff_chunk_rank_global", 
                                                scatter_x_diffcr, scatter_y_diffcr, 
                                                "pre_chunk_rank_global", "gap_dur", "d_e_precr_lo_ig", 
                                                False, shareaxes=True, alpha=0.2);
            savefig(fig, f"{savedir}/scatter-{suff}-2b.pdf")

        _, fig = plot_45scatter_means_flexible_grouping(df, "diff_chunk_rank_global", 
                                            scatter_x_diffcr, scatter_y_diffcr, None, "gap_dur", "date_epoch_precrg", False, shareaxes=True);
        savefig(fig, f"{savedir}/scatter-{suff}-3.pdf")

        _, fig = plot_45scatter_means_flexible_grouping(df, "diff_chunk_rank_global", 
                                            scatter_x_diffcr, scatter_y_diffcr, None, "gap_dur", "date_epoch", False, shareaxes=True);
        savefig(fig, f"{savedir}/scatter-{suff}-4.pdf")
        
        plt.close("all")

def gapdur_mult_plotsummary_3_slower_if_upcoming_chunk_longer(DFGAPS, savedir, exclude_transitions_to_last_chunk=True,
                                                              plot_raw=True):
    """
    Transitions between chunks slower than within?

    PARAMS:
    - exclude_transitions_to_last_chunk, bool, if True, then only considers transitions such as the AB in AABBBC, 
    This ecludes BC (in that example) beucase there's no need to plan.
    """
    from pythonlib.tools.pandastools import aggregGeneral, extract_with_levels_of_conjunction_vars_helper, append_col_with_grp_index
    import matplotlib.pyplot as plt 
    import seaborn as sns

    ##############################
    # Gap duration longer if more items in following chunk? (ABBCC vs. ABBBC)

    ### Prepare
    # Control for (index_gap, n_remain, crglobalpair)
    # Vary by n_remain in chunk\
    print("start", len(DFGAPS))
    dfgaps, _ = extract_with_levels_of_conjunction_vars_helper(DFGAPS, 
                        "nremain_in_chunk", ["animal", "date", "epoch", "gap_chunk_rank_global", "index_gap", "nremain"], 
                        lenient_allow_data_if_has_n_levels=2)
    print("cleaned", len(dfgaps))
    if len(dfgaps)==0:
        return None

    # Prune to datapt
    # NOTE: Decided to include behseq_locs as if remove this then you lose a lot of trials (like 15x drop)
    dfgaps = aggregGeneral(dfgaps, ["date", "animal", "index_gap", "gap_chunk_within_rank", 
                        "gap_shape", "syntax_concrete", "epoch", "nremain", "nremain_in_chunk", "vars_others", 
                        "idx_nrem", "gap_chunk_within_rank", "gap_chunk_rank", "behseq_locs"], 
                        ["gap_dur"],
                        nonnumercols=["gap_chunk_rank_global", "ep_sy_gcr", "epoch_syntax", "ntot", 
                                        "pre_chunk_rank_global", "post_chunk_rank_global", 
                                        "diff_chunk_rank_global", "is_chunk_transition", "date_epoch"])
    print("agged", len(dfgaps))

    #################################
    ### Plot each date
    if plot_raw:
        # for hue in ["idx_nrem", "nremain"]:
        for hue in ["idx_nrem"]:
            fig = sns.catplot(data=dfgaps, x="nremain_in_chunk", y="gap_dur", row="gap_chunk_rank_global", kind="point", 
                        hue=hue, col="date_epoch", errorbar="se")
            savefig(fig, f"{savedir}/catplot-all-hue={hue}.pdf")
            plt.close("all")

    ### Collect all and plot 

    ### (1) First, clean up
    # Only keep cases at transition across chunks.
    # Exclude cases from start button
    dfgaps = dfgaps[dfgaps["pre_chunk_rank_global"]>-1].reset_index(drop=True)

    if exclude_transitions_to_last_chunk:
        # Also only if the upcoming chunks is not the lsat chunk (or else there's no planning).
        dfgaps = dfgaps[dfgaps["nremain"] > dfgaps["nremain_in_chunk"]].reset_index(drop=True)

    if len(dfgaps)==0:
        return None
        
    # Again, remove data that dont have enough variation
    dfgaps, _ = extract_with_levels_of_conjunction_vars_helper(dfgaps, 
                        "nremain_in_chunk", ["animal", "date", "epoch", "gap_chunk_rank_global", "index_gap", "nremain"], 
                        lenient_allow_data_if_has_n_levels=2)
    if dfgaps is None:
        return None

    # - control for (idx and nremain)
    dfgaps = append_col_with_grp_index(dfgaps, ["date", "epoch", "idx_nrem", "gap_chunk_rank_global"], "datapt")

    ### (2) Plot final data that goes into regression
    fig = sns.catplot(data=dfgaps, x="nremain_in_chunk", y="gap_dur", hue="datapt", 
                    col="is_chunk_transition", kind="point", errorbar="se")
    savefig(fig, f"{savedir}/catplot-final_before_regress.pdf")

    # Plot again the split plots, this time the final data before goes into regression
    # for hue in ["idx_nrem", "nremain"]:
    for hue in ["idx_nrem"]:
        fig = sns.catplot(data=dfgaps, x="nremain_in_chunk", y="gap_dur", row="gap_chunk_rank_global", kind="point", 
                    hue=hue, col="date_epoch", errorbar="se")
        savefig(fig, f"{savedir}/catplot-all-hue={hue}-final_before_regress.pdf")
        plt.close("all")

    ### (3) Linear mixed effects
    # For each (datapt) do fit regression line (and collect other useful information)
    # Use linear mixed effects model.
    import statsmodels.formula.api as smf
    from pythonlib.stats.lme import lme_summary_extract_plot
    y = "gap_dur"
    x = "nremain_in_chunk"
    formula = f"{y} ~ {x}"
    rand_grp = "datapt"

    list_summary = []
    for is_chunk_transition in [False, True]:
        df = dfgaps[dfgaps["is_chunk_transition"] == is_chunk_transition].reset_index(drop=True)
        if len(df)>0:
            md = smf.mixedlm(formula, df, groups=df[rand_grp])
            result = md.fit()

            summary_df, fig = lme_summary_extract_plot(result)
            summary_df["is_chunk_transition"] = is_chunk_transition
            list_summary.append(summary_df)

            savefig(fig, f"{savedir}/lme-coefficients-is_chunk_transition={is_chunk_transition}.pdf")

    dfsummary = pd.concat(list_summary)

    # 3. Plot coefficients with error bars
    _dfsummary = dfsummary[dfsummary.index==x]
    var_to_plot = "is_chunk_transition"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        _dfsummary[var_to_plot],
        _dfsummary["coef"],
        yerr=[_dfsummary["coef"] - _dfsummary["ci_lower"], 
            _dfsummary["ci_upper"] - _dfsummary["coef"]],
        fmt="o",
        capsize=5,
        color="black"
    )
    # Add horizontal line at 0
    ax.axhline(0, color="gray", linestyle="--")

    xlabs = _dfsummary[var_to_plot].tolist()
    xticks = range(len(xlabs))
    ax.set_xticks(xticks, xlabs)
    ax.set_xlabel(var_to_plot)
    ax.set_ylabel(f"{y} VS {x}")

    savefig(fig, f"{savedir}/lme-coefficients-combined.pdf")

    plt.close("all")

if __name__=="__main__":

    # NOTE: This IS up to date (not notebook)

    animal = sys.argv[1]
    date = int(sys.argv[2])
    ANALY_VER = "rulesingle"

    from pythonlib.dataset.dataset import load_dataset_daily_helper

    ##### Try loading a single dataset
    D = load_dataset_daily_helper(animal, date)

    SAVEDIR = D.make_savedir_for_analysis_figures_BETTER("syntax_gap_durations")
    import seaborn as sns
    from pythonlib.tools.plottools import savefig
    import os    

    ### Preprocess
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import preprocess_dataset_behavior
    # Preprocess
    ANALY_VER = "rulesingle"
    D, dfgaps, dfgaps_nojumps = preprocess_dataset_behavior(D, ANALY_VER, SAVEDIR)

    # Save processed data
    import pandas as pd
    pd.to_pickle(dfgaps, f"{SAVEDIR}/dfgaps.pkl")
    # D.save(SAVEDIR)

    ##### General plots
    savedir = f"{SAVEDIR}/general"
    os.makedirs(savedir, exist_ok=True)
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import gapdur_plots_general
    gapdur_plots_general(dfgaps, savedir)

    ### Testing specific hypotheses
    
    ##### (1) Much slower if there is 0 in a chunk.
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import gapdur_plots_hypothesis_1
    gapdur_plots_hypothesis_1(dfgaps, SAVEDIR)

    ##### (2) Slower for gaps between chunks
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import gapdur_plots_hypothesis_2
    gapdur_plots_hypothesis_2(dfgaps, dfgaps_nojumps, SAVEDIR)

    ##### (3) Gap duration predicts length of upcoming chunk
    from neuralmonkey.scripts.analy_syntax_good_gap_durations import gapdur_plots_hypothesis_3
    gapdur_plots_hypothesis_3(dfgaps, SAVEDIR)