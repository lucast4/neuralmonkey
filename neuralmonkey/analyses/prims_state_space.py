""" Static represnetation of prims (clean behavior, e..,g, prims in grid), in population 
state space

See notebook: 220713_prims_state_space

"""

import numpy as np
import matplotlib.pyplot as plt
from ..population.dimreduction import plotStateSpace
import random

def plot_results_state_space_by_group(DatDict, ParamsDict, ResDict, 
    plot_dims=[0,1], COLOR_BY = "shape", TEXT_LABEL_BY="group",
    overlay_strokes="beh"):
    
    DatGrp = DatDict["DatGrp"]
    DATAPLOT_GROUPING_VARS = ParamsDict["DATAPLOT_GROUPING_VARS"]
    groupdict = DatDict["groupdict"]
    DS = DatDict["DS"]

    from pythonlib.tools.plottools import move_legend_outside_axes

    LIST_DATAVER = [
        "X_timetrialmean", # (chans, )
        "X_timemean"] # (chans, trials)

    fig, axes = plt.subplots(len(LIST_DATAVER),1, figsize=(15,len(LIST_DATAVER)*10))
    for ax, DATAVER in zip(axes.flatten(), LIST_DATAVER):

        list_x = []
        list_grp_for_color = []
        list_firstpointinds = []
        list_grp_for_text = []
        list_grp_alignedto_firstpoint = []
        for dat in DatGrp:
            x = dat[DATAVER]
            if len(x.shape)==1:
                x = x[:, None]

            def _get_group_(by):
                if by=="group":
                    grp = dat["group"]
                elif by=="shape":
                    indthis = DATAPLOT_GROUPING_VARS.index("shape_oriented")
                    grp = dat["group"][indthis]
                elif by=="location":
                    indthis = DATAPLOT_GROUPING_VARS.index("gridloc")
                    grp = dat["group"][indthis]
                elif by=="gridsize":
                    indthis = DATAPLOT_GROUPING_VARS.index("gridsize")
                    grp = dat["group"][indthis]
                else:
                    print(by)
                    assert False
                return grp

            grp_for_color = _get_group_(COLOR_BY)
            grp_for_text = _get_group_(TEXT_LABEL_BY)
            grp = _get_group_("group")

            list_x.append(x)
            list_firstpointinds.append(len(list_grp_for_color)) # useful for plotting
            list_grp_alignedto_firstpoint.append(grp)
            list_grp_for_color.extend([grp_for_color for _ in range(x.shape[1])])
            list_grp_for_text.append(grp_for_text);

        # Concatenate to (nchans, ndatapts)
        X = np.concatenate(list_x, axis=1)
        assert len(list_grp_for_color)==X.shape[1]

        # Plot
        x1, x2 = plotStateSpace(X, dim1=plot_dims, ax=ax, color=list_grp_for_color)
        move_legend_outside_axes(ax)

        # Put text on first pts.
        for i, ind in enumerate(list_firstpointinds):
            ax.text(x1[ind], x2[ind], list_grp_for_text[i])

        # Plot strokes at each pt
        # PLOT DRAWINGS, with their location in state space
        # get stroke inds
        if overlay_strokes is not None:
            for i, ind in enumerate(list_firstpointinds):
                loc = [x1[ind], x2[ind]]
                grp = list_grp_alignedto_firstpoint[i]

                list_indstroke = groupdict[grp]
                indstroke = random.choice(list_indstroke) # take random one
                strok = DS.extract_strokes(inds=[indstroke], ver_behtask=overlay_strokes)[0][:,:2]

                # center, to strok onset, then rescale
                pt0 = strok[0]
                strok = strok-pt0 # center to onset

                strok = strok/50 # rescale

                XLIM = ax.get_xlim()
                YLIM = ax.get_ylim()
                xdiff = XLIM[1] - XLIM[0]
                ydiff = YLIM[1] - YLIM[0]
                strok[:,0] =  strok[:,0] * xdiff/ydiff # so it looks correct even if aspect is not 1
                strok = strok + loc # move to new locatin

                # plot at that location, starting from strok onset
                ax.plot(strok[:,0], strok[:,1], '-k')
                ax.plot(strok[0,0], strok[0,1], 'ok')

                # DS.plot_single_strok(strok, ax) # dont use this, since it forces aspect=1

    return fig
