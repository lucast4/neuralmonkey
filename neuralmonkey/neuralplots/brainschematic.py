"""
Tools for plotting data onto brain schematic
See snippets.modulation_plot_heatmaps_brain_schematic (move stuff here)
"""

import numpy as np
import matplotlib.pyplot as plt
from neuralmonkey.classes.session import _REGIONS_IN_ORDER as REGIONS_IN_ORDER
from neuralmonkey.classes.session import _REGIONS_IN_ORDER_COMBINED as REGIONS_IN_ORDER_COMBINED
from pythonlib.globals import PATH_NEURALMONKEY

#
# def mapper_bregion_to_index():
#     """ Return dict mapping bregion to index useful for
#     consistent plotting order (hierarhcial)
#     """
#     from neuralmonkey.classes.session import

def datamod_reorder_by_bregion_get_mapper():
    """
    Rreturn dict mapping from bregion to rank, useful for consistent plotting.
    """
    map_region_to_index = {region:i for i, region in enumerate(REGIONS_IN_ORDER)}
    # Also include combined regions

    for i, region in enumerate(REGIONS_IN_ORDER_COMBINED):
        map_region_to_index[region] = i
        
    return map_region_to_index

def datamod_reorder_by_bregion(df, col="bregion"):
    """ reorder rows of dataframe based on bregion, from top to bottom
    RETURNS:
        - df, sorted. DOes not modify input
    """
    if col not in df.columns:
        if "region" in df.columns:
            col = "region"
        else:
            print(df.columns)
            assert False, "whichc olumn holds regions?"
    map_region_to_index = datamod_reorder_by_bregion_get_mapper()
    def F(x):
        return [map_region_to_index[xx] for xx in x] # list of ints
    return df.sort_values(by=col, key=lambda x:F(x)).reset_index(drop=True)

def mapper_combinedbregion_to_meanlocation():
    """
    Map from combined bregion (e..g, "M1") to np array (x,y) coords (2,) shape.
    Note that bregion has no num prefix.
    :return:
    """
    from  neuralmonkey.classes.session import MAP_COMBINED_REGION_TO_REGION

    # 1) MAke map from region (no prefix num) to location
    mapper = mapper_bregion_to_location()
    map_bregion_location = {}
    for k, v in mapper.items():
        map_bregion_location[k[3:]] = v

    # 2) Make map from conbined region to mean loc.
    map_combinedbr_locmean = {}
    for combined_region in MAP_COMBINED_REGION_TO_REGION.keys():
        vals = [map_bregion_location[reg] for reg in MAP_COMBINED_REGION_TO_REGION[combined_region]]
        import numpy as np
        vals = np.stack(vals) # (nreg, 2)
        coord_mean = np.mean(vals, axis=0)
        map_combinedbr_locmean[combined_region] = coord_mean

    return map_combinedbr_locmean

def mapper_bregion_to_location():
    map_bregion_to_location = {}
    map_bregion_to_location["00_M1_m"] = [0, 1.3]
    map_bregion_to_location["01_M1_l"] = [1, 2]
    map_bregion_to_location["02_PMv_l"] = [4, 5.3]
    map_bregion_to_location["03_PMv_m"] = [3.5, 3.3]
    map_bregion_to_location["04_PMd_p"] = [3.3, 1.6]
    map_bregion_to_location["05_PMd_a"] = [5, 1.85]
    map_bregion_to_location["06_SMA_p"] = [-.1, 0.2]
    map_bregion_to_location["07_SMA_a"] = [1.4, 0.3]
    map_bregion_to_location["08_dlPFC_p"] = [7.2, 2.8]
    map_bregion_to_location["09_dlPFC_a"] = [9, 3]
    map_bregion_to_location["10_vlPFC_p"] = [5.8, 5]
    map_bregion_to_location["11_vlPFC_a"] = [8.5, 4]
    map_bregion_to_location["12_preSMA_p"] = [3.2, 0.4]
    map_bregion_to_location["13_preSMA_a"] = [4.5, 0.6]
    map_bregion_to_location["14_FP_p"] = [11, 3.9]
    map_bregion_to_location["15_FP_a"] = [12.5, 4.3]
    return map_bregion_to_location

def regions_get_ordered_by_x(ver="hand", prune_index=True, combined_regions=False):
    """ Get list of regions (e.g, M1_m) orderd by x location, either
    by hand (defyault) or by coords
    - prune_index, if True, then M1_m, else 00_M1_m
    """

    if ver=="hand":
        if combined_regions:
            regions_ordered_by_x = [
                 'M1',
                 'SMA',
                 'PMd',
                 'PMv',
                 'preSMA',
                 'vlPFC',
                 'dlPFC',
                 'FP']    
            prune_index = False # "combined regions dont havew indices"
        else:
            regions_ordered_by_x = [
                 '00_M1_m',
                 '01_M1_l',
                 '06_SMA_p',
                 '07_SMA_a',
                 '04_PMd_p',
                 '03_PMv_m',
                 '02_PMv_l',
                 '05_PMd_a',
                 '12_preSMA_p',
                 '13_preSMA_a',
                 '10_vlPFC_p',
                 '11_vlPFC_a',
                 '08_dlPFC_p',
                 '09_dlPFC_a',
                 '14_FP_p',
                 '15_FP_a']    
            assert prune_index==True, "indices are incorrect"
    else:
        assert combined_regions==False, "not coded..."

        # use actual coords
        # Order regions based on xy location
        map_bregion_to_location = mapper_bregion_to_location()
        # from neuralmonkey.neuralplots.brainschematic import map_bregion_to_location

        tmp = []
        for reg, coords in map_bregion_to_location.items():
            tmp.append([reg, coords[0], coords[1]])
        tmpsorted = sorted(tmp, key=lambda x:x[1])
        regions_ordered_by_x = [x[0] for x in tmpsorted]

    # remove indices
    if prune_index:
        regions_ordered_by_x = [x[3:] for x in regions_ordered_by_x]

    if False:
        print("regions_ordered_by_x:", regions_ordered_by_x)     
    
    return regions_ordered_by_x

def plot_df_from_wideform(dfthis_agg_2d, savedir=None, col1_name="bregion", # just for saving
                          subplot_var=None, valname=None, savesuffix="", # Just for saving
                          norm_method=None,
                          diverge=False, DEBUG=False,
                          zlims=(None, None)):
    """ Plot df that is already wide form, with rows as bregion and columns as each
    subplot. Inputed vars are for labeling purposes, not for tforming dframe.
    Zlim will be matched across all subplots.
    """
    from pythonlib.tools.snstools import heatmap

    map_bregion_to_location = mapper_bregion_to_location()
    map_combinedbr_to_location = mapper_combinedbregion_to_meanlocation()

    # Plot heatmap
    annotate_heatmap = False
    # ZLIMS = [None, None]
    fig_hist, ax_hist, rgba_values = heatmap(dfthis_agg_2d, None, annotate_heatmap, zlims,
                                   diverge=diverge, norm_method=norm_method)
    ax_hist.set_xlabel(subplot_var)
    ax_hist.set_ylabel(col1_name)

    # 1) DEFINE COORDS FOR EACH REGION
    # (horiz from left, vert from top)
    xmult = 33
    ymult = 50
    # xoffset = 230 # if use entire image
    xoffset = 100 # if clip
    yoffset = 30
    for k, v in map_bregion_to_location.items():
        map_bregion_to_location[k] = [xoffset + xmult*v[0], yoffset + ymult*v[1]]
    rad = (xmult + ymult)/4

    for k, v in map_combinedbr_to_location.items():
        map_combinedbr_to_location[k] = [xoffset + xmult*v[0], yoffset + ymult*v[1]]
    rad = (xmult + ymult)/4

    def _find_region_location(region):
        if region in map_bregion_to_location.keys():
            return map_bregion_to_location[region]
        elif region in map_combinedbr_to_location.keys():
            return map_combinedbr_to_location[region]
        else:
            # assume region is like "M1_m" and keys are like "00_M1_m"
            v = None
            FOUND = False
            for k, v in map_bregion_to_location.items():
                if region == k[3:]:
                    loc = v

                    # print(region, k)
                    # print(map_bregion_to_location)
                    assert FOUND==False
                    FOUND = True
            assert FOUND==True
            assert loc is not None, "didnt find!"
            return loc

    map_bregion_to_rowindex = {}
    list_regions = dfthis_agg_2d.index.tolist()
    for i, region in enumerate(list_regions):
        map_bregion_to_rowindex[region] = i

    if DEBUG:
        print("\nindex -- region")
        for k, v in map_bregion_to_rowindex.items():
            print(v, k)

    map_event_to_colindex = {}
    list_events = dfthis_agg_2d.columns.tolist()
    for i, event in enumerate(list_events):
        map_event_to_colindex[event] = i
    if DEBUG:
        print("\nindex -- event")
        for event, i in map_event_to_colindex.items():
            print(i, ' -- ' , event)


    # PLOT
    ncols = len(list_events)
    nrows = int(np.ceil(len(list_events)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), squeeze=False)

    for i, (ax, event) in enumerate(zip(axes.flatten(), list_events)):

        ax.set_title(event)

        # 1) load a cartoon image of brain
    #     image_name = "/home/lucast4/Downloads/thumbnail_image001.png"
        # image_name = "/gorilla3/Dropbox/SCIENCE/FREIWALD_LAB/DATA/brain_drawing_template.jpg"
        image_name = f"{PATH_NEURALMONKEY}/neuralplots/images/brain_drawing_template.jpg"
        im = plt.imread(image_name)
        im = im[:330, 130:]
        ax.imshow(im)

    #     if i==1:
    #         assert False
        for bregion in list_regions:
            irow = map_bregion_to_rowindex[bregion]
            icol = map_event_to_colindex[event]

            col = rgba_values[irow, icol]
            cen = _find_region_location(bregion)

            # 2) each area has a "blob", a circle on this image

            # print(bregion, irow, icol, col, cen)
            c = plt.Circle(cen, rad, color=col, clip_on=False)
            ax.add_patch(c)

    # Remove axis ticks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # SAVE FIG
    if savedir:
        path = f"{savedir}/brainschem-{subplot_var}-{valname}-norm_{norm_method}-{savesuffix}.pdf"
        path_hist = f"{savedir}/brainschem-{subplot_var}-{valname}-norm_{norm_method}-{savesuffix}-HIST.pdf"
        print("Saving to: ", path)
        fig.savefig(path)
        fig_hist.savefig(path_hist)

    return fig, axes

def plot_df_from_longform(df, valname, subplot_var, savedir = None, savesuffix="",
                          DEBUG=False, diverge=False, norm_method=None):
    """
    GOOD - plot given df with values in column (valname) and multiple subplots
    (subplot_var). Must be longform. Here does aggregation by taking the mean,
    if required.
    PARAMS:
    - df, long-form dataframe with N rows per brain region, if N>1, then will
    agg by taking mean.
    - valname, str, column name in df that maps to color (e.g., "val")
    - subplot_var, which variables levels to separate into subplots. Leave None if doesnt
    have.
    """
    from pythonlib.tools.pandastools import convert_to_2d_dataframe
    map_bregion_to_location = mapper_bregion_to_location()

    if "region" in df.columns:
        REGION = "region"
    elif "bregion" in df.columns:
        REGION = "bregion"
    else:
        print(df.columns)
        assert False, "must be  a column called region or bregion"

    # Convert to a 2d dataframe with histrogram and rgb values
    norm_method = None
    annotate_heatmap = False
    ZLIMS = [None, None]

    dfthis_agg_2d, fig_hist, ax_hist, rgba_values = convert_to_2d_dataframe(df, 
                                                 REGION, subplot_var, True, agg_method="mean", 
                                                 val_name=valname, 
                                                 norm_method=norm_method,
                                                 annotate_heatmap=annotate_heatmap,
                                                zlims = ZLIMS, diverge=diverge
                                                )

    fig, axes = plot_df_from_wideform(dfthis_agg_2d, savedir, REGION, # just for saving
                          subplot_var, valname, savesuffix, # Just for saving
                          norm_method,
                          diverge, DEBUG)

    return fig, axes

    # "
    # # 1) DEFINE COORDS FOR EACH REGION
    # # (horiz from left, vert from top)
    # xmult = 33
    # ymult = 50
    # # xoffset = 230 # if use entire image
    # xoffset = 100 # if clip
    # yoffset = 30
    # for k, v in map_bregion_to_location.items():
    #     map_bregion_to_location[k] = [xoffset + xmult*v[0], yoffset + ymult*v[1]]
    # rad = (xmult + ymult)/4
    #
    #
    # def _find_region_location(region):
    #     if region in map_bregion_to_location.keys():
    #         return map_bregion_to_location[region]
    #     else:
    #         # assume region is like "M1_m" and keys are like "00_M1_m"
    #         v = None
    #         FOUND = False
    #         for k, v in map_bregion_to_location.items():
    #             if region == k[3:]:
    #                 loc = v
    #
    #                 # print(region, k)
    #                 # print(map_bregion_to_location)
    #                 assert FOUND==False
    #                 FOUND = True
    #         assert FOUND==True
    #         assert loc is not None, "didnt find!"
    #         return loc
    #
    # map_bregion_to_rowindex = {}
    # list_regions = dfthis_agg_2d.index.tolist()
    # for i, region in enumerate(list_regions):
    #     map_bregion_to_rowindex[region] = i
    #
    # if DEBUG:
    #     print("\nindex -- region")
    #     for k, v in map_bregion_to_rowindex.items():
    #         print(v, k)
    #
    # map_event_to_colindex = {}
    # list_events = dfthis_agg_2d.columns.tolist()
    # for i, event in enumerate(list_events):
    #     map_event_to_colindex[event] = i
    # if DEBUG:
    #     print("\nindex -- event")
    #     for event, i in map_event_to_colindex.items():
    #         print(i, ' -- ' , event)
    #
    #
    # # PLOT:
    # ncols = len(list_events)
    # nrows = int(np.ceil(len(list_events)/ncols))
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows), squeeze=False)
    #
    # for i, (ax, event) in enumerate(zip(axes.flatten(), list_events)):
    #
    #     ax.set_title(event)
    #
    #     # 1) load a cartoon image of brain
    # #     image_name = "/home/lucast4/Downloads/thumbnail_image001.png"
    #     # image_name = "/gorilla3/Dropbox/SCIENCE/FREIWALD_LAB/DATA/brain_drawing_template.jpg"
    #     image_name = f"{PATH_NEURALMONKEY}/neuralplots/images/brain_drawing_template.jpg"
    #     im = plt.imread(image_name)
    #     im = im[:330, 130:]
    #     ax.imshow(im)
    #
    # #     if i==1:
    # #         assert False
    #     for bregion in list_regions:
    #         irow = map_bregion_to_rowindex[bregion]
    #         icol = map_event_to_colindex[event]
    #
    #         col = rgba_values[irow, icol]
    #         cen = _find_region_location(bregion)
    #
    #         # 2) each area has a "blob", a circle on this image
    #
    #         # print(bregion, irow, icol, col, cen)
    #         c = plt.Circle(cen, rad, color=col, clip_on=False)
    #         ax.add_patch(c)
    #
    # # Remove axis ticks
    # for ax in axes.flatten():
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    #
    # # SAVE FIG
    # if savedir:
    #     path = f"{savedir}/brainschem-{subplot_var}-{valname}-{savesuffix}.pdf"
    #     path_hist = f"{savedir}/brainschem-{subplot_var}-{valname}-{savesuffix}-HIST.pdf"
    #     print("Saving to: ", path)
    #     fig.savefig(path)
    #     fig_hist.savefig(path_hist)
    #
    # return fig, axes


def plot_scalar_values(regions, values, diverge=False):
    """
    Helper to take scalar values, each corresponding to a region, and plot those values on a brain
    PARAMS;
    - regions, list of str, each region can p[resem p]
    - values, list of mean value for each region. 
    """
    import pandas as pd

    # map_bregion_to_location = {}
    # map_bregion_to_location["M1_m"] = [0, 1.3]
    # map_bregion_to_location["M1_l"] = [1, 2]
    # map_bregion_to_location["PMv_l"] = [4, 5.3]
    # map_bregion_to_location["PMv_m"] = [3.5, 3.3]
    # map_bregion_to_location["PMd_p"] = [3.3, 1.6]
    # map_bregion_to_location["PMd_a"] = [5, 1.85]
    # map_bregion_to_location["dlPFC_p"] = [7.2, 2.8]
    # map_bregion_to_location["dlPFC_a"] = [9, 3]
    # map_bregion_to_location["vlPFC_p"] = [5.8, 5]
    # map_bregion_to_location["vlPFC_a"] = [8.5, 4]
    # map_bregion_to_location["FP_p"] = [11, 3.9]
    # map_bregion_to_location["FP_a"] = [12.5, 4.3]
    # map_bregion_to_location["SMA_p"] = [-.1, 0.2]
    # map_bregion_to_location["SMA_a"] = [1.4, 0.3]
    # map_bregion_to_location["preSMA_p"] = [3.2, 0.4]
    # map_bregion_to_location["preSMA_a"] = [4.5, 0.6]
    # xmult = 33
    # ymult = 50
    # # xoffset = 230 # if use entire image
    # xoffset = 100 # if clip
    # yoffset = 30
    # for k, v in map_bregion_to_location.items():
    #     map_bregion_to_location[k] = [xoffset + xmult*v[0], yoffset + ymult*v[1]]
    # rad = (xmult + ymult)/4

    ###############################
    # 1) values and regions, collect in a pandas dataframe
    df = pd.DataFrame({"region":regions, "value":values})

    return plot_df_from_longform(df, "value", subplot_var=None, savedir = None, savesuffix="",
                                 DEBUG=False, diverge=diverge)

    # map_bregion_to_value = {}
    # for reg, val in zip(regions, rgba_values):
    #     map_bregion_to_value[reg] = val


    # cmap = sns.color_palette("rocket", as_cmap=True)
    # # Return the colors
    # from matplotlib.colors import Normalize
    # # Normalize data
    # norm = Normalize(vmin=-0.3, vmax=0.3)
    # rgba_values = cmap(norm(scores))
    # rgba_values

    # # Overlay on brain
    # fig, ax = plt.subplots()

    # list_regions = regions

    # # 1) get heatmap

    # image_name = "/gorilla3/Dropbox/SCIENCE/FREIWALD_LAB/DATA/brain_drawing_template.jpg"
    # im = plt.imread(image_name)
    # im = im[:330, 130:]
    # ax.imshow(im)

    # #     if i==1:
    # #         assert False
    # for bregion in list_regions:
    # #     irow = map_bregion_to_rowindex[bregion]
    # #     icol = map_event_to_colindex[event]

    #     col = map_bregion_to_value[bregion]
    #     cen = map_bregion_to_location[bregion]

    #     # 2) each area has a "blob", a circle on this image
    #     c = plt.Circle(cen, rad, color=col, clip_on=False)
    #     ax.add_patch(c)
