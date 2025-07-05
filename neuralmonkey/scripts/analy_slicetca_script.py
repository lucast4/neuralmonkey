import numpy as np
import os
from pythonlib.tools.plottools import savefig
import pandas as pd
import matplotlib.pyplot as plt
from neuralmonkey.classes.population_mult import load_handsaved_wrapper
from neuralmonkey.classes.population_mult import dfallpa_preprocess_fr_normalization
import numpy as np
from slicetca.plotting import grid
import slicetca
import torch
from matplotlib import pyplot as plt
import sys

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def plot_components(model, pa, var_color, savedir):
    """
    Plot components using built-in funciton in slicetca
    """
    from neuralmonkey.analyses.state_space_good import _trajgood_make_colors_discrete_var
    from pythonlib.tools.plottools import legend_add_manual

    times = pa.Times
    labels = pa.Xlabels["trials"][var_color]

    _map_lev_to_color, color_type, colors = _trajgood_make_colors_discrete_var(labels)
    colors_arr = np.stack(colors)

    # sort by trials
    trial_idx = np.argsort(labels).tolist()

    # we sort the neurons of the trial slices according to their peak activity in the first slice.
    if False: # keep unsorted, so they match up across events:
        neuron_sorting_peak_time = np.argsort(np.argmax(components[0][1][0], axis=1))
    else:
        neuron_sorting_peak_time = None

    # call plotting function, indicating index for sorting trials and colors for different angles as well as time
    fig, axes = slicetca.plot(model,
                variables=('trial', 'neuron', 'time'),
                colors=(colors_arr[trial_idx], None, None), # we only want the trials to be colored
                ticks=(None, None, np.linspace(0, len(times),3)), # we only want to modify the time ticks
                tick_labels=(None, None, np.linspace(times[0],times[-1],3)),
                sorting_indices=(trial_idx, neuron_sorting_peak_time, None),
                quantile=0.99, return_fig=True)

    path = f"{savedir}/components-colorby={var_color}.pdf"
    savefig(fig, path)    

    # - legend for the color
    fig, ax = plt.subplots()
    legend_add_manual(ax, _map_lev_to_color.keys(), _map_lev_to_color.values())
    ax.set_title(var_color)
    savefig(fig, f"{savedir}/legend_colorby={var_color}.pdf")


def run(pa, savedir, TWIND_ANALY, list_var_color):
    """
    Entire pipeline to run and do plots
    PARAMS:
    - list_var_color, list of str, variabvles to use for coloring results
    """

    ### Preprocess PA
    # Normalize to Fr range from (0, ~2)
    if False:
        mins = np.min(np.min(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
        maxs = np.max(np.max(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
        pa.X = (pa.X - mins)/(maxs-mins)
    else:
        MAX = 2*np.max(np.std(pa.X.reshape((pa.X.shape[0], pa.X.shape[1] * pa.X.shape[2])), axis=1))
        mins = np.min(np.min(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
        pa.X = (pa.X - mins)/(MAX)        
        # pa.X = pa.X/pa.X.std() # to put in range
        
    pa = pa.slice_by_dim_values_wrapper("times", TWIND_ANALY)

    # Plot exmaple of teh data before it goes into tca
    fig, ax = plt.subplots()
    pa.plotNeurHeat(0, ax=ax)
    savefig(fig, f"{savedir}/example_data_before_tca_single_trial.pdf")

    # Get data in correct foramt    
    data_np = np.transpose(pa.X, (1, 0, 2)) # your_data is a numpy array of shape (trials, neurons, time).
    data_torch = torch.tensor(data_np, device=device, dtype=torch.float32)
    
    ############
    train_mask, test_mask = slicetca.block_mask(dimensions=data_torch.shape,
                                                train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                                test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
                                                fraction_test=0.1,
                                                device=device)
    # we define the tensor
    # this will take a while to run as it fits 3*3*3*4 = 108 models
    if True:
        min_ranks = [2, 0, 0]
        max_ranks = [10, 1, 1]
        processes_grid = 4
        loss_grid, seed_grid = slicetca.grid_search(data_torch,
                                                    min_ranks = min_ranks,
                                                    max_ranks = max_ranks,
                                                    sample_size=4,
                                                    mask_train=train_mask,
                                                    mask_test=test_mask,
                                                    processes_grid=processes_grid,
                                                    seed=1,
                                                    min_std=10**-4,
                                                    learning_rate=5*10**-3,
                                                    max_iter=10**4,
                                                    positive=True)

        # Plot gridserrch results.                                    
        reduction = "mean"
        fig = slicetca.plot_grid(loss_grid, min_ranks=min_ranks, reduction=reduction, return_fig=True)
        path = f"{savedir}/grid_search.pdf"
        savefig(fig, path)

        # Determine how many components to use
        if reduction == "mean":
            reduced_loss_grid = loss_grid.mean(axis=-1)
        elif reduction == "min":
            reduced_loss_grid = loss_grid.min(axis=-1)
        else:
            raise Exception('Reduction should be mean or min.')
        min_index = np.unravel_index(np.argmin(reduced_loss_grid), reduced_loss_grid.shape)
        number_components = tuple([a + b for a, b in zip(min_ranks, min_index)])
    else:
        number_components = (5,0,0)

    #############
    # The tensor is decomposed into 2 trial-, 0 neuron- and 3 time-slicing components.
    components, model = slicetca.decompose(data_torch,
                                        number_components=number_components,
                                        positive=True,
                                        learning_rate=5*10**-3,
                                        min_std=10**-5,
                                        max_iter=10000,
                                        seed=0)

    # For a not positive decomposition, we apply uniqueness constraints
    # model = slicetca.invariance(model)

    # PLOT LOSS
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(model.losses)), model.losses, 'k')
    ax.set_xlabel('iterations')
    ax.set_ylabel('mean squared error')
    ax.set_xlim(0,len(model.losses))
    path = f"{savedir}/losses.pdf"
    savefig(fig, path)

    ################## Save data
    import pickle 
    
    # with open(f"{savedir}/slicetca.pkl", "wb") as f:
    #     pickle.dump(slicetca, f)

    with open(f"{savedir}/pa.pkl", "wb") as f:
        pickle.dump(pa, f)

    with open(f"{savedir}/components.pkl", "wb") as f:
        pickle.dump(components, f)

    with open(f"{savedir}/model.pkl", "wb") as f:
        pickle.dump(model, f)

    params = {
        "number_components":number_components,
        "loss_grid":loss_grid,
        "min_ranks":min_ranks,
        "reduction":reduction,
        "train_mask":train_mask,
        "test_mask":test_mask,
        }
    with open(f"{savedir}/params.pkl", "wb") as f:
        pickle.dump(params, f)

    ############# PLOT COMPONENTS
    for var_color in list_var_color:
        plot_components(model, pa, var_color, savedir)

    # from neuralmonkey.analyses.state_space_good import _trajgood_make_colors_discrete_var
    
    # times = pa.Times
    # for var_color in list_var_color:
    #     labels = pa.Xlabels["trials"][var_color]
    #     _map_lev_to_color, color_type, colors = _trajgood_make_colors_discrete_var(labels)
    #     colors_arr = np.stack(colors)

    #     # sort by trials
    #     trial_idx = np.argsort(labels).tolist()

    #     # we sort the neurons of the trial slices according to their peak activity in the first slice.
    #     if False: # keep unsorted, so they match up across events:
    #         neuron_sorting_peak_time = np.argsort(np.argmax(components[0][1][0], axis=1))
    #     else:
    #         neuron_sorting_peak_time = None

    #     # call plotting function, indicating index for sorting trials and colors for different angles as well as time
    #     fig, axes = slicetca.plot(model,
    #                 variables=('trial', 'neuron', 'time'),
    #                 colors=(colors_arr[trial_idx], None, None), # we only want the trials to be colored
    #                 ticks=(None, None, np.linspace(0, len(times),3)), # we only want to modify the time ticks
    #                 tick_labels=(None, None, np.linspace(times[0],times[-1],3)),
    #                 sorting_indices=(trial_idx, neuron_sorting_peak_time, None),
    #                 quantile=0.99, return_fig=True)

    #     path = f"{savedir}/components-colorby={var_color}.pdf"
    #     savefig(fig, path)    

    #     from pythonlib.tools.plottools import legend_add_manual
    #     # - legend for the color
    #     fig, ax = plt.subplots()
    #     legend_add_manual(ax, _map_lev_to_color.keys(), _map_lev_to_color.values())
    #     ax.set_title(var_color)
    #     savefig(fig, f"{savedir}/legend_colorby={var_color}.pdf")

    plt.close("all")


if __name__ == "__main__":

    ########## PARAMS
    # animal = "Diego"
    # date = 230615

    animal = sys.argv[1]
    date = int(sys.argv[2])
    which_level = sys.argv[3]

    TWIND_ANALY = (-0.15, 0.45)

    ### DERIVED PARAMS
    SAVEDIR = f"/lemur2/lucas/analyses/recordings/main/sliceTCA/{animal}-{date}"

    # Load data
    DFallpa = load_handsaved_wrapper(animal=animal, date=date, version=which_level)
    dfallpa_preprocess_fr_normalization(DFallpa, "across_time_bins")

    # # Normalize to Fr range from (0, ~2)
    # for pa in DFallpa["pa"].values:
    #     if False:
    #         mins = np.min(np.min(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
    #         maxs = np.max(np.max(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
    #         pa.X = (pa.X - mins)/(maxs-mins)
    #     else:
    #         MAX = 2*np.max(np.std(pa.X.reshape((pa.X.shape[0], pa.X.shape[1] * pa.X.shape[2])), axis=1))
    #         mins = np.min(np.min(pa.X, axis=1, keepdims=True), axis=2, keepdims=True)
    #         pa.X = (pa.X - mins)/(MAX)        
    #         # pa.X = pa.X/pa.X.std() # to put in range


    for i, row in DFallpa.iterrows():
        which_level = row["which_level"]
        bregion = row["bregion"]
        event = row["event"]
        twind = row["twind"]
        pa = row["pa"]

        savedir = f"{SAVEDIR}/{which_level}-{event}-{bregion}-{twind}"
        os.makedirs(savedir, exist_ok=True)
        print("... Saving sliceTCA results to:", savedir)

        ### Run
        list_var_color = ["seqc_0_shape", "seqc_0_loc"]
        run(pa, savedir, TWIND_ANALY, list_var_color)

        # # Plot exmaple of teh data before it goes into tca
        # fig, ax = plt.subplots()
        # pa.plotNeurHeat(0, ax=ax)
        # savefig(fig, f"{savedir}/example_data_before_tca_single_trial.pdf")

        # # Get data in correct foramt    
        # data_np = np.transpose(pa.X, (1, 0, 2)) # your_data is a numpy array of shape (trials, neurons, time).
        # data_torch = torch.tensor(data_np, device=device, dtype=torch.float32)
        
        # ############
        # train_mask, test_mask = slicetca.block_mask(dimensions=data_torch.shape,
        #                                             train_blocks_dimensions=(1, 1, 10), # Note that the blocks will be of size 2*train_blocks_dimensions + 1
        #                                             test_blocks_dimensions=(1, 1, 5), # Same, 2*test_blocks_dimensions + 1
        #                                             fraction_test=0.1,
        #                                             device=device)
        # # we define the tensor
        # # this will take a while to run as it fits 3*3*3*4 = 108 models
        # if True:
        #     min_ranks = [2, 0, 0]
        #     max_ranks = [10, 1, 1]
        #     processes_grid = 4
        #     loss_grid, seed_grid = slicetca.grid_search(data_torch,
        #                                                 min_ranks = min_ranks,
        #                                                 max_ranks = max_ranks,
        #                                                 sample_size=4,
        #                                                 mask_train=train_mask,
        #                                                 mask_test=test_mask,
        #                                                 processes_grid=processes_grid,
        #                                                 seed=1,
        #                                                 min_std=10**-4,
        #                                                 learning_rate=5*10**-3,
        #                                                 max_iter=10**4,
        #                                                 positive=True)

        #     # Plot gridserrch results.                                    
        #     reduction = "mean"
        #     fig = slicetca.plot_grid(loss_grid, min_ranks=min_ranks, reduction=reduction, return_fig=True)
        #     path = f"{savedir}/grid_search.pdf"
        #     savefig(fig, path)

        #     # Determine how many components to use
        #     if reduction == "mean":
        #         reduced_loss_grid = loss_grid.mean(axis=-1)
        #     elif reduction == "min":
        #         reduced_loss_grid = loss_grid.min(axis=-1)
        #     else:
        #         raise Exception('Reduction should be mean or min.')
        #     min_index = np.unravel_index(np.argmin(reduced_loss_grid), reduced_loss_grid.shape)
        #     number_components = tuple([a + b for a, b in zip(min_ranks, min_index)])
        # else:
        #     number_components = (5,0,0)

        # #############
        # # The tensor is decomposed into 2 trial-, 0 neuron- and 3 time-slicing components.
        # components, model = slicetca.decompose(data_torch,
        #                                     number_components=number_components,
        #                                     positive=True,
        #                                     learning_rate=5*10**-3,
        #                                     min_std=10**-5,
        #                                     max_iter=10000,
        #                                     seed=0)

        # # For a not positive decomposition, we apply uniqueness constraints
        # # model = slicetca.invariance(model)

        # # PLOT LOSS
        # fig, ax = plt.subplots()
        # ax.plot(np.arange(len(model.losses)), model.losses, 'k')
        # ax.set_xlabel('iterations')
        # ax.set_ylabel('mean squared error')
        # ax.set_xlim(0,len(model.losses))
        # path = f"{savedir}/losses.pdf"
        # savefig(fig, path)

        # ############# PLOT COMPONENTS
        # from neuralmonkey.analyses.state_space_good import _trajgood_make_colors_discrete_var
        # list_var_color = ["seqc_0_shape", "seqc_0_loc"]
        # times = pa.Times
        # for var_color in list_var_color:
        #     labels = pa.Xlabels["trials"][var_color]
        #     _map_lev_to_color, color_type, colors = _trajgood_make_colors_discrete_var(labels)
        #     colors_arr = np.stack(colors)

        #     # sort by trials
        #     trial_idx = np.argsort(labels).tolist()

        #     # we sort the neurons of the trial slices according to their peak activity in the first slice.
        #     if False: # keep unsorted, so they match up across events:
        #         neuron_sorting_peak_time = np.argsort(np.argmax(components[0][1][0], axis=1))
        #     else:
        #         neuron_sorting_peak_time = None

        #     # call plotting function, indicating index for sorting trials and colors for different angles as well as time
        #     fig, axes = slicetca.plot(model,
        #                 variables=('trial', 'neuron', 'time'),
        #                 colors=(colors_arr[trial_idx], None, None), # we only want the trials to be colored
        #                 ticks=(None, None, np.linspace(0, len(times),3)), # we only want to modify the time ticks
        #                 tick_labels=(None, None, np.linspace(times[0],times[-1],3)),
        #                 sorting_indices=(trial_idx, neuron_sorting_peak_time, None),
        #                 quantile=0.99, return_fig=True)

        #     path = f"{savedir}/components-colorby={var_color}.pdf"
        #     savefig(fig, path)    

        #     from pythonlib.tools.plottools import legend_add_manual
        #     # - legend for the color
        #     fig, ax = plt.subplots()
        #     legend_add_manual(ax, _map_lev_to_color.keys(), _map_lev_to_color.values())
        #     ax.set_title(var_color)
        #     savefig(fig, f"{savedir}/legend_colorby={var_color}.pdf")
            
        # plt.close("all")