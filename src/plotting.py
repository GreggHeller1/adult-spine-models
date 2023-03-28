import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg


## how should this work?
#Start with the raw traces.
#Subsample. Is this a stop point or should each sorting aply the sampling?

#Generate the sort mat - needs to be able to be applied to another easily
#right now for columns and rows this isn't actually a matrix... we just pass in the matrix to sort by.

#sort by this matrix - produces mat.
#^^^Sometimes we have this leaving in 2d, sometimes we have brought to 2d already... which one?
#the only reason to leave it 2d is if we want to apply a preferred idrection sorting after
#which most of the time we may want to do because this is something that parses the response nicely.
# the big thing that we want to do is sort across directions by the soma response.
#AND ALSO within directions by the spine response (amplitude, peak time, whatever. )
#I think we want to apply the Across sorting second, since the within shuffles will mess it up.
#so we want to produce a matrix from the within sorts


#plot or compare, etc

def plot_activity_plots(selected_timesteps):
    fig, axs = plt.subplots(1,8)

    #PLot the oritinal traces
    axs[0].imshow(flatten_for_image(selected_timesteps))


    new_idx = cfg.sort_within_then_across_stims(selected_timesteps, cfg.sort_by_mean_amp)
    ordered_traces = cfg.use_as_index(new_idx, selected_timesteps)
    axs[1].imshow(flatten_for_image(ordered_traces))

    sort_mat = cfg.sort_by_peak_time(selected_timesteps)
    sotred_traces = cfg.use_as_index(sort_mat, selected_timesteps)
    axs[2].imshow(sotred_traces)

    sort_mat = cfg.sort_by_onset_time(selected_timesteps)
    sotred_traces = cfg.use_as_index(sort_mat, selected_timesteps)
    axs[3].imshow(sotred_traces)

    sort_mat = cfg.sort_by_mean_amp(selected_timesteps)
    sotred_traces = cfg.use_as_index(sort_mat, selected_timesteps)
    axs[4].imshow(sotred_traces)

    #plot the means in each time bin (stretched to appear the same as the full trace)
    selected_period_means = cfg.get_period_means(selected_timesteps)
    axs[5].imshow(flatten_for_image(selected_period_means))

    #plot the sorted means
    sorted_period_means = cfg.use_as_index(new_idx, selected_period_means)
    axs[6].imshow(flatten_for_image(sorted_period_means))

    #and plot the boolean values (inherited sorting from the sorted means)
    bool_activity = cfg.get_bool_activity(sorted_period_means)
    axs[7].imshow(flatten_for_image(bool_activity))


def flatten_for_image(d3_array):
    return d3_array.reshape(d3_array.shape[0]*d3_array.shape[1], d3_array.shape[2])



#######################




