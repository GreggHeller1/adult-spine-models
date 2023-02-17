import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg


def plot_activity_plots(selected_timesteps, selected_period_means, sorted_period_means, bool_soma_activity):
    fig, axs = plt.subplots(1,7)

    #PLot the oritinal traces
    axs[0].imshow(flatten_for_image(selected_timesteps))

    sort_mat = sort_by_peak_time(selected_timesteps)
    sotred_traces = sort_for_image(selected_timesteps, sort_mat)
    axs[1].imshow(sotred_traces)

    sort_mat = sort_by_steepest_time(selected_timesteps)
    sotred_traces = sort_for_image(selected_timesteps, sort_mat)
    axs[2].imshow(sotred_traces)

    sort_mat = sort_by_mean_amp(selected_timesteps)
    sotred_traces = sort_for_image(selected_timesteps, sort_mat)
    axs[3].imshow(sotred_traces)

    #plot the means in each time bin (stretched to appear the same as the full trace)
    axs[4].imshow(flatten_for_image(selected_period_means))

    #plot the sorted means
    axs[5].imshow(flatten_for_image(sorted_period_means))

    #and plot the boolean values (inherited sorting from the sorted means)
    axs[6].imshow(flatten_for_image(bool_soma_activity))


def sort_for_image(traces, sort_mat):
    sorted_traces = []
    for i, j in zip(sort_mat[0], sort_mat[1]):
        #print(i,j)
        sorted_traces.append(traces[i,j,:])

    return sorted_traces #Should be shape #traces x #timepoints


def flatten_for_image(d3_array):
    return d3_array.reshape(d3_array.shape[0]*d3_array.shape[1], d3_array.shape[2])



#######################




def reorder_columns(mat, sort_mat):
    #Reorder the directions so that the max direction is first
    row_means = np.mean(sort_mat, axis=-1)
    row_ordering = np.argsort(row_means)[::-1]
    row_sorted_mat = mat[row_ordering]
    return row_sorted_mat

def reorder_mat(mat, sort_mat):
    column_sorted_mat = cfg.reorder_rows(mat, sort_mat)
    #^^^ This one is withing trials
    #column_sorted_mat = mat
    return reorder_columns(column_sorted_mat, sort_mat)

def reorder_3d_array(d3_array, sort_mat):
    sorted_array = np.empty(d3_array.shape)
    for i in range(d3_array.shape[2]):
        sorted_array[:,:,i] = reorder_mat(d3_array[:,:,i], sort_mat)
    return sorted_array


#def reorder_3d_array

def sort_by_peak_time(traces):

    #get the times of the peaks
    filtered_traces = traces.copy()
    filtered_traces[traces<3] = 0
    filtered_traces[:,:,-1] = .001
    peak_times = np.argmax(filtered_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(peak_times, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, peak_times.shape))


    return sort_mat

def sort_by_steepest_time(traces):

    #get the times of the peaks
    diff_traces = traces[:,:,1:] - traces[:,:,:-1]
    diff_traces[diff_traces<1.5] = 0
    diff_traces[:,:,-1] = .001
    peak_times = np.argmax(diff_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(peak_times, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, peak_times.shape))


    return sort_mat

def sort_by_max_amp(traces):

    #get the times of the peaks
    trace_means = np.max(traces.copy(), axis=-1)
    #filtered_traces[traces<3] = 0
    #filtered_traces[:,:,-1] = .001
    # = np.argmax(filtered_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(trace_means, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, trace_means.shape))
    sort_mat = sort_mat[:,::-1] #We want sorted high to low, not low to high

    return sort_mat


def sort_by_mean_amp(traces):

    #get the times of the peaks
    trace_means = np.mean(traces.copy(), axis=-1)
    #filtered_traces[traces<3] = 0
    #filtered_traces[:,:,-1] = .001
    # = np.argmax(filtered_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(trace_means, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, trace_means.shape))
    sort_mat = sort_mat[:,::-1] #We want sorted high to low, not low to high

    return sort_mat


def get_period_means(selected_timesteps):
    selected_period_means = np.empty(selected_timesteps.shape)

    #print(selected_timesteps.shape)

    scale_width = 0
    #Reduce each time period trace to the mean in each period
    for i in range(cfg.num_tranges):
        mean_activity_in_trange = np.mean(selected_timesteps[:,:,i*cfg.timepoints_per_period:(i+1)*cfg.timepoints_per_period], axis=2)
        #print(mean_activity_in_trange.shape)
        for j in range(cfg.timepoints_per_period):
            selected_period_means[:,:,j+i*cfg.timepoints_per_period] = mean_activity_in_trange #tried doing this with tile and ran into trouble/
    return selected_period_means



def produce_activity_plots(selected_timesteps):

    #Not super happy with how this is structured... which intermediate matricies need to be saved and kept?
    selected_period_means = get_period_means(selected_timesteps)

    ## we want to sort this one based on the difference between the pre and post stime periods
    first_entry_after_stim = selected_period_means[:,:, cfg.start_s*-1*cfg.frame_rate]
    sort_mat = selected_period_means[:,:, cfg.start_s*-1*cfg.frame_rate] - selected_period_means[:,:,0]

    sorted_period_means = reorder_3d_array(selected_period_means, sort_mat)

    #Use this to determine the soma threshold
    #plt.hist(first_entry_after_stim)
    #spine_threshold = 2
    soma_threshold = .5
    #then boolean over or under the soma threshold
    bool_activity = sorted_period_means>soma_threshold
    return selected_period_means, sorted_period_means, bool_activity
