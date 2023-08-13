import matplotlib.pyplot as plt
import numpy as np

from src import config as cfg
from src import computation as comp
from src import helper_functions as hf


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



def reorder_directions_for_plotting(array, preferred_direcion_idx=None, goal_idx = 0):
    if preferred_direcion_idx is None:
        preferred_direcion_idx = np.argmax(array)
    return np.roll(array, -(preferred_direcion_idx-goal_idx)) #-subtracting 8 will center the peak. what happens if this is negative?



def plot_tuning_curves(ax=None, **kwargs):
    #fig, ax = line_plot_key_labels(ax, **kwargs)
    fig=None
    fig, ax = new_ax(ax)
    preferred_direction = 0
    for label, means_array in kwargs.items():
        if ('soma' in label.lower()) and not('to soma' in label.lower()):
            preferred_direction = np.argmax(np.array(means_array))

    num_stims = len(means_array)
    goal_peak_idx = int(num_stims/2)
    print(goal_peak_idx)
    for key, means_array in kwargs.items():
        ordered_response_array = reorder_directions_for_plotting(means_array, preferred_direcion_idx=preferred_direction, goal_idx = goal_peak_idx)
        ax.plot(ordered_response_array, label=key)

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_xlabel('Direction')
    ax.set_xticks([(goal_peak_idx - num_stims/2)%num_stims,(goal_peak_idx - num_stims/4)%num_stims,goal_peak_idx, (goal_peak_idx + num_stims/4)%num_stims],
                  ['Anti-preferred', 'Orthogonal', 'Preferred', 'Orthogonal'])
    ax.set_ylabel('Normalized amplitude')
    return fig, ax


def line_plot_key_labels(ax=None, **kwargs):

    for key, array in kwargs.items():

        ax.plot(array, label=key)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return fig, ax


def new_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    return fig, ax


#plot or compare, etc


def plot_activity_plots(selected_timesteps):
    fig, axs = plt.subplots(1,8)

    #PLot the oritinal traces
    axs[0].imshow(flatten_for_image(selected_timesteps))


    new_idx = sort_within_then_across_stims(selected_timesteps, sort_by_mean_amp)
    ordered_traces = use_as_index(new_idx, selected_timesteps)
    axs[1].imshow(flatten_for_image(ordered_traces))

    sort_mat = sort_by_peak_time(selected_timesteps)
    sotred_traces = use_as_index(sort_mat, selected_timesteps)
    axs[2].imshow(sotred_traces)

    sort_mat = sort_by_onset_time(selected_timesteps)
    sotred_traces = use_as_index(sort_mat, selected_timesteps)
    axs[3].imshow(sotred_traces)

    sort_mat = sort_by_mean_amp(selected_timesteps)
    sotred_traces = use_as_index(sort_mat, selected_timesteps)
    axs[4].imshow(sotred_traces)

    #plot the means in each time bin (stretched to appear the same as the full trace)
    selected_period_means = hf.get_period_means(selected_timesteps)
    axs[5].imshow(flatten_for_image(selected_period_means))

    #plot the sorted means
    sorted_period_means = use_as_index(new_idx, selected_period_means)
    axs[6].imshow(flatten_for_image(sorted_period_means))

    #and plot the boolean values (inherited sorting from the sorted means)
    bool_activity = hf.get_bool_activity(sorted_period_means)
    axs[7].imshow(flatten_for_image(bool_activity))


def flatten_for_image(d3_array):
    return d3_array.reshape(d3_array.shape[0]*d3_array.shape[1], d3_array.shape[2])



#######################
#For Neuron

def _plot_section_from_above(section, ax=None):
    fig, ax = new_ax(ax)
    try:
        coords_array = np.array(section.psection()['morphology']['pts3d'])
        ax.plot(coords_array[:,0], coords_array[:,1])
    except Exception as E:
        pass #its probably the soma

def _plot_section(section, ax=None, c='b'):
    fig, ax = new_ax(ax)
    try:
        coords_array = np.array(section.psection()['morphology']['pts3d'])
        ax.plot(coords_array[:,0], coords_array[:,1], coords_array[:,2], c=c)
    except Exception as E:
        pass #its probably the soma


def fovs_on_morphology(h, spine_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,12))
    ax = fig.add_subplot(111, projection='3d')

    for section in h.allsec():
        _plot_section(section, ax)

    for i, (activity, metadata) in enumerate(hf.fov_generator(spine_data)):
        #print(metadata['structural_data']['order'][0][0], metadata['structural_data']['DistanceFromRoot_um'][0][0], metadata['structural_data']['DistanceAlongBranch_um'][0][0])

        xyz_coords = hf.get_xyz_coords_of_fov(spine_data, fov_num = i)
        nearest_section, sec_coords, min_dist  = hf.find_closest_section(h, xyz_coords)
        _plot_section(nearest_section, ax, c='g')

        #order, dist, dist_from_branch = hf.get_branch_order_and_dist(nearest_section, xyz_coords)
        #print(      order, dist, dist_from_branch)
        #print('###')
        label_txt = hf.get_fov_name(spine_data, i)

        ax.plot(xyz_coords[0], xyz_coords[1], xyz_coords[2], marker='x', label=label_txt)
        ax.view_init(elev=90., azim=0.)

    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)


def manual_and_auto_shift(h, spine_data, fov_num, shifts_by_fov):
    pass



def spine_centers_on_morphology(h, spine_data, fov_num, manual_adjstment = np.array([0,0,0]), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize = (12,12))
        ax = fig.add_subplot(111, projection='3d')

    xyz_coords = hf.get_xyz_coords_of_fov(spine_data, fov_num = fov_num)
    nearest_section, sec_coords, min_dist  = hf.find_closest_section(h, xyz_coords)
    _plot_section(nearest_section, ax, c='g')

    spines_pixel_coords = hf.get_spines_pixel_coords(spine_data, fov_num)
    for i in range(spines_pixel_coords.shape[1]):
        spine_global_coords = hf.get_spine_global_coords(h, spine_data, fov_num, i, manual_adjstment)
        nearest_section, sec_coords, min_dist  = hf.find_closest_section(h, spine_global_coords)
        _plot_section(nearest_section, ax, c='g')
        ax.plot(spine_global_coords[0], spine_global_coords[1], spine_global_coords[2], marker='x', label=i)
        ax.view_init(elev=90., azim=0.)
    ax.set_aspect('equal')
    return ax

def manual_and_auto_shift(h, spine_data, fov_num, selected_manual_shift, optimal_shift):
    fig, axs = plt.subplots(figsize = (24,12))
    ax0 = fig.add_subplot(121, projection='3d')
    ax1 = fig.add_subplot(122, projection='3d')
    spine_centers_on_morphology(h, spine_data, fov_num, manual_adjstment = selected_manual_shift, ax=ax0)
    spine_centers_on_morphology(h, spine_data, fov_num, manual_adjstment = optimal_shift, ax=ax1)


#################
#Sorting algs


def reorder_rows(mat, sort_mat):
    """Reorder the activity within each direction with the max amplitude first
    """
    #assert(mat.shape==sort_mat.shape)
    column_ordering = np.argsort(sort_mat)
    row_sorted_mat = np.empty(mat.shape)
    for row in range(column_ordering.shape[0]):
        row_sorted_mat[row,:] = mat[row,:][column_ordering[row,::-1]]
    return row_sorted_mat


def return_as_is(mat):
    return mat


def sort_by_mean_amp(traces):
    #get the means
    trace_means = np.mean(traces.copy(), axis=-1)

    #get the order to sort them
    sort_order = np.argsort(trace_means, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, trace_means.shape))
    sort_mat = sort_mat[:,::-1] #We want sorted high to low, not low to high
    return sort_mat

def sort_within_then_across_stims(selected_timesteps, sorting_function = sort_by_mean_amp):
    a = sort_within_all_stims(selected_timesteps, sorting_function)
    b = sort_across_stims(selected_timesteps)
    sort_mat = use_as_index(b,a)
    return sort_mat



def use_as_index(index_array, array_to_be_indexed):
    if len(index_array.shape) == 3:
        return array_to_be_indexed[index_array[:,:,0],index_array[:,:,1]]
    elif len(index_array.shape) == 2:
        return array_to_be_indexed[index_array[0,:],index_array[1,:]]

def my_mean(traces):
    return np.mean(traces, axis=-1)

def sort_across_stims(trace_mat, apply_to_traces = my_mean, apply_to_rows = my_mean):
    #Reorder the directions so that the max direction is first
    #would we ever want to do this using anything other than mean? I could see mean, max, peak time... if some stims respond later
    trace_stat_mat = apply_to_traces(trace_mat)
    #print('#', trace_stat_mat.shape)
    row_stat_vect = apply_to_rows(trace_stat_mat)
    #print('#', row_stat_vect.shape)
    row_ordering = np.argsort(row_stat_vect)[::-1]
    index_mat_shape = trace_mat.shape[:-1]+(2,)
    index_mat = np.zeros(index_mat_shape)
    for column in range(index_mat.shape[1]):
        index_mat[:,column,1] = column
        index_mat[:,column,0] = row_ordering
    return index_mat.astype(int)


def sort_within_all_stims(trace_mat, sorting_function):
    """Reorder the activity within each direction with the max amplitude first
    """
    index_mat_shape = trace_mat.shape[:-1]+(2,)
    index_mat = np.zeros(index_mat_shape)

    for row in range(index_mat.shape[0]):
        ordering = sorting_function(trace_mat[row, :,:])
        #print(ordering.shape)
        #print('#', index_mat[row,:,:].shape)
        index_mat[row,:,1] = ordering
        index_mat[row,:,0] = row
    return index_mat.astype(int)



def sort_by_max_amp(traces):
    #get the means
    trace_means = np.max(traces.copy(), axis=-1)

    #get the order to sort them
    sort_order = np.argsort(trace_means, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, trace_means.shape))
    sort_mat = sort_mat[:,::-1] #We want sorted high to low, not low to high
    return sort_mat

def sort_by_peak_time(traces):

    #get the times of the peaks
    filtered_traces = traces.copy()
    threshold_in_standard_deviations = 3
    filtered_traces[traces<threshold_in_standard_deviations] = 0
    filtered_traces[:,:,-1] = .001  #<- this part will be problematic if we pass a 2d instead of 1D, but only necessary to account for random blips... probaby a better solution
    peak_times = np.argmax(filtered_traces, axis=-1)
    #print('###', peak_times.shape)
    #get the order to sort them
    sort_order = np.argsort(peak_times, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, peak_times.shape))
    return sort_mat #<- is actually just a 2xN array, 2 indicies for each entry.

def sort_by_steepest_time(traces):
    diff_traces = traces[:,:,1:] - traces[:,:,:-1]
    threshold_in_standard_deviations = 3
    diff_traces[diff_traces<threshold_in_standard_deviations*np.std(diff_traces)] = 0
    diff_traces[:,:,-1] = .001
    peak_times = np.argmax(diff_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(peak_times, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, peak_times.shape))
    return sort_mat

def sort_by_onset_time(traces):
    diff_traces = traces[:,:,1:] - traces[:,:,:-1]
    threshold_in_standard_deviations = 2
    std = np.std(diff_traces)
    last_timestep = 1.1*(threshold_in_standard_deviations*std)
    diff_traces[:,:,-1] = last_timestep
    diff_traces = diff_traces>(threshold_in_standard_deviations*std)
    onsets_times = np.argmax(diff_traces, axis=-1)

    #get the order to sort them
    sort_order = np.argsort(onsets_times, axis=None)
    sort_mat = np.array(np.unravel_index(sort_order, onsets_times.shape))
    return sort_mat



def get_most_similar_spine(soma_data, all_spine_activity_array, ordering_func = return_as_is):

    #Should we be passing in the soma data and THEN subselecting? don't really see why not. may need other metadata at some point
    #^^ Now implemented

    soma_traces = hf.get_traces(soma_data)
    #soma_traces = comp.select_timesteps(soma_traces)
    soma_activity_mat = ordering_func(soma_traces)

    similarity_list = []
    fov_num_list = []
    spine_num_list = []
    for i in range(all_spine_activity_array.shape[0]):


        #this_spine_traces = np.array(fov_field_2['trial_traces'][:,:,0,:,i].swapaxes(0,-1))

        #this_spine_traces = get_traces(spine_data, fov=fov, spine_index=i)
        this_spine_sub_traces = all_spine_activity_array[i,:,:,:]
        this_spine_activity_mat = ordering_func(this_spine_sub_traces)

        similarity = comp.compare_tuning_curves(soma_activity_mat, this_spine_activity_mat)
        #similarity = np.dot(np.array(soma_activity_mat).flatten(), np.array(this_spine_activity_mat).flatten())
        #similarity = np.mean(soma_activity_mat - this_spine_activity_mat)

        similarity_list.append(similarity)
        spine_num_list.append(i)

    similarity_list = np.array(similarity_list)
    spine_num_list = np.array(spine_num_list)
    ordering = np.argsort(similarity_list)[::-1]

    return similarity_list[ordering], spine_num_list[ordering]





