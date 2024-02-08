import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import functools
from statannotations.Annotator import Annotator

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
        if ('soma' in label.lower()) and not('to_soma' in label.lower()):
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



##########################################################################
#For the main_plotting_loop... should just put that here too??

def plot_decorator(plot_func):
    @functools.wraps(plot_func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        fig, ax = plt.subplots()

        print("###########################################################")
        print('')
        print('Creating next plot')
        #run the function
        prefix = plot_func(*args, fig=fig, ax=ax,  **kwargs)
        print(f'Finished with {prefix}')

        #do something after
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        ## Get the legend handles and add them to plt.legend, so that the right handles are addigned
        #legend_handles, _= plt.gca().get_legend_handles_labels()
        #plt.legend(handles = legend_handles, loc='right',
        #           labels=['Group 1: C1', 'Group 1: C2', 'Group 2: C1', 'Group 2: C2'])


        figname = f'{prefix}{"-"*bool(prefix)}model_performance_summaries.{cfg.figure_type}'
        fig_path = os.path.join(cfg.collect_summary_at_path, figname)
        print(f'Saving figure to {fig_path}')
        fig.savefig(fig_path, bbox_inches='tight')
        return fig, ax
    return wrapper_decorator

    

@plot_decorator
def plot_model_simulation_scores(df,  prefix='plot2', fig=None, ax=None):
    sns.lineplot(data=df.loc[df['responsive_status'] == False], x='nickname', y='model_soma_similarity_score', hue='experiment_id',
                 palette='Blues', marker='o')
    sns.lineplot(data=df.loc[df['responsive_status'] == True], x='nickname', y='model_soma_similarity_score', hue='experiment_id',
                 palette='Reds', marker='o')
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=-45, ha="left", rotation_mode="anchor") #from https://stackoverflow.com/questions/28615887/how-to-move-a-tick-label-in-matplotlib
    return prefix


@plot_decorator
def plot_4(df,  prefix='plot4', fig=None, ax=None):
    #Democratic, distance, size
    palette_colors = [cfg.colors['unresponsive'], cfg.colors['responsive']]
    prefix = my_violin_swarm(df,  x='weights', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            #order=['resp_only', 'democratic_weights', 'unresp_only'],
                            alpha= .6,
                            hue = 'responsive_status',
                            split=True
                            )
    return prefix

@plot_decorator
def plot_4_2(df,  prefix='plot4', fig=None, ax=None):
    #Democratic, distance, size
    palette_colors = [cfg.colors['unresponsive'], cfg.colors['responsive']]
    prefix = my_violin_swarm(df,  x='weights', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            #order=['resp_only', 'democratic_weights', 'unresp_only'],
                            alpha= .6,
                            hue = 'responsive_status',
                            split=False
                            )
    return prefix

@plot_decorator
def plot_3(df,  prefix='plot3', fig=None, ax=None):
    #Democratic, distance, size
    print(df.head())
    palette_colors = [cfg.colors['unresponsive'], cfg.colors['responsive']]
    prefix = my_violin_swarm(df,  x='weights', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            #order=['democratic_weights', 'size_weights', 'dist_weights'],
                            alpha= .6,
                            hue = 'responsive_status',
                            split=True
                            )
    return prefix

@plot_decorator
def plot_3_2(df,  prefix='plot3', fig=None, ax=None):
    #Democratic, distance, size
    print(df.head())
    palette_colors = [cfg.colors['unresponsive'], cfg.colors['responsive']]
    prefix = my_violin_swarm(df,  x='weights', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            #order=['democratic_weights', 'size_weights', 'dist_weights'],
                            alpha= .6,
                            hue = 'responsive_status',
                            split=False
                            )
    return prefix

@plot_decorator
def plot_2(df,  prefix='plot2', fig=None, ax=None):
    palette_colors = [cfg.colors['baps_trials_only'], cfg.colors['responsive'], cfg.colors['no_bap_trials']]
    prefix = my_violin_swarm(df,  x='trial_subset', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            order=['baps_trials_only', 'all_trials', 'no_bap_trials'],
                            alpha= .6,
                            )
    return prefix

@plot_decorator
def plot_1(df,  prefix='plot1', fig=None, ax=None):
    palette_colors = [cfg.colors['responsive'], cfg.colors['unresponsive']]
    prefix = my_violin_swarm(df,  x='responsive_status', y='model_soma_similarity_score',
                            prefix=prefix, fig=fig, ax=ax,
                            palette=palette_colors,
                            alpha=.6,
                            order=[True, False],
                            )
    return prefix


def my_violin_swarm(df, x, y, prefix='', fig=None, ax=None, alpha=.4, palette=None,
                    cut=2, linewidth_vi=4, linewidth_sw=1, size=15, color_sw='k', order=None,
                    hue=None, split=False
                    ):
    if not(palette is None):
        palette = sns.color_palette(palette, len(palette))


    #sns.violinplot(data=df, x='responsive_status', y='model_soma_similarity_score')
    if cfg.plot_type == 'violin':
        sns.violinplot(data=df, x=x, y=y, palette=palette, hue=hue, split=split,
                       saturation=.75, cut=cut, linewidth=linewidth_vi,
                       order=order, hue_order=order)

    if cfg.plot_type == 'bar':
        sns.barplot(data=df, x=x, y=y, palette=palette, hue=hue,
                       saturation=.75, linewidth=linewidth_vi,
                       order=order, hue_order=order)

    ax = plt.gca()
    pairs_list = []
    for i, category_1 in enumerate(df.loc[:,x].unique()):
        for j, category_2 in enumerate(df.loc[:,x].unique()):
            if j>i: #this prevents comparing to self and repeated pars
                pairs_list.append((category_1, category_2))

    try:
        annotator = Annotator(ax, pairs_list, data=df, x=x, y=y, order=order)
        annotator.configure(test='t-test_paired', text_format='star', loc='outside')
        annotator.apply_and_annotate()
    except Exception as E:
        annotator = Annotator(ax, pairs_list, data=df, x=x, y=y, order=order)
        annotator.configure(test='t-test_ind', text_format='star', loc='outside')
        annotator.apply_and_annotate()
    

    for violin in ax.collections[::]:
        violin.set_alpha(alpha)
    #sns.violinplot(data=df, x='responsive_status', y='model_soma_similarity_score', hue='responsive_status', order=['responsive', 'unresponsive'], hue_order=None,
    #    bw='scott', cut=0, scale='area', scale_hue=True, gridsize=100, width=0.8, inner='box',
    #    split=False, dodge=True, orient=None, linewidth=None, color='r', palette='Reds', saturation=0.75)
    #Pallette - sns.color_palette(my_palette, 2)

    if cfg.include_swarm:
        sns.stripplot(data=df,  x=x, y=y, marker="x", linewidth=linewidth_sw, size=size, color=color_sw, order=order)#)# , , )#, order=['responsive', 'unresponsive'])
    #sns.swarmplot(data=df, x='responsive_status', y='model_soma_similarity_score', hue='responsive_status', order=['responsive', unresponsive], hue_order=None, dodge=False,
    #    orient=None, color=None, palette=sns.color_palette(my_palette, 2), size=5, edgecolor='gray', linewidth=0, hue_norm=None,
    #    native_scale=False, formatter=None, legend='auto', warn_thresh=0.05, ax=None, **kwargs)


    #also plot connecting them 
    if cfg.include_line:
        sns.lineplot(data=df,  x=x, y=y, hue='experiment_id',
                 color=color_sw)#, order=order)

    csv_name = f'{prefix}{"-"*bool(prefix)}model_performance_summaries.csv'
    csv_path = os.path.join(cfg.collect_summary_at_path, csv_name)
    df.to_csv(csv_path)
    return prefix

def plot_all_simulation_scores(df):

    #first plot: democratic_scores split by soma repsonsiveness

    plot_1(df[(df['weights']=='democratic_weights') & (df['trial_subset']=='all_trials')])


    #second_plot:
    #normalize the similarity scores first
    if cfg.normalize_similarity_scores:
        exp_ids = df['experiment_id'].unique()
        for exp_id in exp_ids:
            bool_index = df['experiment_id'] == exp_id
            exp_df = df[bool_index]
            democratic_index = (df['weights']=='democratic_weights') & (df['trial_subset']=='all_trials')
            #print(democratic_index)
            #print('%%%%%%')
            democratic_sim_score = float(exp_df[democratic_index]['model_soma_similarity_score'])
            #print(democratic_sim_score)
            exp_df.loc[:,'model_soma_similarity_score'] = exp_df.loc[:,'model_soma_similarity_score']/democratic_sim_score
            #must use .loc when setting so it doesn't copy the column

            #print(exp_df['model_soma_similarity_score'])
            #put it back in the larger data frame
            df[bool_index] = exp_df


    plot_model_simulation_scores(df, prefix='normalized')
    plot_model_simulation_scores(df[df['trial_subset']=='all_trials'], prefix='normalized2')

    plot_2(df[(df['weights']=='democratic_weights')
           & (df['responsive_status']==True)
           ], prefix = 'plot2R')

    #plot_2(df[(df['weights']=='democratic_weights')
    #       & (df['responsive_status']==True)
    #       & ((df['experiment_id']=='ASC16')
    #       | (df['experiment_id']=='ASC20')
    #       | (df['experiment_id']=='ASC28 cell 3')
    #       | (df['experiment_id']=='ASC22'))
    #       ], prefix = 'plot2Rs')

    #plot_3(df[
    #           ((df['weights']=='democratic_weights')
    #           | (df['weights']=='size_weights')
    #           | (df['weights']=='dist_weights'))
    #           & (df['trial_subset']=='all_trials')
    #           ])

    #plot_3_2(df[
    #       ((df['weights']=='democratic_weights')
    #       | (df['weights']=='size_weights')
    #       | (df['weights']=='size_AND_dist')
    #       | (df['weights']=='dist_weights'))
    #       & (df['trial_subset']=='all_trials')
    #       & ((df['experiment_id']=='ASC16')
    #       | (df['experiment_id']=='ASC20')
    #       | (df['experiment_id']=='ASC28 cell 3')
    #       | (df['experiment_id']=='ASC22'))
    #       ], prefix = 'plot3Rs')

    plot_3_2(df[
           ((df['weights']=='democratic_weights')
           | (df['weights']=='size_weights')
           | (df['weights']=='size_AND_dist')
           | (df['weights']=='dist_weights'))
           & (df['trial_subset']=='all_trials')
           & (df['responsive_status']==True)
           ], prefix = 'plot3R')

    #plot_4(df[
    #           ((df['weights']=='democratic_weights')
    #           | (df['weights']=='resp_only')
    #           | (df['weights']=='unresp_only'))
    #           & (df['trial_subset']=='all_trials')
    #           ])

    plot_4_2(df[
               ((df['weights']=='democratic_weights')
               | (df['weights']=='resp_only')
               | (df['weights']=='unresp_only'))
               & (df['trial_subset']=='all_trials')
           & (df['responsive_status']==True)
           ], prefix = 'plot4Rs')

def plot_simulation_tuning_curves(df, prefix=''):

        ############
        #Plots and output - loop through the CSV and produce this
        ############
        #unpack the df
        df_dict = {}
        print(df.head())
        for index, row in df.iterrows():
            df_dict[index] = row
            print(row)

        #Tuning curve plots
        ############
        fig, axs = plt.subplots(nrows=4, ncols=1)

        linear_model_sub_dict = {k: df_dict[k] for k in ('soma','democratic', 'spine_size', 'distance_to_soma')}
        _, ax = plot_tuning_curves(axs[0], **linear_model_sub_dict)

        responsive_model_sub_dict = {k: df_dict[k] for k in ('soma','democratic', 'unresponsive', 'responsive')}
        _, ax = plot_tuning_curves(axs[1], **responsive_model_sub_dict)

        size_sub_dict = {k: df_dict[k] for k in ('soma','top_20_size', 'bottom_20_size', 'random_20')}
        _, ax = plot_tuning_curves(axs[2], **size_sub_dict)

        dist_sub_dict = {k: df_dict[k] for k in ('soma','top_20_distance', 'bottom_20_distance', 'random_20')}
        _, ax = plot_tuning_curves(axs[3], **dist_sub_dict)

        #Save the Plot
        figname = f"{prefix}{'_'*bool(prefix)}model_tuning_curves.png"
        fig_path = os.path.join(cfg.collect_summary_at_path, figname)
        print(f'Saving figure to {fig_path}')
        fig.savefig(fig_path, bbox_inches='tight')


        #Response timing plot
        ###########

def save_trace_image(traces, prefix='', ax=None):
    fig, ax = new_ax(ax)
    
    traces_name = f'{prefix}{"-"*bool(prefix)}_trace_array.npy'
    trace_path = os.path.join(cfg.collect_summary_at_path, 'traces', traces_name)
    print(f'Saving traces to {trace_path}')
    np.save(trace_path, np.array(traces))

    test_traces = np.load(trace_path)
    selected_timesteps = comp.select_timesteps(np.array(test_traces))
    new_idx = sort_within_all_stims(selected_timesteps, sort_by_mean_amp)#, plot.sort_by_mean_amp)
    ordered_traces = use_as_index(new_idx, selected_timesteps)

    ax.imshow(flatten_for_image(ordered_traces), aspect='auto')

    figname = f'{prefix}{"-"*bool(prefix)}_trace_image.png'
    fig_path = os.path.join(cfg.collect_summary_at_path, 'traces', figname)
    print(f'Saving figure to {fig_path}')
    fig.savefig(fig_path, bbox_inches='tight')