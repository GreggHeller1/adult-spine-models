
import numpy as np
from src import data_io as io
from src import config as cfg


def get_example_traces(spine_data):
    ref = spine_data['dend_cell'][2,0]
    fov_field_2 = spine_data[ref]
    ex_spine_trace = np.array(fov_field_2['trial_traces'][:,:,0,:,0].swapaxes(0,-1))
    return ex_spine_trace

def get_summed_spine_trace(spine_data):
    #Get all the traces from all the spines
    #TODO could we make this a generator?

    ex_spine_trace = get_example_traces(spine_data)
    summed_spine_traces = np.zeros(ex_spine_trace.shape)
    for fov in spine_data['dend_cell'][2,:]:
        #ref = spine_data['dend_cell'][2,fov]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            this_spine_traces = get_traces(spine_data, fov=fov, spine_index=i)
            summed_spine_traces += this_spine_traces

    return summed_spine_traces


def get_summed_trial_sampled_spine_trace(spine_data):
    sampling_mat = trial_sampling(spine_data)
    #dimensions = stims x simulated_trials x FOVs.

    ex_spine_trace = get_example_traces(spine_data)
    stims = ex_spine_trace.shape[0]
    samples = ex_spine_trace.shape[-1]
    simulated_trials = sampling_mat.shape[1]
    num_fovs = sampling_mat.shape[-1]
    assert stims == sampling_mat.shape[0]

    summed_traces = np.zeros((stims, simulated_trials, samples))
    for stim_num in range(stims):
        for trail_num in range(simulated_trials):
            for fov_num in range(num_fovs):
                this_fov = get_fov(spine_data, fov=fov_num)
                spine_count = this_fov['trial_traces'].shape[-1]
                for i in range(spine_count):
                    #Might be faster not to go all the way back to the file here...
                    this_spine_traces = get_traces(spine_data, fov=fov_num, spine_index=i)
                    og_trial_num = sampling_mat[stim_num, trail_num, fov_num]
                    summed_traces[stim_num, trail_num, :] += this_spine_traces[stim_num, og_trial_num, :]
    return summed_traces





def test_MC_Pitts_model( soma_data, soma_act_thresh):
    #For now just minumize the pairwise difference between the two?
    pass


def plot_MC_Pitts_model():
    #sort by activity sorting within each direction (based off counts)


    #sort by same directional sorting as somas

    #plot both
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(flatten_for_image(bool_soma_activity))
    #axs[1].imshow(


def run_MC_Pitts_model(spine_data, spine_act_thresh, soma_count_thresh):

    #loop through each spine
    #TODO could we make this a generator?
    count_active_spines = np.zeros(spine_traces.shape)
    for fov in spines['dend_cell'][2,:]:
        ref = spines['dend_cell'][2,fov]
        fov_field_2 = spines[ref]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            this_spine_traces = np.array(fov_field_2['trial_traces'][:,:,0,:,i].swapaxes(0,-1))

            #select the time period means of interest
            this_spine_sub_traces = select_timesteps(soma_traces)
            this_spine_period_means = get_period_means(this_spine_sub_traces)

            #apply spine_act_threshold on each
            bool_input = int(this_spine_period_means>spine_act_thresh)

            #add to sum (bool will make it a count
            count_active_spines += bool_input

    #done looping
    #apply soma_count_thresh to get to bool
    simulated_soma_response =  int(count_active_spines>soma_count_thresh)
    return simulated_soma_response


def trial_sampling(spine_data):
    simulated_trials_per_stim = 10

    ex_spine_trace = get_example_traces(spine_data)
    stims = ex_spine_trace.shape[0]
    stim_repeats = ex_spine_trace.shape[1]
    fovs = spine_data['dend_cell'][2,:].shape[0]

    sampling_mat = np.random.randint(0,stim_repeats,(stims, simulated_trials_per_stim, fovs)) #dimensions = stims x simulated_trials x FOVs. each index is the og_trial - direction as assumed the same order
    return sampling_mat

def get_fov(activity_data_struct, fov=0):
    try:
        if fov%1==0:
            ref = activity_data_struct['dend_cell'][2,fov]
        else:
            ref = fov
    except Exception as E:
        ref = fov
    spine_field_2 = activity_data_struct[ref]#['DSI']
    return spine_field_2


def get_traces(activity_data_struct, fov=0, spine_index=0):
    try:
        this_fov = get_fov(activity_data_struct, fov=fov)
        spine_traces = np.array(this_fov['trial_traces'][:,:,0,:,spine_index].swapaxes(0,-1))
        traces = spine_traces

    except IndexError as E:
        soma_field_2 = io._todict(activity_data_struct[2])
        soma_traces = np.array(soma_field_2['trial_traces'])
        traces = soma_traces

    return traces




def get_most_similar_spine(soma_data, spine_data, ordering_func = cfg.return_as_is):

    #Should we be passing in the soma data and THEN subselecting? don't really see why not. may need other metadata at some point
    #^^ Now implemented

    soma_traces = get_traces(soma_data)
    soma_traces = select_timesteps(soma_traces)
    soma_activity_mat = ordering_func(soma_traces)

    similarity_list = []
    fov_num_list = []
    spine_num_list = []
    for j, fov in enumerate(spine_data['dend_cell'][2,:]):
        #ref = spine_data['dend_cell'][2,0]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            #this_spine_traces = np.array(fov_field_2['trial_traces'][:,:,0,:,i].swapaxes(0,-1))
            this_spine_traces = get_traces(spine_data, fov=fov, spine_index=i)
            this_spine_sub_traces = select_timesteps(this_spine_traces)
            this_spine_activity_mat = ordering_func(this_spine_sub_traces)

            similarity = np.dot(soma_activity_mat.flatten(), this_spine_activity_mat.flatten())
            #similarity = np.mean(soma_activity_mat - this_spine_activity_mat)

            similarity_list.append(similarity)
            fov_num_list.append(j)
            spine_num_list.append(i)

    similarity_list = np.array(similarity_list)
    fov_num_list = np.array(fov_num_list)
    spine_num_list = np.array(spine_num_list)
    ordering = np.argsort(similarity_list)[::-1]

    return similarity_list[ordering], fov_num_list[ordering], spine_num_list[ordering]



#####################################


def select_timesteps(traces):

    selected_timesteps = traces[:,:,cfg.first_sample_to_take:cfg.last_sample_to_take]
    return selected_timesteps


