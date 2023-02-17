
import numpy as np

from src import config as cfg



def get_summed_spine_trace(spine_data):
    #Get all the traces from all the spines
    #TODO could we make this a generator?
    ref = spine_data['dend_cell'][2,0]
    fov_field_2 = spine_data[ref]
    ex_spine_trace = np.array(fov_field_2['trial_traces'][:,:,0,:,0].swapaxes(0,-1))

    summed_spine_traces = np.zeros(ex_spine_trace.shape)
    for fov in spine_data['dend_cell'][2,:]:
        #ref = spine_data['dend_cell'][2,fov]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            this_spine_traces = np.array(fov_field_2['trial_traces'][:,:,0,:,i].swapaxes(0,-1))
            summed_spine_traces += this_spine_traces
    return summed_spine_traces


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
        ref = spines['dend_cell'][2,0]
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



def get_most_similar_spine(soma_sub_traces, spine_data):
    soma_activity_vect = soma_sub_traces.flatten()
    max_dot = 0
    best_match_traces = None
    best_match_FOV = None
    best_match_idx = None
    for fov in spine_data['dend_cell'][2,:]:
        #ref = spine_data['dend_cell'][2,0]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            this_spine_traces = np.array(fov_field_2['trial_traces'][:,:,0,:,i].swapaxes(0,-1))
            this_spine_sub_traces = select_timesteps(this_spine_traces)
            this_spine_activity_vect = this_spine_sub_traces.flatten()
            dot = np.dot(soma_activity_vect, this_spine_activity_vect)
            if dot>max_dot:
                max_dot = dot
                best_match_traces = this_spine_sub_traces
                best_match_FOV = fov
                best_match_idx = i

    return best_match_traces, best_match_FOV, best_match_idx, max_dot



#####################################


def select_timesteps(traces):

    selected_timesteps = traces[:,:,cfg.first_sample_to_take:cfg.last_sample_to_take]
    return selected_timesteps


