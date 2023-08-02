
import numpy as np
from src import config as cfg
from src import helper_functions as hf
import xarray as xr

def compile_spine_traces(spine_data, subset = 'all'):
    #idea is to put them all in a single array that can be sliced quickly
    #dimensison are spines x stims x trials x timestamps

    #we want to be able to slice this so that for a certain spine (all spines in an FOV)
    #we grab the Nth trial - all timstams and all stims
    #ideally we could do this in a single slice...
    #we will be able to do this using this notation
    #######
    #a = np.zeros((5, 16, 10, 91))
    #print(a.shape)
    ##lets try to grab the 1st presentation for spine 1 and the second for spine 2
    #sliced = a[np.arange(0, a.shape[0], 1), :,(0,2,4,6,7) , :]
    #print(sliced.shape)
    ########

    activity_list = []
    spines_per_fov_list = []
    spine_labels = []
    for i, (fov_activity_meta, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        fov_activity = np.array(fov_activity_meta['trial_traces'][:,:,0,:,:])

        fov_activity = fov_activity.swapaxes(0,-1)  #send samples to the last axes
        fov_activity = fov_activity.swapaxes(  1, 2) #bring directions in front of presentations
            #Now should be spines x directions x presentations x samples
        zeta_scores = hf.get_fov_zetas(fov_activity_meta)
        if i == 1:
            #zeta_scores[5] = 0 #ONly want to ignore this one for 26 cell 3...
            pass
        if 'unresponsive' in subset.lower():
            include_spines = np.where(zeta_scores > .05)[0]
        elif 'responsive' in subset.lower():
            include_spines = np.where(zeta_scores <= .05)[0]
        else:
            include_spines = np.arange(0,len(zeta_scores), 1) #indicies 0, 1, 2... N spines to subselect ALL spines
        print(include_spines)
        fov_activity_subset = fov_activity[include_spines, :, :, :]

        activity_list.extend(list(fov_activity_subset))

        spines_in_fov = len(fov_activity_subset)
        spines_per_fov_list.append(spines_in_fov)

        these_spine_labels = [f'fov_{i}_spine_{j}' for j in range(spines_in_fov)]

        spine_labels.extend(these_spine_labels)

    print('####')
    all_spine_activity_array = np.array(activity_list) #this is spines x directions x presentations x samples

    #here seems like a good time to make this into an xarray as well...
    #except it looks like xarray won't let me slice this way. So would have to go to numpy and back again...
    #spine_labels = ['fov_name'+str(i) for i in range(a.shape[0])]
    direction_labels = hf.get_direction_labels(spine_data, fov_num = 0)
    sample_labels = hf.get_trial_time_labels(spine_data, fov_num = 0)

    data_xr = xr.DataArray(all_spine_activity_array,
    coords={'spines': spine_labels,'directions': direction_labels,'samples': sample_labels},
    dims=["spines", "directions", "presentations", "samples"])
    ######
    return data_xr, spines_per_fov_list


def democratic_weights(spine_data, traces):
    return traces


def apply_weights(weights, traces):
    tile_dims = (len(traces['directions']), len(traces['samples']), 1)
    tiled_weights = np.tile(weights, tile_dims )
    tiled_weights = np.rollaxis(tiled_weights, -1)
    weighted_traces = traces*tiled_weights
    return weighted_traces

def weights_size_lin(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        size_vector.extend(list(fov_metadata['spine_size'])[0])
    size_array = np.array(size_vector)
    weight_array = weight_from_size(size_array)
    return apply_weights(weight_array, traces)

def weight_from_size(size_array):
    #y = mx+b
    b = .2
    m = .8
    max_val = np.max(size_array)
    return size_array/max_val*m+b


def top_20_size(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        size_vector.extend(list(fov_metadata['spine_size'])[0])
    size_array = np.array(size_vector)
    weight_array = binary_weights(size_array, 20)
    return apply_weights(weight_array, traces)

def bottom_20_size(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        size_vector.extend(list(fov_metadata['spine_size'])[0])
    size_array = np.array(size_vector)
    weight_array = binary_weights(size_array, -20)
    return apply_weights(weight_array, traces)

def binary_weights(size_array, threshold_percentage):
    total_spines = len(size_array)
    threshold_n = round(total_spines*threshold_percentage/100)
    ind = np.argpartition(size_array, threshold_n)[threshold_n:]

    weights = np.zeros(size_array.shape)
    weights[ind] = 1
    return weights

def weights_neck_len_lin(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        size_vector.extend(list(fov_metadata['spine_size'])[0])
    size_array = np.array(size_vector)
    weight_array = weight_from_size(size_array)
    return apply_weights(weight_array, traces)

def weight_from_neck_len(size_array):
    #y = mx+b
    b = 1
    m = -.8
    max_val = np.max(size_array)
    return size_array/max_val*m+b


def random_20(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        num_spines = hf.get_num_spines_in_fov(fov_metadata)
        fov_dist = hf.get_dist_from_root(spine_data, fov_num = i)
        dist_vect = [fov_dist]*num_spines
        size_vector.extend(dist_vect)
    size_array = np.array(size_vector)
    weight_array = random_binary_weights(size_array, 20)
    return apply_weights(weight_array, traces)

def random_binary_weights(size_array, threshold_percentage):
    total_spines = len(size_array)
    threshold_n = round(total_spines*threshold_percentage/100)
    ind = np.random.choice(total_spines, size=threshold_n, replace=False)
    weights = np.zeros(size_array.shape)
    weights[ind] = 1
    return weights



def top_20_distance(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        num_spines = hf.get_num_spines_in_fov(fov_metadata)
        fov_dist = hf.get_dist_from_root(spine_data, fov_num = i)
        dist_vect = [fov_dist]*num_spines
        size_vector.extend(dist_vect)
    size_array = np.array(size_vector)
    weight_array = binary_weights(size_array, 20)
    return apply_weights(weight_array, traces)


def bottom_20_distance(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        num_spines = hf.get_num_spines_in_fov(fov_metadata)
        fov_dist = hf.get_dist_from_root(spine_data, fov_num = i)
        dist_vect = [fov_dist]*num_spines
        size_vector.extend(dist_vect)
    size_array = np.array(size_vector)
    weight_array = binary_weights(size_array, -20)
    return apply_weights(weight_array, traces)


def weights_distance_lin(spine_data, traces):
    size_vector = []
    for i, (fov_activity, fov_metadata )in enumerate(hf.fov_generator(spine_data)):
        num_spines = hf.get_num_spines_in_fov(fov_metadata)
        fov_dist = hf.get_dist_from_root(spine_data, fov_num = i)
        dist_vect = [fov_dist]*num_spines
        size_vector.extend(dist_vect)
    size_array = np.array(size_vector)
    weight_array = weight_from_size(size_array)
    return apply_weights(weight_array, traces)

def weight_from_dist(size_array):
    #y = mx+b
    b = 1
    m = -.6
    max_val = np.max(size_array)
    return size_array/max_val*m+b



def linear_integration(traces):
    return np.sum(traces, axis=0) #should be directions x samples

def somatic_identity(traces):
    #This somatic function is pass through, it does not apply an additional linear or nonlinear normalization on top of the integration
    return traces


#Now we need a function to slice this array meaningfully
#basically we need a different random integer for each FOV
#mutliply that out to be an array of the right length
def compute_model_output_from_random_sampled_fovs(spine_data,
                                                all_spine_activity_array,
                                                spines_per_fov_list,
                                                simulated_trials_per_stim = 10,
                                                weight_function=democratic_weights,
                                                integration_function = linear_integration,
                                                somatic_function = somatic_identity,
                                                ):
    num_directions = len(all_spine_activity_array['directions'])
    num_samples= len(all_spine_activity_array['samples'])
    model_output = xr.DataArray(np.zeros((num_directions, simulated_trials_per_stim, num_samples)),
    coords={'directions': all_spine_activity_array['directions'],'samples': all_spine_activity_array['samples']},
    dims=["directions", "presentations", "samples"])

    for i in range(simulated_trials_per_stim):
        simulated_trial_traces = sample_trial_from_fov(all_spine_activity_array, spines_per_fov_list)

        #multiply by weights here
        weighted_simulated_trial_traces = weight_function(spine_data, simulated_trial_traces)

        #apply integration model here
        simulated_input_to_soma = integration_function(weighted_simulated_trial_traces)

        #apply somatic nonlinearity here
        simulated_output_of_soma = somatic_function(simulated_input_to_soma)

        model_output[:,i,:] = simulated_output_of_soma


    return model_output #should be directions x simulated_trials x samples (like the soma)




def sample_trial_from_fov(all_spine_activity_array, spines_per_fov_list):
    num_presentations = len(all_spine_activity_array['presentations'])
    #print(len(spines_per_fov_list))
    #draw a random integer for each fov
    rand_trial = np.random.randint(0, high=num_presentations, size=(len(spines_per_fov_list)))
    rand_trials_for_spines = list(np.repeat(rand_trial, spines_per_fov_list))
    spine_indicies = np.arange(0, len(all_spine_activity_array['spines']), 1)
    numpyfied_activity = np.array(all_spine_activity_array)
    simulated_trail_traces = numpyfied_activity[spine_indicies, :, rand_trials_for_spines, :]

    simulated_trail_traces = xr.DataArray(simulated_trail_traces,
    coords={'spines': all_spine_activity_array['spines'], 'directions': all_spine_activity_array['directions'],'samples': all_spine_activity_array['samples']},
    dims=["spines", "directions", "samples"])
    return simulated_trail_traces #should be spines x directions x samples




def get_summed_spine_trace_depracated(spine_data):
    #Get all the traces from all the spines
    #TODO could we make this a generator?

    ex_spine_trace = hf.get_example_traces(spine_data)
    summed_spine_traces = np.zeros(ex_spine_trace.shape)

    for fov in spine_data['dend_cell'][2,:]:
        #ref = spine_data['dend_cell'][2,fov]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]
        for i in range(spine_count):
            this_spine_traces = hf.get_traces(spine_data, fov=fov, spine_index=i)
            summed_spine_traces += this_spine_traces

    return summed_spine_traces


def get_summed_trial_sampled_spine_trace_depracated(spine_data):
    sampling_mat = trial_sampling(spine_data)
    #dimensions = stims x simulated_trials x FOVs.

    ex_spine_trace = hf.get_example_traces(spine_data)
    stims = ex_spine_trace.shape[0]
    samples = ex_spine_trace.shape[-1]
    simulated_trials = sampling_mat.shape[1]
    num_fovs = sampling_mat.shape[-1]
    assert stims == sampling_mat.shape[0]

    summed_traces = np.zeros((stims, simulated_trials, samples))
    for stim_num in range(stims):
        for trail_num in range(simulated_trials):
            for fov_num in range(num_fovs):
                this_fov = hf.get_fov(spine_data, fov=fov_num)
                spine_count = this_fov['trial_traces'].shape[-1]
                for i in range(spine_count):
                    #Might be faster not to go all the way back to the file here...
                    this_spine_traces = hf.get_traces(spine_data, fov=fov_num, spine_index=i)
                    og_trial_num = sampling_mat[stim_num, trail_num, fov_num]
                    summed_traces[stim_num, trail_num, :] += this_spine_traces[stim_num, og_trial_num, :]
    return summed_traces




def trial_sampling(spine_data):
    simulated_trials_per_stim = 10

    ex_spine_trace = hf.get_example_traces(spine_data)
    stims = ex_spine_trace.shape[0]
    stim_repeats = ex_spine_trace.shape[1]
    fovs = spine_data['dend_cell'][2,:].shape[0]

    sampling_mat = np.random.randint(0,stim_repeats,(stims, simulated_trials_per_stim, fovs)) #dimensions = stims x simulated_trials x FOVs. each index is the og_trial - direction as assumed the same order
    return sampling_mat


#####################################

def linear_normalization(array_in):
    zeroed_array = array_in#-min(array_in)
    return zeroed_array/max(zeroed_array)

def select_timesteps(traces, first_sample =cfg.first_sample_to_take, last_sample =cfg.last_sample_to_take):
    selected_timesteps = traces[:,:,first_sample:last_sample]
    return selected_timesteps

def get_stim_on_traces(traces):
    return select_timesteps(traces, first_sample =cfg.stim_start, last_sample =cfg.stim_end  )


def compute_tuning_curve(traces):
    on_period = get_stim_on_traces(traces)
    #on_period = on_period.reshape(on_period.shape[0], on_period.shape[1]*on_period.shape[2])
    try:
        means = on_period.mean(dim='samples')
        means = means.mean(dim='presentations')
    except TypeError as E:
        means = on_period.mean(axis=1)
        means = means.mean(axis=1)
    return means

def compare_tuning_curves(soma_traces, spine_traces):
    soma_means = compute_tuning_curve(soma_traces)
    spine_means = compute_tuning_curve(spine_traces)
    return np.dot(soma_means, spine_means)

#####################
#Neuron stuff



def compute_branch_order_and_dist(section, xyz_coords):
    order = 1
    dist = 0
    seg_fraction, seg_coords, _ = find_closest_segment(section, xyz_coords)
    dist_from_branch = section.L*seg_fraction
    dist+=dist_from_branch

    this_section = section
    for i in range(1000): #setting this to something reasonable for now, will avoid infinite loops
        parent  = this_section.parentseg().sec
        #print(parent.name())
        if 'soma' in parent.name():
            return order, dist, dist_from_branch
        order+=1
        dist += parent.L
        this_section = parent
    return order, dist, dist_from_branch


def get_branch_order_and_dist(h, spine_data, shifts_by_fov):
    stat_dict = {}
    for fov_num, fov in enumerate(spine_data['dend_cell'][2,:]):
        fov_name = hf.get_fov_name(spine_data, fov_num)
        print(fov_name)
        stat_dict[fov_name] = {}

        #current_input_dict[fov_num] = {}
        #ref = spine_data['dend_cell'][2,fov]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]

        #should probably make this more elegant, I have copy and pasted this motif...
        spines_pixel_coords = hf.get_spines_pixel_coords(spine_data, fov_num)
        spines_global_coords = np.zeros((spines_pixel_coords.shape[1], 3))
        for i in range(spines_pixel_coords.shape[1]):
            spines_global_coords[i,:] = hf.get_spine_global_coords(h, spine_data, fov_num, i)

        manual_shift = shifts_by_fov[fov_num]
        #print(manual_shift)
        if (manual_shift == np.array([0,0,0])).all():
            print('using inferred shift')
            optimal_shift = list(estimate_offset(h, spine_data, fov_num, max_shift=10, iterations=2))
            #optimal_shift.append(0)
            shift = np.array(optimal_shift)
        else:
            print('using manual shift')
            shift = manual_shift



        test_spines_global_coords = shift_coords(spines_global_coords, shift)

        dist_list = []
        order_list = []
        for i in range(spine_count):
            nearest_section, sec_coords, min_sec_dist = hf.find_closest_section(h, test_spines_global_coords[i,:])
            #sec_fraction, seg_coords, min_seg_dist = find_closest_segment(nearest_section, test_spines_global_coords[i,:])
            order, dist, dist_from_branch = compute_branch_order_and_dist(nearest_section, test_spines_global_coords[i,:])
            print(f'order: {order}, dist: {dist}')
            dist_list.append(dist)
            order_list.append(order)
            #iclamp = h.IClamp(nearest_section(sec_fraction))
            #current_input_dict[fov_num][i] = iclamp
            #pp.pprint(nearest_section.psection()['point_processes'])
            #raise
        stat_dict[fov_name]['distance_to_soma'] = dist_list
        stat_dict[fov_name]['branch_order'] = order_list
    return stat_dict






def estimate_offset(h, spine_data, fov_num, max_shift = 10, iterations = 2, verbose = False):
    spines_pixel_coords = hf.get_spines_pixel_coords(spine_data, fov_num)

    coords_array, section_list = generate_section_mapped_3dpoints(h)


    spines_global_coords = np.zeros((spines_pixel_coords.shape[1], 3))
    for i in range(spines_pixel_coords.shape[1]):
        spines_global_coords[i,:] = hf.get_spine_global_coords(h, spine_data, fov_num, i)


    sections_to_search = []
    #I think its now running fast enough that we can just do it all again... but this is a bit cleaner and doesn't require passing back out the section names
    shift_idxs_to_test = [[-max_shift,-max_shift], [max_shift,-max_shift], [-max_shift,max_shift], [-max_shift,-max_shift], [0, 0]]
    for (x_shift, y_shift) in shift_idxs_to_test:
        shift = np.array([x_shift, y_shift, 0])
        test_spines_global_coords = spines_global_coords+np.tile(shift, (spines_global_coords.shape[0], 1))

        for i in range(test_spines_global_coords.shape[1]):
            spine_global_coords = test_spines_global_coords[i,:]
            nearest_section, sec_coords, min_dist  = find_closest_section_fast(coords_array, section_list, spine_global_coords)
            if not (nearest_section in sections_to_search):
                sections_to_search.append(nearest_section)

    print(f'Searching within these sections {sections_to_search}')


    # Did this to try to refine, make it use segment coordinates instead of section coordinates.
    full_coords_list = []
    full_section_list = []
    location_within_section_list = []
    for section in sections_to_search:
        coords_array, section_list, location_within_section = generate_segment_mapped_3dpoints(section)
        full_coords_list.extend(list(coords_array.T))
        full_section_list.extend(section_list)
        location_within_section_list.extend(location_within_section)
    all_segment_coords = np.array(full_coords_list)
    #print(f'####{all_segment_coords.shape}')
    #all_segment_coords = coords_array
    #full_section_list = section_list


    def find_optimal_shift(max_shift, iter_num, base_shift, all_segment_coords, full_section_list, verbose = verbose):
        if verbose:
            print(f'on iteration {iter_num}')
        cum_dist_at_shift = np.zeros((2*max_shift, 2*max_shift))

        # now implemented faster mechanism so don't have to regrab global spine coords

        for x_shift in range(-max_shift,max_shift):
            #print(x_shift)
            for y_shift in range(-max_shift,max_shift):
                this_x_shift =  x_shift/(10**iter_num)
                this_y_shift =  y_shift/(10**iter_num)
                manual_adjustment = np.array([this_x_shift+base_shift[0], this_y_shift+base_shift[1], 0])
                #print(manual_adjustment)

                cum_dist_at_shift[x_shift+max_shift, y_shift+max_shift] = find_distance_for_shift(all_segment_coords, full_section_list, spines_global_coords, manual_adjustment)
        #print(cum_dist_at_shift)

        #TODO if we wanted to make this better we could impose a cost here - don't want to move far distances for small gains
        #so scale the distance by the shift. But would require a lot of finnicky tweaking I think...

        optimal_shift = np.unravel_index(cum_dist_at_shift.argmin(), cum_dist_at_shift.shape)
        optimal_shift_relative = (np.array(optimal_shift) - np.array([max_shift, max_shift]))/(10**iter_num)+base_shift
        if verbose:
            print(f'Optimal shift is: {optimal_shift_relative}')
            print(f'Minimized cumulative distance is: {cum_dist_at_shift.min()}')
        return optimal_shift_relative

    base_shift = np.array([0,0])
    for i in range(iterations):
        base_shift = find_optimal_shift(max_shift, i, base_shift, all_segment_coords, full_section_list)
    optimal_shift = list(base_shift)
    optimal_shift.append(0)
    optimal_shift = np.array(optimal_shift)
    if verbose:
        check_min_dist = find_distance_for_shift(all_segment_coords, full_section_list, spines_global_coords, optimal_shift, verbose=verbose)
        print(f'Optimal shift is: {optimal_shift}')
        print(f'Minimized cumulative distance is: {check_min_dist}')
    return optimal_shift


def generate_section_mapped_3dpoints(h):
    coords_list = []
    section_list = []
    for section in h.allsec():
        points_list = section.psection()['morphology']['pts3d']
        coords_list.extend(points_list)
        this_sec_list = [section]*len(points_list)
        section_list.extend(this_sec_list)
    coords_array = np.array(coords_list)
    coords_array = coords_array[:, :3] #remove the radius

    return coords_array, section_list



def find_closest_section_fast(coords_array, section_list, xyz_coords):
    tiled_xyz_coords = np.tile(xyz_coords, (max(coords_array.shape),1))
    assert(tiled_xyz_coords.shape[1] == 3)
    diffs = coords_array - tiled_xyz_coords
    dists = np.linalg.norm(diffs, axis=1)

    nearest_section = section_list[dists.argmin()]
    nearest_coords = coords_array[dists.argmin(),:]
    return nearest_section, nearest_coords, dists.min()


def generate_segment_mapped_3dpoints(section):
    xCoord, yCoord, zCoord = hf.returnSegmentCoordinates(section)

    all_seg_coords = np.array([xCoord, yCoord, zCoord])
    section_list = [section]*len(xCoord)
    location_within_section = np.linspace(0,1,section.nseg)
    return all_seg_coords, section_list, location_within_section




def find_closest_segment(section, xyz_coords):
    xCoord, yCoord, zCoord = hf.returnSegmentCoordinates(section)

    all_seg_coords = np.array([xCoord, yCoord, zCoord])
    #print(seg_coords.shape)
    find_coords = np.tile(xyz_coords, (len(xCoord), 1)).T
    #print(find_coords.shape)
    dists = np.linalg.norm(all_seg_coords - find_coords, axis=0)
    #print(dists.shape)
    nearest_segment_i = dists.argmin()
    min_dist = dists.min()
    seg_coords = all_seg_coords[:,nearest_segment_i]

    #for i, (x,y,z) in enumerate(zip(xCoord, yCoord, zCoord)):
    #    dist = np.linalg.norm(xyz_coords - np.array([x,y,z])) #last point is radius
    #    try:
    #        if min_dist> dist:
    #            min_dist = dist
    #            nearest_segment_i = i
    #            seg_coords =  np.array([x,y,z])
    #    except UnboundLocalError as E:
    #        min_dist = dist
    #        nearest_segment_i = i
    #        seg_coords =  np.array([x,y,z])
    seg_fraction = nearest_segment_i/(section.nseg-1)
    return seg_fraction, seg_coords, min_dist,




def total_distance(coords_array, section_list, global_coords, verbose=False):
    total_distance = 0
    for i in range(global_coords.shape[0]):
        spine_global_coords = global_coords[i,:]
        nearest_section, sec_coords, min_dist  = find_closest_section_fast(coords_array, section_list, spine_global_coords)
        if verbose:
            print(f'spine num: {i}, spine coords: {spine_global_coords}')
            print(f'section: {nearest_section}, section_coords: {sec_coords}, min_sec_dist: {min_dist}')
        total_distance += min_dist
    #print(f'total_distance: {total_distance}')
    return total_distance


def find_distance_for_shift(coords_array, section_list, global_coords, apply_shift, verbose=False):
    if verbose:
        print(f'Calculating distance for all spines with a shift of {apply_shift}!!!!!')
    shifted_global_coords = global_coords - np.tile(apply_shift, (global_coords.shape[0], 1))
    return total_distance(coords_array, section_list, shifted_global_coords, verbose=verbose)


def find_distance_for_shift_slow(h, spine_data, fov_num, shift_coords, verbose=False):
    cumulative_distance=0
    spines_pixel_coords = hf.get_spines_pixel_coords(spine_data, fov_num)

    coords_array, section_list = generate_section_mapped_3dpoints(h)

    spines_global_coords = np.zeros((spines_pixel_coords.shape[1], 3))
    for i in range(spines_pixel_coords.shape[1]):
        spines_global_coords[i,:] = hf.get_spine_global_coords(h, spine_data, fov_num, i)

    cumulative_dist_fast_method = find_distance_for_shift(coords_array, section_list, spines_global_coords, shift_coords, verbose=verbose)
    if verbose:
        print(f'Cumulative distance from fast method: {cumulative_dist_fast_method}')


    for i in range(spines_pixel_coords.shape[1]):
        spines_global_coords[i,:] = hf.get_spine_global_coords(h, spine_data, fov_num, i, shift_coords)
    for i in range(spines_pixel_coords.shape[1]):
        nearest_section, sec_coords, min_dist = hf.find_closest_section(h, spines_global_coords[i,:])
        seg_fraction, seg_coords, min_dist = find_closest_segment(nearest_section, spines_global_coords[i,:])
        if verbose:
            print(f'Spine {i} is {min_dist} microns from section {nearest_section}[{seg_fraction}]')
        cumulative_distance+= min_dist
    return cumulative_distance



def shift_coords(coords_array, shift):
    return coords_array+np.tile(shift, (coords_array.shape[0], 1))


def create_current_sources(h, spine_data, shifts_by_fov):
    current_input_dict = {}
    for fov_num, fov in enumerate(spine_data['dend_cell'][2,:]):
        current_input_dict[fov_num] = {}
        #ref = spine_data['dend_cell'][2,fov]
        fov_field_2 = spine_data[fov]#['DSI']
        spine_count = fov_field_2['trial_traces'].shape[-1]

        #should probably make this more elegant, I have copy and pasted this motif...
        spines_pixel_coords = hf.get_spines_pixel_coords(spine_data, fov_num)
        spines_global_coords = np.zeros((spines_pixel_coords.shape[1], 3))
        for i in range(spines_pixel_coords.shape[1]):
            spines_global_coords[i,:] = hf.get_spine_global_coords(h, spine_data, fov_num, i)

        manual_shift = shifts_by_fov[fov_num]
        #print(manual_shift)
        if (manual_shift == np.array([0,0,0])).all():
            print('using inferred shift')
            optimal_shift = list(estimate_offset(h, spine_data, fov_num, max_shift=10, iterations=2))
            #optimal_shift.append(0)
            shift = np.array(optimal_shift)
        else:
            print('using manual shift')
            shift = manual_shift

        test_spines_global_coords = shift_coords(spines_global_coords, shift)

        for i in range(spine_count):
            nearest_section, sec_coords, min_sec_dist = hf.find_closest_section(h, test_spines_global_coords[i,:])
            sec_fraction, seg_coords, min_seg_dist = find_closest_segment(nearest_section, test_spines_global_coords[i,:])

            iclamp = h.IClamp(nearest_section(sec_fraction))
            current_input_dict[fov_num][i] = iclamp
            #pp.pprint(nearest_section.psection()['point_processes'])
            #raise
    return current_input_dict


##################
#Probably doing this a different way


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




