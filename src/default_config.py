import numpy as np


#Config for main
######################

data_path = "/Users/Gregg/Dropbox (MIT)/2021 Gregg Sur rotation/ASC_experimental_data/2022-11 Soma Data"#r"/Users/Gregg/Dropbox (MIT)/2021 Gregg rotation/kyle_data"
#"/Users/Gregg/Dropbox (MIT)/2021 Gregg rotation/Katya_data/Problematic _FOVs/BM021_Cell6_p24/Dend4"#

def get_spines_path(data_path):
    return data_path.replace('Soma', 'Spine')

collect_summary_at_path = r"/Users/Gregg/Dropbox (MIT)/2021 Gregg Sur rotation/ASC_experimental_data/adult_spine_model_results"
#Images and plots will be placed here to easily perform a basic QC
#and ensure that that distance matricies reflect the desired values

subfolder_name = 'adult_spine_models'
#within each session directory, this subfolder will be created and contain
#the detailed output of the sucessful algorithm (distance matricies, and other measurements)

re_run = True
#This will skip any directories that already contain the dendritic distance subfolder named VVV

walk_dirs = True
#This will walk throught the lower level directories, useful to turn off for debugging

precision = '%1.3f'
#formatting string for numpy savetxt to determin how many decimals to include in the output csvs




#parameters for models/data
######################

fov_pixel_dim = np.array([512, 512])
pixel_size = .09 #microns/pixels


simulated_trials_per_stim = 100

soma_threshold = .5


start_s = -1 #number of seconds before the stim onset to begin the window of interest
end_s = 2  #(or 4 for viewing larger spread)    #number of seconds after the stim onset to end the window of interest
seconds_per_bin = 1.0  #defines the size of the bins that we want to predict

assert((end_s-start_s)%seconds_per_bin==0, 'The time range must be evenly divisible by the bin size')
frame_rate = 10  #frames per second
stim_start = 40 #the sample on which the stim comes on
stim_end = 50 #sample when the stim goes off

ms_per_frame = 1000/frame_rate

first_sample_to_take = stim_start+start_s*frame_rate
last_sample_to_take = stim_start+end_s*frame_rate


timepoints_per_period = int(seconds_per_bin*frame_rate)  #int(pre_post.shape[1]/num_tranges) <- might be safer to do it his way... will run into problems when not an integer multiple
num_tranges = int((last_sample_to_take - first_sample_to_take)/timepoints_per_period)




#Neuron Configuration
######################
nseg_density = 20  #segment per micron



