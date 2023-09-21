from src import data_io as io
from src import plotting as plot
from src import computation as comp
from src import helper_functions as hf
from src import config as cfg
import matplotlib.colors as cm


import xarray as xr
#import pandas as pd
import numpy as np
import ipdb
from matplotlib import pyplot as plt
#import seaborn as sns
import pandas as pd

from PIL import Image #this needs to be after matplotlib??
from scipy import stats  
from collections import defaultdict
import os
import traceback
import logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def init_globals():
    globals = defaultdict(dict)
    globals['failed_list']= []
    globals['data_dict_list'] = [] #Best practice is to create a list of dictionaries and then convert to df
    #see https://stackoverflow.com/questions/28056171/how-to-build-and-fill-pandas-dataframe-from-for-loop
    #and https://stackoverflow.com/questions/57000903/what-is-the-fastest-and-most-efficient-way-to-append-rows-to-a-dataframe
    return globals

def main_loop():
    globals = init_globals()
    
    if os.path.isdir(cfg.data_path) and cfg.walk_dirs:
        print('Walking to muleiple datasets')
        for current_data_dir, dirs, files in os.walk(cfg.data_path, topdown=False):
            for filename in files:
                filepath = os.path.join(current_data_dir, filename)
                unprocessed = True# not (cfg.subfolder_name in dirs)
                if cfg.re_run or unprocessed:
                    main(filepath, globals)
                
    else:
        print('Running on single dataset')
        main(cfg.data_path, globals)
    
    #io.save_summary_plots(globals)
    io.save_named_iterable_to_json(failed_dirs_list=globals['failed_list'])
    io.save_named_iterable_to_json(failed_dirs_errors=globals['errors'])
    
    df = pd.DataFrame(globals['data_dict_list'])
    io.save_csv(df, name_keywords='single_neuron_simulation_scores')


def tuning_curve_df_helper(model_keyword, means):
    result_list = [model_keyword]
    column_list = ['model_keyword (V), stimulus (->)']
    for i, xarray_mean in enumerate(means):

        #can add stim descriptions here
        #print(xarray_mean.Coordinates)
        column_list.append(float(xarray_mean.directions))#need to make this label more general
        result_list.append(float(xarray_mean.values))
    return column_list, result_list


def main(current_data_path, globals = None):
    if globals is None:
        globals = init_globals()
    
    
    print("Current data dir: " + str(current_data_path))
    try:
        #load soma data
        current_data_dir, filename = os.path.split(current_data_path)
        experiment_id, extension = os.path.splitext(filename)
        soma_path = current_data_path
        soma_data = io.loadmat(soma_path)
        if not(hf.include_soma(soma_data)):
            print("Excluding: " + str(current_data_path))
        else:
            print("Attempting to run adult spine models for data dir: " + str(current_data_path))
            responsive_status = hf.get_responsive_status(soma_data)
            print(f'responsive status: {hf.get_responsive_status(soma_data)}')
        
            #Soma tuning curve
            soma_traces = hf.get_traces(soma_data)
            #soma_means_normalized, soma_max_amplitude = comp.get_normalized_means(soma_traces)
            
            kyle_tuning_curve = hf.get_precomputed_tuning_curve(soma_data)
            soma_tuning_curve_normalized = comp.linear_normalization(kyle_tuning_curve)
            
            
            ############
            #Get everything ready to run all the models (lists for each parameter)
            ############
            ####load spine data
            spines_path = cfg.get_spines_path(soma_path)
            spine_data = io.loadmat(spines_path)


            #### Get diferenct trace matricies for each exclusion criteria
            included_trials_list = {
                'all_trials': comp.all_trials', 
                'baps_trials_only': comp.baps_trials_only, 
                'no_bap_trials' comp.no_bap_trials:
                }
            spine_activities_dict = {}
            spine_counts_dict = {}
            for inclusion_criteria, mask_func in included_trials_list.items():
                inclusion_type_spine_traces_dict[inclusion_criteria], spine_counts_dict[inclusion_criteria] = compile_spine_traces(spine_data, mask_func=mask_func)

            #### Get weight matricies for each model we want to test
            weight_functions_dict = {
                'dem_weights': comp.democratic_weights, 
                'size_weights': comp.weights_from_size_lin, 
                'dist_weights': comp.weights_from_dist_lin, 
                'resp_only': comp.responsive_spines_bin, 
                'unresp_only': comp.unresponsive_spines_bin
                }
            weight_matricies_dict['size_and_dist'] = weight_matricies_dict[weights_from_size_lin] * weight_matricies_dict[weights_from_dist_lin]
            weight_matricies_dict['size_and_dist_responsive_only'] = weight_matricies_dict['size_and_dist']*weight_matricies_dict[responsive_spines_bin]


            #### List the different integration models to use
            integration_models = [comp.linear_integration]


            #### List the different somatic output functions to use
            somatic_functions = [comp.somatic_identity]


            ############
            #Run the models (using the lists from the previous section)
            ############
            #a dictionary to put all the results in
            model_outputs = {}

            #### Run the model 1 simulated stimulus block at a time. 
            #There is probably a way to grab all the simulated trials at the same time using Numpy, this would be much faster
            for includedd_trial_types, spine_traces in inclusion_type_spine_traces_dict.items():
                subset_name = str(includedd_trial_types)
                spines_per_fov_list = spine_counts_dict[includedd_trial_types]
                for i in range(simulated_trials_per_stim):
                    simulated_trial_traces = sample_trial_from_fov(spine_traces, spines_per_fov_list)

                    for weight_name, weights in weight_matricies_dict.items()
                        #multiply by weights here
                        weighted_simulated_trial_traces = apply_weights(weights, simulated_trial_traces)

                        for integration_function in integration_models:
                            #apply integration model here
                            simulated_input_to_soma = integration_function(weighted_simulated_trial_traces)
                            #integration_func_name = 'lin_int'
                            integration_func_name = integration_function.__name__

                            for somatic_function in somatic_functions:
                                #apply somatic nonlinearity (if using) here
                                simulated_output_of_soma = somatic_function(simulated_input_to_soma)
                                #somatic_func_name = ''
                                somatic_func_name = somatic_function.__name__

                                full_model_name = f'{weight_name}-{subset_name}-{integration_func_name}-{somatic_func_name}'
                                try:
                                    type(model_outputs[full_model_name])
                                except KeyError as E:
                                    model_outputs[full_model_name] = comp.init_traces_xarray(spine_traces, simulated_trials_per_stim)
                                model_outputs[full_model_name][:,i,:] = simulated_output_of_soma


            ############
            #Compare the model outputs with the somatic tuning curve
            ############
            #We have to do this AFTER we have generated all traces (run through all simulated_trials_per_stim)
            #If we can eliminate this loop and use Numpy then we can do this directly at the end of the last loop
            #and won't need to save somatic traces from past models

            #Maybe also want to save PNG images here? since we don't keep the traces in any other form... 
            #I wonder how big it would be to save all of them
            #If we run 10k trials, 16 stims, 80 samples, *32 bytes per smaple thats 409 million... 400 mb? half a gig? pretty big but not impossible. might compress nicely. 

            save_plot(fig, current_data_dir)


            simulation_means_list = []
            df_row = ['soma']
            df_row.extend(list(soma_tuning_curve_normalized))
            print(df_row)
            simulation_means_list.append(df_row)
                
            soma_amps = comp.compute_normalized_trial_means(soma_traces)
            soma_amps_df = comp.convert_trial_amps_to_df(soma_amps, source='soma')

            #Anova loses power with unequal numbers...
            #if we actually wanted to boostrap this we could, wrap the above into a function, repeat it and then average the output values. 
            #not sure if this would actually be any better statsitically though...
            for model_keyword, model_traces in model_outputs.items():
                    model_tuning_curve_normalized, model_max_amplitude = comp.compute_normalized_tuning_curves(model_traces)
                    model_corr_to_soma = stats.pearsonr(soma_tuning_curve_normalized, model_tuning_curve_normalized)
                    model_similarity_score = comp.compare_tuning_curves(soma_means_normalized, means)

                    #This is where we can do the ANOVA too - make sure it can take different numbers in each group
                    model_amps = comp.compute_normalized_trial_means(model_traces)
                    model_amps_df = comp.convert_trial_amps_to_df(model_amps, source='model')

                    #Run anova with Model type, stimulus and each trial is a replicate
                    anova_sim = comp.compare_tuning_curves_anova(soma_amps_df, model_amps_df)

                #put the means into the model_dict list which will be saved as a seperate CSV for each cell
                ############
                
                column_list, result_list = tuning_curve_df_helper(model_keyword, model_tuning_curve_normalized)
                simulation_means_list.append(result_list)

                
                #Put the correlations into globals
                ############
                simulation_scores_dict = {
                    'experiment_id': experiment_id,
                    'model_type': model_keyword, 
                    'model_correlation_to_soma_r': model_corr_to_soma[0],
                    'model_correlation_to_soma_p': model_corr_to_soma[1],
                    'model_soma_similarity_score': model_similarity_score,
                    'responsive_status': responsive_status
                }
                
                globals['data_dict_list'].append(simulation_scores_dict)
                
            globals['data_dict_list'].append(simulation_scores_dict)

            #save the means for each simulation as a csv as well 
            df = pd.DataFrame(simulation_means_list, columns = column_list)
            name_str = f'{experiment_id}_simulation_mean_stim_response'
            io.save_csv(df, name_keywords=name_str)
            
    except Exception as E:
        globals['failed_list'].append(current_data_dir)
        err_str = f"There was an error processing data file: {current_data_path}"
        logger.error(err_str)
        logger.warning(traceback.format_exc())
        globals['errors'][current_data_dir] = traceback.format_exc()
        #raise(E)
    
    
if __name__ == "__main__":
    main_loop()
    
    