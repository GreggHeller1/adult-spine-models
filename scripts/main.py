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
                soma_df = 'ASC' in filename
                if soma_df and (cfg.re_run or unprocessed):
                    main(filepath, globals)
    else:
        print('Running on single dataset')
        main(cfg.data_path, globals)
    
    print("###########################################################")
    print("###########################################################")
    print("Finished running all models for all cells, saving aggregate data")
    print("###########################################################")
    #io.save_summary_plots(globals)
    
    df = pd.DataFrame(globals['data_dict_list'])
    io.save_csv(df, name_keywords='single_neuron_simulation_scores')

    io.save_named_iterable_to_json(failed_dirs_list=globals['failed_list'])
    io.save_named_iterable_to_json(failed_dirs_errors=globals['errors'])


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
    print("###########################################################")
    print("###########################################################")
    print("Current data dir: " + str(current_data_path))
    print("###########################################################")
    print("###########################################################")
    print("")
    print("")
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
        



            ############
            #Get everything ready to run all the models (lists for each parameter)
            ############
            print('Loading spine data and filtering traces')
            ####load spine data
            spines_path = cfg.get_spines_path(soma_path)
            spine_data = io.loadmat(spines_path)


            #### Get diferenct trace matricies for each exclusion criteria
            included_trials_list = {
                'all_trials': comp.all_trials,
            #    'baps_trials_only': comp.baps_trials_only, 
            #    'no_bap_trials': comp.no_bap_trials,
                }
            inclusion_type_spine_traces_dict = {}
            spine_counts_dict = {}
            for inclusion_criteria, mask_func in included_trials_list.items():
                inclusion_type_spine_traces_dict[inclusion_criteria], spine_counts_dict[inclusion_criteria] = comp.compile_spine_traces(spine_data, mask_func=mask_func)

            #### Get weight matricies for each model we want to test
            print("###########################################################")
            print('')
            print('Generating weight matricies for each model')
            weight_functions_dict = {
                'democratic_weights': comp.democratic_weights,
            #    'size_weights': comp.weights_from_size_lin,
            #    'dist_weights': comp.weights_from_distance_lin,
            #    'resp_only': comp.responsive_spines_bin,
            #    'unresp_only': comp.unresponsive_spines_bin
                }

            weight_matricies_dict = {}
            for weight_name, weight_function in weight_functions_dict.items():
                weight_matricies_dict[weight_name] = weight_function(spine_data)
            #weight_matricies_dict['size_AND_dist'] = weight_matricies_dict['size_weights'] * weight_matricies_dict['dist_weights']
            #weight_matricies_dict['size_dist_resp_only'] = weight_matricies_dict['size_AND_dist']*weight_matricies_dict['resp_only']


            #### List the different integration models to use
            integration_models = [comp.linear_integration]


            #### List the different somatic output functions to use
            somatic_functions = [comp.somatic_identity]


            ############
            #Run the models (using the lists from the previous section)
            ############
            print("###########################################################")
            print('')
            print('Begining to run the models')
            #a dictionary to put all the results in
            model_outputs = {}
            model_name_dict = {}

            #### Run the model 1 simulated stimulus block at a time. 
            #There is probably a way to grab all the simulated trials at the same time using Numpy, this would be much faster
            for j, (included_trial_types, spine_traces) in enumerate(inclusion_type_spine_traces_dict.items()):
                subset_name = str(included_trial_types)
                spines_per_fov_list = spine_counts_dict[included_trial_types]
                print(f"Running models on included trials: {included_trial_types}")
                for i in range(cfg.simulated_trials_per_stim):
                    simulated_trial_traces = comp.sample_trial_from_fov(spine_traces, spines_per_fov_list)
                    #TODO eventually should numpify this ^^^ loop....

                    for weight_name, weights in weight_matricies_dict.items():
                        #print(f"Using weight matrix: {weight_name}")
                        #multiply by weights here
                        weighted_simulated_trial_traces = comp.apply_weights(weights, simulated_trial_traces)

                        for integration_function in integration_models:
                            #apply integration model here
                            simulated_input_to_soma = integration_function(weighted_simulated_trial_traces)
                            #integration_func_name = 'lin_int'
                            integration_func_name = integration_function.__name__
                            #print(f"Using integration function: {integration_func_name}")

                            for somatic_function in somatic_functions:
                                #apply somatic nonlinearity (if using) here
                                simulated_output_of_soma = somatic_function(simulated_input_to_soma)
                                #somatic_func_name = ''
                                somatic_func_name = somatic_function.__name__
                                #print(f"Using somatic function: {somatic_func_name}")

                                full_model_name = f'{weight_name}-{subset_name}-{integration_func_name}-{somatic_func_name}'
                                nickname =  f'{weight_name}-{j}'
                                try:
                                    type(model_outputs[full_model_name])
                                except KeyError as E:
                                    model_outputs[full_model_name] = comp.init_traces_xarray(spine_traces, cfg.simulated_trials_per_stim)
                                model_outputs[full_model_name][:,i,:] = simulated_output_of_soma

                                model_name_dict[full_model_name] = {
                                    'weights': weight_name,
                                    'trial_subset': subset_name,
                                    'integration_function': integration_func_name,
                                    'somatic_function': somatic_func_name,
                                    'nickname': nickname
                                }


            #run_shuffles
            shuffle_scores_dict = {}
            if cfg.num_shuffles:
                shuffle_scores_list = []
                #Soma tuning curve retrived now so we only keep ths imilarity scores
                soma_traces = hf.get_traces(soma_data)
                #soma_means_normalized, soma_max_amplitude = comp.get_normalized_means(soma_traces)

                kyle_tuning_curve = hf.get_precomputed_tuning_curve(soma_data)
                soma_tuning_curve_normalized = comp.linear_normalization(kyle_tuning_curve)

                print('Running shuffles ')
                j=j+1
                ################################
                included_trial_types = 'all_trials'
                spine_traces = inclusion_type_spine_traces_dict[included_trial_types]
                subset_name = str(included_trial_types)
                spines_per_fov_list = spine_counts_dict[included_trial_types]
                print(f"Running shuffles on included trials: {included_trial_types}")
                for k in range(cfg.num_shuffles):
                    print(f'shuffle {k}')
                    #shuffle spine traces here
                    shuffled_spine_traces = comp.shuffle_spine_traces(spine_traces)
                    shuffled_spine_traces = xr.DataArray(shuffled_spine_traces,
                        coords={'spines': spine_traces['spines'], 'directions': spine_traces['directions'],'samples': spine_traces['samples']},
                        dims=['spines', "directions", "presentations", "samples"]
                        )
                    #this is spines x directions x presentations x samples
                    
                    shuffled_output_traces = comp.init_traces_xarray(spine_traces, cfg.simulated_trials_per_stim)

                    for i in range(cfg.simulated_trials_per_stim):
                        simulated_trial_traces = comp.sample_trial_from_fov(shuffled_spine_traces, spines_per_fov_list)
                        #TODO eventually should numpify this ^^^ loop....

                        ############################
                        weight_name = 'democratic_weights'
                        weights = weight_matricies_dict[weight_name]
                        #print(f"Using weight matrix: {weight_name}")
                        #multiply by weights here
                        weighted_simulated_trial_traces = comp.apply_weights(weights, simulated_trial_traces)

                        #############################
                        integration_function = comp.linear_integration
                        #apply integration model here
                        simulated_input_to_soma = integration_function(weighted_simulated_trial_traces)
                        #integration_func_name = 'lin_int'
                        integration_func_name = integration_function.__name__
                        #print(f"Using integration function: {integration_func_name}")

                        ##############################
                        somatic_function  = comp.somatic_identity
                        somatic_func_name = somatic_function.__name__
                        #apply somatic nonlinearity (if using) here
                        simulated_output_of_soma = somatic_function(simulated_input_to_soma)

                        shuffled_output_traces[:,i,:] = simulated_output_of_soma


                    model_tuning_curve_normalized, model_max_amplitude = comp.compute_normalized_tuning_curves(shuffled_output_traces)

                    #Compute various similarity metrics
                    ############
                    #Correlation
                    try:
                        model_corr_to_soma = stats.pearsonr(soma_tuning_curve_normalized, model_tuning_curve_normalized)
                    except ValueError as E:
                        globals['errors'][current_data_dir] = f'One or more of the model tuning curves seems to have been all NaNs: {full_model_name}'
                        model_tuning_curve_normalized = np.nan_to_num(model_tuning_curve_normalized, nan=0.0, posinf=0.0, neginf=0.0)
                    #Dot product based similarity score
                    model_similarity_score = comp.compare_tuning_curves(soma_tuning_curve_normalized, model_tuning_curve_normalized)

                    #now we need to put them in a dataframe so all 3 can be saved.

                    shuffle_scores_dict = {
                        'model_correlation_to_soma_r': model_corr_to_soma[0],
                        'model_correlation_to_soma_p': model_corr_to_soma[1],
                        'model_soma_similarity_score': model_similarity_score,
                        }
                    shuffle_scores_list.append(shuffle_scores_dict)
                df = pd.DataFrame(shuffle_scores_list)


                full_model_name = f'{weight_name}-{str(included_trial_types)}-{integration_func_name}-{somatic_func_name}'
                unshuffled_traces = model_outputs[full_model_name]
                unshuffled_curve, unshuffled_max_amplitude = comp.compute_normalized_tuning_curves(unshuffled_traces)
                unshuffled_model_score = comp.compare_tuning_curves(soma_tuning_curve_normalized, unshuffled_curve)
                shuffle_array = df.loc[:, 'model_soma_similarity_score']
                num_lower = (shuffle_array > unshuffled_model_score).sum()
                pvalue = num_lower/len(shuffle_array)

                io.save_csv(df, name_keywords=f'shuffle_scores_{experiment_id}_pvalue_{pvalue}')

       
            ############
            #Save the outputs
            ############
            #Maybe also want to save PNG images here? since we don't keep the traces in any other form...
            #I wonder how big it would be to save all of them
            #If we run 10k trials, 16 stims, 80 samples, *32 bytes per smaple thats 409 million... 400 mb? half a gig? pretty big but not impossible. might compress nicely.
            #... but when we have a higher sampling rate and more stimuli this will become pretty insustainable
            #would definitely need to be able to compress it.

            #but this will be a problem for RAM too. So need to be able to save it. Then we can memmap when we open.

            #oof but we need to do this for each cell. Would be nice to just have 1 file for each cell. or even 1 file for te whole thing
            #appropriate way to do this would be to add a dimensions for each model type and each cell
            #and havea seperate resource that maps the ordering
            #we should probably create it as an memmap to begin with
            #and then we fill it as we go, also savin the mapping file as we go

            #could also use an xarray, but seems like the memmaping is not as clean.

            #for model_keyword, model_traces in model_outputs.items():
            #    io.save_model_traces(model_traces, name_keywords=model_keyword)


            ############
            #Compare the model outputs with the somatic tuning curve
            ############
            #We have to do this AFTER we have generated all traces (run through all simulated_trials_per_stim)
            #If we can eliminate this loop and use Numpy then we can do this directly at the end of the last loop
            #and won't need to save somatic traces from past models

            #If we are doing the memmap the loop might actually be better or equialent because it keeps the size of those manageable
            print("###########################################################")
            print('')
            print(f"Computing similarity metrics for each model")

            #Soma tuning curve
            soma_traces = hf.get_traces(soma_data)
            #soma_means_normalized, soma_max_amplitude = comp.get_normalized_means(soma_traces)

            kyle_tuning_curve = hf.get_precomputed_tuning_curve(soma_data)
            soma_tuning_curve_normalized = comp.linear_normalization(kyle_tuning_curve)

            simulation_means_list = []
            df_row = ['soma']
            df_row.extend(list(soma_tuning_curve_normalized))
            #print(df_row)
            simulation_means_list.append(df_row)
                
            #We need each trial amp/mean in addition to the tuning curves. Calculating myself instead of taking
            #kyles because I don't want to throw any out, keep things consistent between the soma and the model.
            #don't really want an 0s or Nans if we can help it.
            soma_amps = comp.compute_normalized_trial_means(soma_traces)
            soma_amps_df = comp.convert_trial_amps_to_df(soma_amps, source='soma')

            #Anova loses power with unequal numbers...
            #if we actually wanted to boostrap this we could, wrap the above into a function, repeat it and then average the output values. 
            #not sure if this would actually be any better statsitically though. The limitation is still the number of soma trials.
            for full_model_name, model_traces in model_outputs.items():

                #save the traces image
                name = f'{experiment_id}_{full_model_name}_{cfg.simulated_trials_per_stim}'
                plot.save_trace_image(model_traces, prefix=name)
                soma_name = f'{experiment_id}_soma'
                plot.save_trace_image(soma_traces, prefix = soma_name)

                model_tuning_curve_normalized, model_max_amplitude = comp.compute_normalized_tuning_curves(model_traces)

                #put the means into the model_dict list which will be saved as a seperate CSV for each cell
                ############
                column_list, result_list = tuning_curve_df_helper(full_model_name, model_tuning_curve_normalized)
                simulation_means_list.append(result_list)

                #Compute various similarity metrics
                ############

                #Correlation
                try:
                    model_corr_to_soma = stats.pearsonr(soma_tuning_curve_normalized, model_tuning_curve_normalized)
                except ValueError as E:
                    globals['errors'][current_data_dir] = f'One or more of the model tuning curves seems to have been all NaNs: {full_model_name}'
                    model_tuning_curve_normalized = np.nan_to_num(model_tuning_curve_normalized, nan=0.0, posinf=0.0, neginf=0.0)
                #Dot product based similarity score
                model_similarity_score = comp.compare_tuning_curves(soma_tuning_curve_normalized, model_tuning_curve_normalized)

                #Anova - with Model type, stimulus and each trial is a replicate
                #model_amps = comp.compute_normalized_trial_means(model_traces)
                #model_amps_df = comp.convert_trial_amps_to_df(model_amps, source='model')

                #anova_sim = comp.compare_tuning_curves_anova(soma_amps_df, model_amps_df)
                
                #Put the correlations into globals
                ############
                simulation_scores_dict = {
                    'experiment_id': experiment_id,
                    'full_model_name': full_model_name,

                    'model_correlation_to_soma_r': model_corr_to_soma[0],
                    'model_correlation_to_soma_p': model_corr_to_soma[1],
                    'model_soma_similarity_score': model_similarity_score,
                    'responsive_status': responsive_status
                }
                simulation_scores_dict = dict(simulation_scores_dict, **model_name_dict[full_model_name])
                globals['data_dict_list'].append(simulation_scores_dict)
                
            print("###########################################################")
            print('')
            print(f"Saving the model similarity scores")
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
        raise(E)
    
    
if __name__ == "__main__":
    main_loop()
    
    