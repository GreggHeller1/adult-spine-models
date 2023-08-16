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

    
def main(current_data_path, globals = None):
    if globals is None:
        globals = init_globals()
    
    
    print("Attempting to run adult spine models for data dir: " + str(current_data_path))
    try:
        #load soma data
        current_data_dir, filename = os.path.split(current_data_path)
        experiment_id, extension = os.path.splitext(filename)
        soma_path = current_data_path
        soma_data = io.loadmat(soma_path)
        
        responsive_status = hf.get_responsive_status(soma_data)
        print(f'responsive status: {hf.get_responsive_status(soma_data)}')
        
        #Soma tuning curve
        soma_traces = hf.get_traces(soma_data)
        soma_means_normalized, soma_max_amplitude = comp.get_normalized_means(soma_traces)
        
        kyle_means = hf.get_precomputed_tuning_curve(soma_data)
        kyle_means_normalized = comp.linear_normalization(kyle_means)
        
        
        ############
        #Run models on spine traces
        ############
        #load spine data
        spines_path = cfg.get_spines_path(soma_path)
        spine_data = io.loadmat(spines_path)

        #parameterizations of all the models we want to run
        model_dict = {
            'democratic': {'weight_function': comp.democratic_weights, 'subset':'all'},
            'spine_size': {'weight_function': comp.weights_size_lin, 'subset':'all'},
            'distance_to_soma': {'weight_function': comp.weights_distance_lin, 'subset':'all'},
            'unresponsive': {'weight_function': comp.democratic_weights, 'subset':'unresponsive'},
            'responsive': {'weight_function': comp.democratic_weights, 'subset':'responsive'},
            'top_20_size': {'weight_function': comp.top_20_size, 'subset':'all'},
            'bottom_20_size': {'weight_function': comp.bottom_20_size, 'subset':'all'},
            'top_20_distance': {'weight_function': comp.top_20_distance, 'subset':'all'},
            'bottom_20_distance': {'weight_function': comp.bottom_20_distance, 'subset':'all'},
            'random_20': {'weight_function': comp.random_20, 'subset':'all'},
        }
        
        #model_correlations_to_soma = {}
        #model_similarity_scores = {}
        #traces = defaultdict(dict)
        #traces[f'soma'] = soma_traces
        
        simulation_means_list = []
        means_normalized = defaultdict(dict)
        label_dict = {}
        label_dict['soma'] = f'soma, Max z score = {soma_max_amplitude}'
        means_normalized['soma'] = soma_means_normalized
        for model_keyword, model_params in model_dict.items():
            #boostrap here
            bootsraps = 1
            for i in range(bootsraps):
                traces, means, max_amplitude = comp.run_subset_model(spine_data, 
                                                            weight_function = model_params['weight_function'],
                                                            subset=model_params['subset'])

                model_corr_to_soma = stats.pearsonr(soma_means_normalized, means)
                model_similarity_score = comp.compare_tuning_curves(soma_means_normalized, means)

            #put the values in the dictionaries to be used in aggregate
            #traces[model_keyword] = traces
            #means_normalized[model_keyword] = means
            #label_dict[model_keyword] = f'{model_keyword}, Corr to soma = {model_corr_to_soma}'
            #model_correlations_to_soma[model_keyword] = model_corr_to_soma
            #model_similarity_scores[model_keyword] = model_similarity_score
            
            #put the means into the model_dict list which will be saved as a seperate CSV for each cell
            ############
            
            result_list = [model_keyword]
            column_list = ['model_keyword (V), stimulus (->)']
            for i, xarray_mean in enumerate(means):
                
                #can add stim descriptions here
                #print(xarray_mean.Coordinates)
                column_list.append(float(xarray_mean.directions))#need to make this label more general
                result_list.append(float(xarray_mean.values))
            simulation_means_list.append(result_list)

            
            #Put the correlations into globals
            ############
            simulation_scores_dict = {
                'experiment_id': experiment_id,
                'model_type': model_keyword, 
                'model_correlation_to_soma_r': model_corr_to_soma[0],
                'model_correlation_to_soma_p': model_corr_to_soma[1],
                'model_soma_similarity_score': model_similarity_score,
                'responsive_status': responsive_status #This seems a little out of place here... will be repeated a lot, doesn't change with model
                #maybe better to put them in two seperate dataframes? easier to just consider responsive and not have to filter out...but theres also an easy way to subset in pandas
            }
            
            globals['data_dict_list'].append(simulation_scores_dict)
            
        #save the means for each simulation as a csv as well 
        df = pd.DataFrame(simulation_means_list, columns = column_list)
        name_str = f'{experiment_id}_simulation_mean_stim_response'
        io.save_csv(df, name_keywords=name_str)
        
        #Democratic
        #democratic_model_all_traces = comp.get_summed_trial_sampled_spine_trace(spine_data)
        #sum_spine_sub_traces = comp.select_timesteps(summed_spine_traces_1)
        #^^ Deprecated, but may be useful for debugg
    except Exception as E:
        globals['failed_list'].append(current_data_dir)
        err_str = f"There was an error processing data file: {current_data_path}"
        logger.error(err_str)
        logger.warning(traceback.format_exc())
        globals['errors'][current_data_dir] = traceback.format_exc()
        #raise(E)
    
    
if __name__ == "__main__":
    main_loop()
    
    