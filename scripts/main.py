from src import data_io as io
from src import plotting as plot
from src import computation as comp
from src import helper_functions as hf
from src import config as cfg


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
    globals = {
    'failed_list': [],
    'errors': {},
    'model_correlations_to_soma': defaultdict(dict),
    }
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
    
    #bar plots with data points of the correlation for each model
    fig, ax = plt.subplots()
    count = 0
    arbitrary_exp = globals['model_correlations_to_soma'][list(globals['model_correlations_to_soma'].keys())[0]]
    model_types = list(arbitrary_exp.keys())
    sum_correlations = np.zeros(len(model_types))
    bar_locations = list(range(len(model_types)))
    for experiment_id, model_dict in globals['model_correlations_to_soma'].items():
        data_list = []
        for i, model_type in enumerate(model_types):
            correlation_value = model_dict[model_type][0]
            p_value = model_dict[model_type][1]
            print(correlation_value)
            data_list.append(correlation_value)
            sum_correlations[i] += correlation_value
            if p_value <.001:
                ax.scatter(i, correlation_value+.02, marker='*', c='k')
        
        count += 1
        if np.mean(np.array(data_list))>.5:
            ax.plot(bar_locations, data_list, label=experiment_id)
        else:
            ax.plot(bar_locations, data_list)
        
    ax.legend()
    mean_correlations = sum_correlations/count
    ax.bar(bar_locations, mean_correlations)
    ax.set_xlabel('Model type')
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(model_types, rotation=-40, ha='left')
    ax.set_ylabel('Correlation to measured soma tuning curve (r value)')
    
    figname = 'model_performance_summaries.png'
    fig_path = os.path.join(cfg.collect_summary_at_path, figname)
    print(f'Saving figure to {fig_path}')
    fig.savefig(fig_path, bbox_inches='tight')

    #want to color/filter these by whether the neuron is responsive or not. 
    #Indicate which ones are significant somehow 
    
        

    
def get_normalized_means(traces):
    trial_means = comp.compute_tuning_curve(traces)
    trial_means_normalized = comp.linear_normalization(trial_means)
    max_amp = np.max(trial_means)
    return trial_means_normalized, max_amp
    

def run_model(spine_data, spine_activity_array, spines_per_fov_list, weight_function):
    model_traces = comp.compute_model_output_from_random_sampled_fovs(spine_data, 
                                                                      spine_activity_array, 
                                                                      spines_per_fov_list, 
                                                                      simulated_trials_per_stim=cfg.simulated_trials_per_stim)
    model_means_normalized, model_max_amplitude = get_normalized_means(model_traces)
    return model_traces, model_means_normalized, model_max_amplitude


def run_subset_model(spine_data, weight_function = comp.democratic_weights, subset='all'):
    spine_activity_array, spines_per_fov_list = comp.compile_spine_traces(spine_data, subset = subset)
    return run_model(spine_data, spine_activity_array, spines_per_fov_list, weight_function = weight_function)



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
        
        #Soma tuning curve
        soma_traces = hf.get_traces(soma_data)
        soma_means_normalized, soma_max_amplitude = get_normalized_means(soma_traces)
        
        kyle_means = hf.get_precomputed_tuning_curve(soma_data)
        kyle_means_normalized = comp.linear_normalization(kyle_means)
        
        
        ############
        #Run models on spine traces
        ############
        #load spine data
        spines_path = cfg.get_spines_path(soma_path)
        spine_data = io.loadmat(spines_path)

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
        
        model_correlations_to_soma = defaultdict(dict)
        traces = defaultdict(dict)
        traces[f'soma'] = soma_traces
        
        means_normalized = defaultdict(dict)
        label_dict = {}
        label_dict['soma'] = f'soma, Max z score = {soma_max_amplitude}'
        means_normalized['soma'] = soma_means_normalized
        for model_keyword, model_params in model_dict.items():
            traces, means, max_amplitude = run_subset_model(spine_data, 
                                                            weight_function = model_params['weight_function'],
                                                            subset=model_params['subset'])
            
            model_corr_to_soma = stats.pearsonr(soma_means_normalized, means)
                                                                                
            #put the values in the dictionaries to be used in aggregate
            traces[model_keyword] = traces
            means_normalized[model_keyword] = means
            label_dict[model_keyword] = f'{model_keyword}, Corr to soma = {model_corr_to_soma}'
            model_correlations_to_soma[model_keyword] = model_corr_to_soma
            
        #Democratic
        #democratic_model_all_traces = comp.get_summed_trial_sampled_spine_trace(spine_data)
        #sum_spine_sub_traces = comp.select_timesteps(summed_spine_traces_1)
        #^^ Deprecated, but may be useful for debugging
                
        
        ############
        #Plots and output
        ############
        
        #Tuning curve plots
        ############
        fig, axs = plt.subplots(nrows=4, ncols=1)

        linear_model_sub_dict = {label_dict[k]: means_normalized[k] for k in ('soma','democratic', 'spine_size', 'distance_to_soma')}
        _, ax = plot.plot_tuning_curves(axs[0], **linear_model_sub_dict)
        
        responsive_model_sub_dict = {label_dict[k]: means_normalized[k] for k in ('soma','democratic', 'unresponsive', 'responsive')}
        _, ax = plot.plot_tuning_curves(axs[1], **responsive_model_sub_dict)
        
        size_sub_dict = {label_dict[k]: means_normalized[k] for k in ('soma','top_20_size', 'bottom_20_size', 'random_20')}
        _, ax = plot.plot_tuning_curves(axs[2], **size_sub_dict)
        
        dist_sub_dict = {label_dict[k]: means_normalized[k] for k in ('soma','top_20_distance', 'bottom_20_distance', 'random_20')}
        _, ax = plot.plot_tuning_curves(axs[3], **dist_sub_dict)
        
        #Save the Plot
        figname = experiment_id+'_model_tuning_curves.png'
        fig_path = os.path.join(cfg.collect_summary_at_path, figname)
        print(f'Saving figure to {fig_path}')
        fig.savefig(fig_path, bbox_inches='tight')
        
        
        #Response timing plot
        ############
        
        
        
        #Put the correlations into globals
        ############
        globals['model_correlations_to_soma'][experiment_id] = model_correlations_to_soma
        
        
        
    except Exception as E:
        globals['failed_list'].append(current_data_dir)
        err_str = f"There was an error processing data file: {current_data_path}"
        logger.error(err_str)
        logger.warning(traceback.format_exc())
        globals['errors'][current_data_dir] = traceback.format_exc()
        #raise(E)
    
    
if __name__ == "__main__":
    main_loop()
    
    