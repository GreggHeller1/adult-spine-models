from src import plotting as plot
from src import config as cfg
import os
import pandas as pd

def main_plotting_loop():
    for filename in os.listdir(cfg.collect_summary_at_path):
        if 'simulation_mean_stim_response' in filename:
            if False:
                print("###########################################################")
                print("###########################################################")
                print(f'Plotting tuning curves for {filename}')
                print("###########################################################")
                print("###########################################################")
                print('')
                exp = filename.split('_')[0]
                print(f'creating plots for {exp}')
                fullpath = os.path.join(cfg.collect_summary_at_path, filename)
                df = pd.read_csv(fullpath, index_col='model_keyword (V), stimulus (->)')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                print(df.columns)
                plot.plot_simulation_tuning_curves(df, prefix=exp)
        #if 'single_neuron_simulation_scores_10000' in filename:
        if 'single_neuron_simulation_scores_1000_01292024' in filename:
            pass
            print("###########################################################")
            print("###########################################################")
            print('Plotting summary plots across different model types')
            print("###########################################################")
            print("###########################################################")
            print('')
            fullpath = os.path.join(cfg.collect_summary_at_path, filename)
            df = pd.read_csv(fullpath, index_col=0)
            plot.plot_all_simulation_scores(df)

if __name__ == "__main__":
    main_plotting_loop()