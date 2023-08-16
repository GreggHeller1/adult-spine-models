

    #unpack the df

    ugh_this_is_gross = {}

    for index, row in df.iterrows():

    #bar plots with data points of the correlation for each model
    #this is sloppy and really should be a more generic plotting function but I'm rushing...
    #should remake this all working directly off of the dataframe (!!!!!)
    fig, ax = plt.subplots()
    count = 0
    #arbitrary_exp = single_neuron_simulation_scores['model_correlations_to_soma'][list(single_neuron_simulation_scores['model_correlations_to_soma'].keys())[0]]

    model_types = df['model_types'].unique()

    bar_locations = list(range(len(model_types)))
    num_cells = len(df[df['experiment_id'] == exp_ids[0]])

    colors = {
        'responsive': [ plt.get_cmap('autumn')(x) for x in np.linspace(0, 1, num_cells+1)],
        'unresponsive': [ plt.get_cmap('winter')(x) for x in np.linspace(0, 1, num_cells+1)]
    }


    sum_correlations = {
        'responsive': np.zeros(len(model_types)),
        'unresponsive': np.zeros(len(model_types))
    }
    counts = {
        'responsive': 0,
        'unresponsive': 0
    }
    for index, row in df.iterrows():
        experiment_id = row[experiment_id]
        data_list = []
        num_significant = 0
        responsiveness = row['resonsive_status']
        print(f'responsive status part 2: {responsiveness}')
        if responsiveness:
            responsive_key = 'responsive'
            color_val = colors[responsive_key][counts[responsive_key]]
            counts[responsive_key]+=1
        else:
            responsive_key = 'unresponsive'
            color_val = colors[responsive_key][counts[responsive_key]]
            counts[responsive_key]+=1
        for i, model_type in enumerate(model_types):
            correlation_value = model_dict[model_type][0]
            similarity_score = single_neuron_simulation_scores['model_similarity_scores'][experiment_id][model_type]
            p_value = model_dict[model_type][1]
            print(correlation_value)
            #data_list.append(correlation_value)
            data_list.append(similarity_score)
            sum_correlations[responsive_key][i] += similarity_score
            if p_value <.001:
                if num_significant >0:
                    ax.scatter(i, similarity_score+.005, marker='*', c=color_val)
                else:
                    ax.scatter(i, similarity_score+.005, marker='*', c=color_val, label=experiment_id)
                num_significant += 1

        count += 1
        if np.mean(np.array(data_list))>.5:
            ax.plot(bar_locations, data_list, color = color_val)
        else:
            ax.plot(bar_locations, data_list, color = color_val)


    for responsive_key, sums in sum_correlations.items():
        color_val = colors[responsive_key][counts[responsive_key]]
        mean_correlations = sums/counts[responsive_key]
        label_str = f'{responsive_key} mean similarity'
        ax.bar(bar_locations, mean_correlations, color = color_val, alpha = .5, label= label_str)

    ax.legend()
    ax.bar(bar_locations, mean_correlations)
    ax.set_xlabel('Model type')
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(model_types, rotation=-40, ha='left')
    ax.set_ylabel('Similarity score to soma tuning curve (dot product)')

    figname = 'model_performance_summaries.png'
    fig_path = os.path.join(cfg.collect_summary_at_path, figname)
    print(f'Saving figure to {fig_path}')
    fig.savefig(fig_path, bbox_inches='tight')

    #want to color/filter these by whether the neuron is responsive or not.
    #Indicate which ones are significant somehow

    ##Same fig but normalized to the best response
    fig, ax = plt.subplots()
    sum_correlations = {
        'responsive': np.zeros(len(model_types)),
        'unresponsive': np.zeros(len(model_types))
    }
    counts = {
        'responsive': 0,
        'unresponsive': 0
    }
    for experiment_id, model_dict in single_neuron_simulation_scores['model_correlations_to_soma'].items():
        data_list = []
        responsiveness = single_neuron_simulation_scores['responsive_status'][experiment_id]
        if responsiveness:
            responsive_key = 'responsive'
            color_val = colors[responsive_key][counts[responsive_key]]
            counts[responsive_key]+=1
        else:
            responsive_key = 'unresponsive'
            color_val = colors[responsive_key][counts[responsive_key]]
            counts[responsive_key]+=1
        #trying to get everything on the same page comparison wise I think this makes sense
        #max_similarity_score = 0
        #for i, model_type in enumerate(model_types):
        #    similarity_score = single_neuron_simulation_scores['model_similarity_scores'][experiment_id][model_type]
        #    max_similarity_score = max(similarity_score, max_similarity_score)
        dem_similarity_score = single_neuron_simulation_scores['model_similarity_scores'][experiment_id]['democratic']

        for i, model_type in enumerate(model_types):
            correlation_value = model_dict[model_type][0]
            similarity_score = single_neuron_simulation_scores['model_similarity_scores'][experiment_id][model_type]/dem_similarity_score
            p_value = model_dict[model_type][1]
            print(correlation_value)
            #data_list.append(correlation_value)
            data_list.append(similarity_score)
            sum_correlations[responsive_key][i] += similarity_score
            if p_value <.001:
                ax.scatter(i, similarity_score+.005, marker='*', c=color_val, label=experiment_id)

        count += 1
        if np.mean(np.array(data_list))>.5:
            ax.plot(bar_locations, data_list, color = color_val)
        else:
            ax.plot(bar_locations, data_list, color = color_val)

    for responsive_key, sums in sum_correlations.items():
        color_val = colors[responsive_key][counts[responsive_key]]
        mean_correlations = sums/counts[responsive_key]
        label_str = f'{responsive_key} mean similarity'
        ax.bar(bar_locations, mean_correlations, color = color_val, label= label_str)

    ax.legend()
    ax.set_xlabel('Model type')
    ax.set_xticks(bar_locations)
    ax.set_xticklabels(model_types, rotation=-40, ha='left')
    ax.set_ylabel('Normalized similarity score to soma tuning curve (dot product)')

    figname = 'model_performance_summaries_normalized.png'
    fig_path = os.path.join(cfg.collect_summary_at_path, figname)
    print(f'Saving figure to {fig_path}')
    fig.savefig(fig_path, bbox_inches='tight')


