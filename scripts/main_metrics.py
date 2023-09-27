




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