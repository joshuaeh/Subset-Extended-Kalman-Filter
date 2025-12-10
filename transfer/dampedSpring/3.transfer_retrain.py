"""retrain NN on target system data, randomly initializing weights each time."""

# imports
import ray
from dampedSpring import *

# constants

# script
# if __name__ == "__main__":
#     initialization_dir = RESULTS_DIR.joinpath("transfer", "finetune")
#     initialization_dir.mkdir(parents=True, exist_ok=True)
#     for scenario in SCENARIOS:
#         scenario_name = transfer_scenario_name(scenario)
#         scenario_dir = initialization_dir.joinpath(scenario_name)
#         scenario_dir.mkdir(parents=True, exist_ok=True)
#         for data_dim in TARGET_DATA_DIM:
#             data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
#             data_dim_dir.mkdir(parents=True, exist_ok=True)
#             for data_iteration in range(N_ITERATIONS):
#                 data_iteration_dir = data_dim_dir.joinpath(
#                     get_data_iteration_name(data_iteration)
#                 )
#                 data_iteration_dir.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
            
    ASHA_PARAMS = {
        "time_attr": "training_iteration",
        "max_t": 100,
        # "grace_period": 0,
        "reduction_factor": 4,
    }
    
    TUNE_CONFIG = {
        "metric": "val_loss",
        "mode": "min",
        "max_concurrent_trials": 4,
        "num_samples": 50,
    }
    
    RUN_CONFIG = {
        "verbose": 1,
        "storage_path": "/tmp/",
    }
    
    CHECKPOINT_CONFIG = {
        "checkpoint_score_attribute": "val_loss",
        "checkpoint_score_order": "min",
        "num_to_keep": 1,
        "checkpoint_frequency": 100,
        "checkpoint_at_end": True,
    }
    
    
    initialization_dir = RESULTS_DIR.joinpath("transfer", "retrain")
    initialization_dir.mkdir(parents=True, exist_ok=True)
    
    # --- check how many are complete ---
    n_complete = 0
    for data_iteration in range(N_ITERATIONS):
        for scenario in SCENARIOS:
            scenario_name = transfer_scenario_name(scenario)
            scenario_dir = initialization_dir.joinpath(scenario_name)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for data_dim in TARGET_DATA_DIM:
                data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
                data_dim_dir.mkdir(parents=True, exist_ok=True)

                data_iteration_dir = data_dim_dir.joinpath(
                    get_data_iteration_name(data_iteration)
                )
                data_iteration_dir.mkdir(parents=True, exist_ok=True)
                
                for method in ["adam", "sekf", "lbfgs"]:
                    method_dir = data_iteration_dir.joinpath(method)
                    method_dir.mkdir(parents=True, exist_ok=True)
                    
                    all_trials_path = method_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                    best_result_path = method_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                    model_weights_path = method_dir.joinpath(MODEL_FILENAME)
                    ray_results_path = method_dir.joinpath("ray_results.zip")
                    if all(
                        [
                            all_trials_path.exists(),
                            best_result_path.exists(),
                            model_weights_path.exists(),
                            ray_results_path.exists(),
                        ]
                    ):
                        n_complete += 1
    print(f"\n\n\nNumber of complete runs: {n_complete}\n\n\n")
                        
    # --- end: check how many are complete ---
    for data_iteration in range(N_ITERATIONS):
        for scenario in SCENARIOS:
            scenario_name = transfer_scenario_name(scenario)
            scenario_dir = initialization_dir.joinpath(scenario_name)
            scenario_dir.mkdir(parents=True, exist_ok=True)
            for data_dim in TARGET_DATA_DIM:
                data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
                data_dim_dir.mkdir(parents=True, exist_ok=True)

                data_iteration_dir = data_dim_dir.joinpath(
                    get_data_iteration_name(data_iteration)
                )
                data_iteration_dir.mkdir(parents=True, exist_ok=True)
                ### Data
                data_indices = get_data_indices(data_dim, data_iteration)
                x, y = get_transfer_data(scenario, indices=data_indices)
                x += rng.normal(0, MEASUREMENT_NOISE_STD, x.shape)
                y += rng.normal(0, MEASUREMENT_NOISE_STD, y.shape)
                x_train, y_train, x_validation, y_validation, x_test, y_test = (
                    train_val_test_split(
                        x,
                        y,
                        n_train=int(0.9 * data_dim),
                        n_validation=data_dim - int(0.9 * data_dim),
                        n_test=9_000,
                    )
                )
                data = {
                    "train_x": x_train,
                    "train_y": y_train,
                    "val_x": x_validation,
                    "val_y": y_validation,
                    "test_x": x_test,
                    "test_y": y_test,
                }
                
                data_ref = ray.put(data)
                
                # experiment stopper if no improvement after 

                ### Adam
                method_dir = data_iteration_dir.joinpath("adam")
                method_dir.mkdir(parents=True, exist_ok=True)
                all_trials_path = method_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                best_result_path = method_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                model_weights_path = method_dir.joinpath(MODEL_FILENAME)
                ray_results_path = method_dir.joinpath("ray_results.zip")
                if all(
                    [
                        all_trials_path.exists(),
                        best_result_path.exists(),
                        model_weights_path.exists(),
                        ray_results_path.exists(),
                    ]
                ):
                    print(
                        f"Skipping existing results for {method_dir}, delete directory to rerun"
                    )
                else:
                    config = {
                        "lr": tune.loguniform(1e-6, 1e-1),
                        "batch_size": tune.qlograndint(1, data_dim, 2),
                        "N_batches_per_step": 10,
                        "lr_patience": 10,
                        "lr_factor": tune.uniform(0.0, 1.0),
                        # "mask_fn_quantile_thresh": tune.quniform(0.0, 1.0, 0.05),
                        "initialize_weights": "random",
                        "data_ref": data_ref,
                    }
                    _ASHA_PARAMS = ASHA_PARAMS.copy()
                    _ASHA_PARAMS["max_t"] = 50
                    scheduler = ASHAScheduler(**ASHA_PARAMS)

                    tuner = tune.Tuner(
                        tune.with_resources(
                            DampedSpringTrainer,
                            resources={"cpu": 1},
                        ),
                        tune_config=tune.TuneConfig(
                            scheduler=scheduler,
                            **TUNE_CONFIG
                        ),
                        param_space=config,
                        run_config=tune.RunConfig(
                            name=f"dampedSpring_transfer_adam_retrain_{scenario_name}_{data_dim_name(data_dim)}_{get_data_iteration_name(data_iteration)}",
                            # storage_path=RAY_STORAGE_PATH,
                            checkpoint_config=tune.CheckpointConfig(
                                **CHECKPOINT_CONFIG
                            ),
                            **RUN_CONFIG
    
                        ),
                    )
                    results = tuner.fit()

                    best_result = results.get_best_result("val_loss", "min")

                    # save metrics csvs
                    print(f"Best trial config: {best_result.config}")
                    print(f"Best trial final validation loss: {best_result.metrics}")
                    metrics_df = results.get_dataframe()
                    metrics_df.to_csv(all_trials_path)
                    best_result_df = best_result.metrics_dataframe
                    best_result_df.to_csv(best_result_path)
                    # move model weights to standard location
                    print(f"{best_result.path=}")
                    print(f"{best_result.checkpoint=}")
                    with best_result.checkpoint.as_directory() as checkpoint_dir:
                        model_path = Path(checkpoint_dir).joinpath(MODEL_FILENAME)
                        shutil.copy(str(model_path), str(model_weights_path))

                    # compress all results directories and move to case_dir
                    zip_dir(
                        results.experiment_path,
                        ray_results_path,
                    )

                ### SEKF
                method_dir = data_iteration_dir.joinpath("sekf")
                method_dir.mkdir(parents=True, exist_ok=True)
                all_trials_path = method_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                best_result_path = method_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                model_weights_path = method_dir.joinpath(MODEL_FILENAME)
                optimizer_path = method_dir.joinpath(OPTIMIZER_FILENAME)
                ray_results_path = method_dir.joinpath("ray_results.zip")
                if all(
                    [
                        all_trials_path.exists(),
                        best_result_path.exists(),
                        model_weights_path.exists(),
                        ray_results_path.exists(),
                    ]
                ):
                    print(
                        f"Skipping existing results for {method_dir}, delete directory to rerun"
                    )
                else:
                    config = {
                        # "R": tune.choice([0, 0.01, 0.05, 0.1, 0.5, 1.0]),
                        "R": 0.01,
                        "Q": tune.choice([1e-6, 1e-4, 1e-2, 1e-1]),
                        "p0": tune.choice([0.01, 1.0, 10.0, 100.0]),
                        "batch_size": tune.choice([1, 2, 4, 8]),
                        "N_batches_per_step": 10,
                        # "mask_fn_quantile_thresh": tune.quniform(0.05, 1.0, 0.05),
                        "initialize_weights": "random",
                        "data_ref": data_ref,
                    }

                    scheduler = ASHAScheduler(
                        **ASHA_PARAMS
                    )

                    tuner = tune.Tuner(
                        tune.with_resources(
                            DampedSpringTrainer_SEKF,
                            resources={"cpu": 1},
                        ),
                        tune_config=tune.TuneConfig(
                            scheduler=scheduler,
                            **TUNE_CONFIG
                        ),
                        param_space=config,
                        run_config=tune.RunConfig(
                            name=f"dampedSpring_transfer_sekf_retrain_{scenario_name}_{data_dim_name(data_dim)}_{get_data_iteration_name(data_iteration)}",
                            checkpoint_config=tune.CheckpointConfig(
                                **CHECKPOINT_CONFIG
                            ),
                            **RUN_CONFIG
                        ),
                    )
                    results = tuner.fit()

                    best_result = results.get_best_result("val_loss", "min")

                    # save metrics csvs
                    print(f"Best trial config: {best_result.config}")
                    print(f"Best trial final validation loss: {best_result.metrics}")
                    metrics_df = results.get_dataframe()
                    metrics_df.to_csv(all_trials_path)
                    best_result_df = best_result.metrics_dataframe
                    best_result_df.to_csv(best_result_path)
                    # move model weights to standard location
                    print(f"{best_result.path=}")
                    print(f"{best_result.checkpoint=}")
                    with best_result.checkpoint.as_directory() as checkpoint_dir:
                        model_path = Path(checkpoint_dir).joinpath(MODEL_FILENAME)
                        shutil.copy(str(model_path), str(model_weights_path))
                        opt_path = Path(checkpoint_dir).joinpath(OPTIMIZER_FILENAME)
                        shutil.copy(opt_path, optimizer_path)
                    # remove all other models from results directory
                    ray_dir = Path(results.experiment_path)
                    for f in ray_dir.rglob(OPTIMIZER_FILENAME):
                        f.unlink()
                    # compress all results directories and move to case_dir
                    zip_dir(results.experiment_path, ray_results_path)

                ### LBFGS
                method_dir = data_iteration_dir.joinpath("lbfgs")
                method_dir.mkdir(parents=True, exist_ok=True)
                all_trials_path = method_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                best_result_path = method_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                model_weights_path = method_dir.joinpath(MODEL_FILENAME)
                ray_results_path = method_dir.joinpath("ray_results.zip")
                if all(
                    [
                        all_trials_path.exists(),
                        best_result_path.exists(),
                        model_weights_path.exists(),
                        ray_results_path.exists(),
                    ]
                ):
                    print(
                        f"Skipping existing results for {method_dir}, delete directory to rerun"
                    )
                else:
                    config = {
                        "lr": tune.loguniform(1e-6, 2),
                        "batch_size": tune.qlograndint(1, data_dim, 2),
                        # "max_iter": tune.choice([5, 20, 50, 100]),
                        "max_iter": 20,
                        # "lr_history_size": tune.choice([5, 10, 20, 40]),
                        "lr_history_size": 10,
                        # "lr_patience": tune.choice([10, 20, 40]),
                        "lr_patience": 10,
                        "lr_factor": tune.uniform(0.1, 1.0),
                        "N_batches_per_step": 10,
                        "initialize_weights": "random",
                        "data_ref": data_ref,
                    }

                    scheduler = ASHAScheduler(
                        **ASHA_PARAMS
                    )

                    tuner = tune.Tuner(
                        tune.with_resources(
                            DampedSpringTrainer_LBFGS,
                            resources={"cpu": 1},
                        ),
                        tune_config=tune.TuneConfig(
                            scheduler=scheduler,
                            **TUNE_CONFIG
                        ),
                        param_space=config,
                        run_config=tune.RunConfig(
                            name=f"dampedSpring_transfer_lbfgs_retrain_{scenario_name}_{data_dim_name(data_dim)}_{get_data_iteration_name(data_iteration)}",
                            checkpoint_config=tune.CheckpointConfig(
                                **CHECKPOINT_CONFIG
                            ),
                            **RUN_CONFIG
                        ),
                    )
                    results = tuner.fit()

                    best_result = results.get_best_result("val_loss", "min")

                    # save metrics csvs
                    print(f"Best trial config: {best_result.config}")
                    print(f"Best trial final validation loss: {best_result.metrics}")
                    metrics_df = results.get_dataframe()
                    metrics_df.to_csv(all_trials_path)
                    best_result_df = best_result.metrics_dataframe
                    best_result_df.to_csv(best_result_path)
                    # move model weights to standard location
                    print(f"{best_result.path=}")
                    print(f"{best_result.checkpoint=}")
                    with best_result.checkpoint.as_directory() as checkpoint_dir:
                        model_path = Path(checkpoint_dir).joinpath(MODEL_FILENAME)
                        shutil.copy(str(model_path), str(model_weights_path))

                    zip_dir(
                        results.experiment_path,
                        ray_results_path,
                    )

                del data_ref
    ray.shutdown()