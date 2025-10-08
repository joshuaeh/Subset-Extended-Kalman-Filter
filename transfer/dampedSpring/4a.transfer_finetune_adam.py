"""retrain NN on target system data, randomly initializing weights each time."""

# imports
from dampedSpring import *

# constants

# script
if __name__ == "__main__":
    initialization_dir = RESULTS_DIR.joinpath("transfer", "finetune")
    initialization_dir.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        scenario_name = transfer_scenario_name(scenario)
        scenario_dir = initialization_dir.joinpath(scenario_name)
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for data_dim in TARGET_DATA_DIM:
            data_dim_dir = scenario_dir.joinpath(data_dim_name(data_dim))
            data_dim_dir.mkdir(parents=True, exist_ok=True)
            for data_iteration in range(N_ITERATIONS):
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
                        f"Skipping existing results for {data_iteration_dir}, delete directory to rerun"
                    )
                else:
                    config = {
                        "lr": tune.loguniform(1e-6, 1e-1),
                        "batch_size": tune.qlograndint(1, data_dim, 2),
                        "lr_patience": 10,
                        "lr_factor": tune.uniform(0.0, 1.0),
                        "mask_fn_quantile_thresh": tune.quniform(0.0, 1.0, 0.05),
                        "initialize_weights": "finetune",
                    }

                    scheduler = ASHAScheduler(
                        time_attr="training_iteration",
                        max_t=1000,
                        grace_period=50,
                        reduction_factor=2,
                    )

                    trial_stopper = TrialPlateauStopper(
                        metric="val_loss",
                        std=0.00001,
                        num_results=50,
                        grace_period=50,
                        mode="min",
                    )

                    tuner = tune.Tuner(
                        tune.with_resources(
                            tune.with_parameters(DampedSpringTrainer, data=data),
                            resources={"cpu": 1},
                        ),
                        tune_config=tune.TuneConfig(
                            metric="val_loss",
                            mode="min",
                            scheduler=scheduler,
                            max_concurrent_trials=4,
                            num_samples=100,
                            # reuse_actors=True
                        ),
                        param_space=config,
                        run_config=tune.RunConfig(
                            verbose=1,
                            name=f"dampedSpring_transfer_adam_retrain_{scenario_name}_{data_dim_name(data_dim)}_{get_data_iteration_name(data_iteration)}",
                            storage_path=RAY_STORAGE_PATH,
                            checkpoint_config=tune.CheckpointConfig(
                                num_to_keep=1,
                                checkpoint_frequency=1000,
                                checkpoint_at_end=True,
                            ),
                            stop=trial_stopper,
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