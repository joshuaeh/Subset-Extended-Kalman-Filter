"""Train MLP model on source damped spring system"""

# imports
from dampedSpring import *

# constants

# script
if __name__ == "__main__":
    # generate filesystem if doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["training", "transfer"]:
        RESULTS_DIR.joinpath(subdir).mkdir(parents=True, exist_ok=True)
    for method in INIT_METHODS:
        RESULTS_DIR.joinpath("transfer", method).mkdir(parents=True, exist_ok=True)

    # source system data
    if not TRAINING_DATA_PATH.exists():
        x0, x_dot0 = (
            rng.uniform(-5, 5, N_SOURCE_TOTAL),
            rng.uniform(-5, 5, N_SOURCE_TOTAL),
        )
        X, Y = generate_dataset(x0, x_dot0, t_eval=np.arange(1, 21, 1))
        np.savez(TRAINING_DATA_PATH, X=X, Y=Y)

    # Target system data w/same exogenous inputs
    if not TRANSFER_DATA_PATH.exists():
        x0, x_dot0 = (
            rng.uniform(-5, 5, N_SOURCE_TOTAL),
            rng.uniform(-5, 5, N_SOURCE_TOTAL),
        )
        transfer_data = {}
        for scenario in SCENARIOS:
            X, Y = generate_dataset(x0, x_dot0, t_eval=np.arange(1, 21, 1), **scenario)
            transfer_data[transfer_scenario_name(scenario)] = Y
        transfer_data["X"] = X  # same X for all scenarios
        np.savez(TRANSFER_DATA_PATH, **transfer_data)

    if not MODEL0_WEIGHTS_PATH.exists():
        # load training data
        training_data = np.load(TRAINING_DATA_PATH)
        X = training_data["X"]
        Y = training_data["Y"]
        # add measurement noise
        X += rng.normal(0, MEASUREMENT_NOISE_STD, X.shape)
        Y += rng.normal(0, MEASUREMENT_NOISE_STD, Y.shape)
        # split and convert to tensors
        x_train, y_train, x_validation, y_validation, x_test, y_test = (
            train_val_test_split(
                X,
                Y,
                n_train=N_SOURCE_TRAIN,
                n_validation=N_SOURCE_VALIDATION,
                n_test=N_SOURCE_TEST,
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

        config = {
            "lr": tune.loguniform(1e-6, 1e-1),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "lr_patience": tune.choice([10, 20, 50, 100]),
            "lr_factor": tune.uniform(0.1, 0.9),
            "initialize_weights": "random",
        }

        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=1000,
            grace_period=250,
            reduction_factor=2,
        )

        trial_stopper = TrialPlateauStopper(
            metric="val_loss", std=0.00001, num_results=50, grace_period=50, mode="min"
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
                name=f"dampedSpring_training",
                storage_path=RAY_STORAGE_PATH,
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1, checkpoint_frequency=1000, checkpoint_at_end=True
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
        metrics_df.to_csv(
            str(RESULTS_DIR.joinpath("training", ALL_TRIALS_BASE_FILENAME))
        )
        best_result_df = best_result.metrics_dataframe
        best_result_df.to_csv(
            str(RESULTS_DIR.joinpath("training", BEST_RESULT_BASE_FILENAME))
        )
        # move model weights to standard location
        print(f"{best_result.path=}")
        print(f"{best_result.checkpoint=}")
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            model_path = Path(checkpoint_dir).joinpath(MODEL_FILENAME)
            target_path = MODEL0_WEIGHTS_PATH
            shutil.copy(str(model_path), str(target_path))

        # compress all results directories and move to case_dir
        shutil.move(
            results.experiment_path, RESULTS_DIR.joinpath("training", "ray_results")
        )
