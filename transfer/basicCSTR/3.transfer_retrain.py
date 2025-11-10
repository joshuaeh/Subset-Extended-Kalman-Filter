# imports
from basicCSTR import *

# constants
method = "adam"

# script
if __name__ == "__main__":
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
    train_data = np.load(DATA_DIR.joinpath("training_data.npz"))
    transfer_data = np.load(TRANSFER_DATA_PATH)
    
    train_x_scale = train_data["Y"][:-61*24*60].std(axis=0)
    train_x_mean = train_data["Y"][:-61*24*60].mean(axis=0)
    train_u_scale = train_data["U"][:-61*24*60].std(axis=0)
    train_u_mean = train_data["U"][:-61*24*60].mean(axis=0)
    
    print(f"Train x scale: {train_x_scale}")
    print(f"Train x mean: {train_x_mean}")
    print(f"Train u scale: {train_u_scale}")
    print(f"Train u mean: {train_u_mean}")
    
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
    transfer_results = np.load(TRANSFER_DATA_PATH)
    
    # --- configs ---
    configs = {
        "adam": {
            "lr": tune.loguniform(1e-6, 1e-1),
            "batch_size": tune.qlograndint(1, data_dim, 2),
            "lr_patience": tune.choice([10, 20, 50, 100]),
            "lr_factor": tune.uniform(0.1, 0.9),
            "initialize_weights": "random"
        },
        "sekf": {
            "R": tune.choice([0, 0.01, 0.05, 0.1, 0.5, 1.0]),
            "Q": tune.choice([0, 1e-6, 1e-4, 1e-2, 1e-1]),
            "p0": tune.choice([0.01, 0.1, 0.5, 1.0, 10.0, 100.0]),
            "batch_size": tune.qlograndint(1, min(20, data_dim), 2),
            # "mask_fn_quantile_thresh": tune.uniform(0.0, 1.0),
            "initialize_weights": "random",
        },
        "lbfgs": {
            "lr": tune.loguniform(1e-6, 1e0),
            "lr_max_iter": tune.choice([5, 10, 20, 50]),
            "lr_history_size": tune.choice([5, 10, 20]),
            "lr_patience": tune.choice([10, 20, 50, 100]),
            "lr_factor": tune.uniform(0.1, 0.9),
            "initialize_weights": "random",
        }
    }
    
    trainables = {
        "adam": BasicCSTRTrainer,
        "sekf": CSTRTrainer_SEKF,
        "lbfgs": CSTRTrainer_LBFGS,
    }
    
    counter = 0
    total_runs = 12 * 3 * 3  # months * training_days * methods
    
    for month in range(12):
        month_start = sum(MONTH_DAYS[:month]) * 24 * 60
        month_end = month_start + MONTH_DAYS[month] * 24 * 60
        for training_days in [0.25, 1, 7]:
            for method in ["adam", "sekf", "lbfgs"]:
                TRANSFER_RESULTS_DIR = RESULTS_DIR.joinpath("transfer","retraining", f"month_{month+1}", f"days_{training_days}", method)
                TRANSFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                counter += 1
                print(f"Running transfer retraining {counter} of {total_runs}: month {month+1}, days {training_days}, method {method}")
                if not TRANSFER_RESULTS_DIR.joinpath("results.npz").exists():
                    train_validation_idx = month_start + 0.8 * (training_days * 24 * 60)
                    validation_test_idx = month_start + (training_days * 24 * 60)
                    all_trials_path = TRANSFER_RESULTS_DIR.joinpath(ALL_TRIALS_BASE_FILENAME)
                    best_result_path = TRANSFER_RESULTS_DIR.joinpath(BEST_RESULT_BASE_FILENAME)
                    model_weights_path = TRANSFER_RESULTS_DIR.joinpath(MODEL_FILENAME)
                    ray_results_path = TRANSFER_RESULTS_DIR.joinpath("ray_results.zip")
                    
                    data = {
                        "train_y": transfer_data["Y"][month_start:train_validation_idx],
                        "train_u": transfer_data["U"][month_start:train_validation_idx],
                        "val_y": transfer_data["Y"][train_validation_idx:validation_test_idx],
                        "val_u": transfer_data["U"][train_validation_idx:validation_test_idx],
                        "test_y": transfer_data["Y"][validation_test_idx:month_end],
                        "test_u": transfer_data["U"][validation_test_idx:month_end],
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
                            tune.with_parameters(trainables[method], data=data),
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
                        param_space=configs[method],
                        run_config=tune.RunConfig(
                            verbose=0,
                            name=f"dampedSpring_retrain_month{month+1}_days{training_days}_{method}",
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
            
            