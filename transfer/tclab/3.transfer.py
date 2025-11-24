import os
from tclab_utils import *

# constants
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

# script
if __name__ == "__main__":
    # ensure directories exist
    retrain_dir = RESULTS_DIR.joinpath("retrain")
    retrain_dir.mkdir(parents=True, exist_ok=True)
    finetune_dir = RESULTS_DIR.joinpath("finetune")
    finetune_dir.mkdir(parents=True, exist_ok=True)
    
    # train data
    train_df = pd.read_csv("data/tclab_sim_data.csv")
    X_data = train_df[["Ta1", "Ta2"]].to_numpy()[:int(0.8*len(train_df))]
    U_data = train_df[["Q1", "Q2", "Ta"]].to_numpy()[:int(0.8*len(train_df))]
    train_x_scale = X_data.std(axis=0)
    train_x_mean = X_data.mean(axis=0)
    train_u_scale = U_data.std(axis=0)
    train_u_mean = U_data.mean(axis=0)
    
    # get data
    df = pd.read_csv("data/tclab_measured_data.csv", index_col="dt", parse_dates=True)
    df = df.sort_index(inplace=False)
    # df[["Ta1", "Ta2"]] = df[["Ta1", "Ta2"]].interpolate(method="nearest", limit_direction="both", )
    df[["Ta1", "Ta2"]] = df[["Ta1", "Ta2"]].interpolate(method="time", limit_direction="both", )
    df["Ta"] = df[["Ta1", "Ta2"]].mean(axis=1)
    df.drop(columns=["Ta1", "Ta2"], inplace=True)
    df.dropna(subset=["T1", "T2"], inplace=True)
    df["dt"] = df["t"] - df["t"].shift(1)
    df["valid"] = ((df["dt"] > 9.9) & (df["dt"] < 10.1))
    df["invalid"] = ~df["valid"]
    df = df["2025-10-01":]  # only keep the 7 contiguous days
    
    METHODS = {
        "SEKF": TCLabTrainer_SEKF, 
        "LBFGS": TCLabTrainer_LBFGS, 
        "Adam": TCLabTrainer,
    }
    
    counter = 0
    total_runs = 2 * 5 * 4 * 3
    
    for initialization_name, initialization_config in INITIALIZATIONS.items():
        for day in range(DAYS):
            for data_dim_name, data_dim in DATA_DIM.items():
                for method, trainer in METHODS.items():
                    
                    counter += 1
                    print(f"Run {counter} of {total_runs}: {initialization_name}, day {day}, data dim {data_dim_name}, method {method}")
                    run_dir = RESULTS_DIR.joinpath(initialization_name, f"day{day}", data_dim_name, method)
                    run_dir.mkdir(parents=True, exist_ok=True)
                    all_trials_path = run_dir.joinpath(ALL_TRIALS_BASE_FILENAME)
                    best_result_path = run_dir.joinpath(BEST_RESULT_BASE_FILENAME)
                    model_weights_path = run_dir.joinpath(MODEL_FILENAME)
                    ray_results_path = run_dir.joinpath("ray_results.zip")
                    
                    if not all(p.exists() for p in [all_trials_path, best_result_path, model_weights_path, ray_results_path]):
                        
                        configs = {
                            "Adam": {
                                "lr": tune.loguniform(1e-6, 1e-1),
                                "batch_size": tune.qlograndint(1, int(0.9*data_dim)-60, 2),
                                "lr_patience": tune.choice([10, 20, 50, 100]),
                                "lr_factor": tune.uniform(0.1, 0.9),
                                "mask_fn_quantile_thresh": tune.uniform(0.0, 1.0),
                                "initialize_weights": "random"
                            },
                            "SEKF": {
                                # "R": tune.choice([0, 0.01, 0.05, 0.1, 0.5, 1.0]),
                                "R": tune.loguniform(1e-8, 1.0),
                                # "Q": tune.choice([0, 1e-6, 1e-4, 1e-2, 1e-1]),
                                "Q": tune.loguniform(1e-8, 1e-1),
                                "p0": tune.choice([0.01, 0.1, 0.5, 1.0, 10.0, 100.0]),
                                "batch_size": tune.qlograndint(1, min(20, int(0.9*data_dim)-60), 2),
                                "mask_fn_quantile_thresh": tune.uniform(0.0, 1.0),
                                "initialize_weights": "random",
                            },
                            "LBFGS": {
                                "lr": tune.loguniform(1e-6, 1e0),
                                "batch_size": tune.qlograndint(1, int(0.9*data_dim)-60, 2),
                                "lr_max_iter": tune.choice([5, 10, 20, 50]),
                                "lr_history_size": tune.choice([5, 10, 20]),
                                "lr_patience": tune.choice([10, 20, 50, 100]),
                                "lr_factor": tune.uniform(0.1, 0.9),
                                "initialize_weights": "random",
                            }
                        }
                    
                        # get data for day
                        train_count = int(0.9 * data_dim)
                        train_begin_idx = day * MEASUREMENTS_PER_DAY
                        train_end_idx = train_begin_idx + train_count
                        val_begin_idx = train_end_idx
                        val_end_idx = train_begin_idx + data_dim
                        test_begin_idx = 5 * MEASUREMENTS_PER_DAY
                        test_end_idx = 7 * MEASUREMENTS_PER_DAY
                        
                        config = configs[method].copy()
                        config.update({
                            "initialize_weights": initialization_config,
                            "x_scale": train_x_scale,
                            "x_mean": train_x_mean,
                            "u_scale": train_u_scale,
                            "u_mean": train_u_mean,
                            "train_begin_idx": train_begin_idx,
                            "train_end_idx": train_end_idx,
                            "val_begin_idx": val_begin_idx,
                            "val_end_idx": val_end_idx,
                            "test_begin_idx": test_begin_idx,
                            "test_end_idx": test_end_idx,
                        })
                        if initialization_name == "retrain":
                            config["mask_fn_quantile_thresh"] = None  # no masking for retrain
                        
                        # ---
                        scheduler = ASHAScheduler(
                            time_attr="training_iteration",
                            max_t=1000,
                            grace_period=50,
                            reduction_factor=2,
                        )

                        trial_stopper = TrialPlateauStopper(
                            metric="val_loss",
                            std=0.00001,
                            num_results=10,
                            grace_period=50,
                            mode="min",
                        )

                        tuner = tune.Tuner(
                            tune.with_resources(
                                tune.with_parameters(trainer, data=df),
                                resources={"cpu": 1},
                            ),
                            tune_config=tune.TuneConfig(
                                metric="val_L2e",
                                mode="min",
                                scheduler=scheduler,
                                max_concurrent_trials=2,
                                num_samples=50,
                                # reuse_actors=True
                            ),
                            param_space=config,
                            run_config=tune.RunConfig(
                                verbose=0,
                                name=f"tclab_{initialization_name}_day{day}_{data_dim_name}_{method}",
                                # storage_path=RAY_STORAGE_PATH,
                                storage_path="/tmp/",
                                checkpoint_config=tune.CheckpointConfig(
                                    num_to_keep=1,
                                    checkpoint_frequency=1000,
                                    checkpoint_at_end=True,
                                ),
                                stop=trial_stopper,
                            ),
                        )
                        results = tuner.fit()

                        best_result = results.get_best_result("val_L2e", "min")

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
                        
                            
                        
                    
                    
                
    
    
    