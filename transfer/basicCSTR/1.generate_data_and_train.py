"""Generate operating data and train Neural ODE"""

# imports
import ray

from basicCSTR import *

# constants

# script
if __name__ == "__main__":
    # ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_RESULTS_DIR = RESULTS_DIR.joinpath("training")
    TRAINING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # training data
    if not TRAINING_DATA_PATH.exists():
        t_span = np.arange(0, 8760 * 60, 1)
        x0 = default_rng.uniform(0, 2, size=(3,))
        U = get_U(t_span, 1.0, 3.0, 90)
        res = sim(dXdt, t_span, x0, U, constants, noise=np.array([0.01, 0.01, 0.01]))
        np.savez(TRAINING_DATA_PATH, **res)
    
    # transfer data
    if not TRANSFER_DATA_PATH.exists():
        transfer_constants = constants.copy()
        transfer_constants.update({"V": 12.0})
        t_span = np.arange(0, 8760 * 60, 1)
        x0 = default_rng.uniform(0, 2, size=(3,))
        U = get_U(t_span, 1.0, 3.0, 90)
        res = sim(dXdt, t_span, x0, U, transfer_constants, noise=np.array([0.01, 0.01, 0.01]))
        np.savez(TRANSFER_DATA_PATH, **res)
    
    # train model if weights don't exist
    if not MODEL0_PATH.exists():
        # load training data
        res = dict(np.load(TRAINING_DATA_PATH))
        for k, v in res.items():
            res[k] = v.astype(np.float64)
        
        data_len = len(res["Y"])
        val_test_index = data_len - 31 * 24 * 60  # last month test data
        train_val_index = val_test_index - 30 * 24 * 60  # prior month validation data
        
        data = {
            "train_y": res["Y"][:train_val_index],
            "train_u": res["U"][:train_val_index],
            "val_y": res["Y"][train_val_index:val_test_index],
            "val_u": res["U"][train_val_index:val_test_index],
            "test_y": res["Y"][val_test_index:],
            "test_u": res["U"][val_test_index:],
        }
        
        data_ref = ray.put(data)
        
        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64, 128]),
            "lr_patience": tune.choice([10, 20, 50, 100]),
            "lr_factor": tune.uniform(0.1, 0.9),
            "scaling": tune.choice([True, False]),
            "data_ref": data_ref,
        }
        
        scheduler = ASHAScheduler(
            max_t=500,
            grace_period=20,
            reduction_factor=2
        )
        
        tuner = tune.Tuner(
            tune.with_resources(
                BasicCSTRTrainer,
                resources={"cpu": 1},
            ),
            tune_config=tune.TuneConfig(
                metric="val_L2e",
                mode="min",
                scheduler=scheduler,
                max_concurrent_trials=4,
                num_samples=50,
            ),
            param_space=config,
            run_config=tune.RunConfig(
                verbose=1,
                name="basicCSTR_training",
                # storage_path=RAY_STORAGE_PATH,
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=100,
                    checkpoint_at_end=True,
                ),
            ),
        )
        results = tuner.fit()
        
        best_result = results.get_best_result("val_L2e", "min")
        
        metrics_df = results.get_dataframe()
        metrics_df.to_csv(TRAINING_RESULTS_DIR.joinpath("all_trials_metrics.csv"))
        best_result_df = best_result.metrics_dataframe
        best_result_df.to_csv(TRAINING_RESULTS_DIR.joinpath("best_trial_metrics.csv"))
        
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            shutil.copy(
                Path(checkpoint_dir).joinpath(MODEL_FILENAME),
                TRAINING_RESULTS_DIR.joinpath(MODEL_FILENAME),
            )
            shutil.copy(
                Path(checkpoint_dir).joinpath(OPTIMIZER_FILENAME),
                TRAINING_RESULTS_DIR.joinpath(OPTIMIZER_FILENAME),
            )
                
                