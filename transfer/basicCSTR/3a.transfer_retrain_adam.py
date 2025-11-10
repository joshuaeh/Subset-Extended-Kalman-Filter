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
    transfer_results = np.load(TRANSFER_DATA_PATH)
    
    for month in range(12):
        month_start = sum(MONTH_DAYS[:month]) * 24 * 60
        month_end = month_start + MONTH_DAYS[month] * 24 * 60
        for training_days in [1, 10]:
            TRANSFER_RESULTS_DIR = RESULTS_DIR.joinpath("transfer","retraining", f"month_{month+1}", "days_{training_days}", method)
            TRANSFER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            if not TRANSFER_RESULTS_DIR.joinpath("results.npz").exists():
                train_val_idx = month_start + int(0.8 * training_days * 24 * 60)
                val_test_idx = train_val_idx + training_days * 24 * 60
                data = {
                    "train_y": transfer_results["Y"][month_start:train_val_idx],
                    "train_u": transfer_results["U"][month_start:train_val_idx],
                    "val_y": transfer_results["Y"][train_val_idx:val_test_idx],
                    "val_u": transfer_results["U"][train_val_idx:val_test_idx],
                    "test_y": transfer_results["Y"][val_test_idx:month_end],
                    "test_u": transfer_results["U"][val_test_idx:month_end],
                }
                
                config = {
                    "lr": tune.loguniform(1e-4, 1e-1),
                    "batch_size": tune.choice([16, 32, 64, 128]),
                    "lr_patience": tune.choice([10, 20, 50, 100]),
                    "lr_factor": tune.uniform(0.1, 0.9),
                    "scaling": tune.choice([True, False]),
                }
                
                scheduler = ASHAScheduler(
                    max_t=500,
                    grace_period=20,
                    reduction_factor=2
                )
                
                tuner = tune.Tuner(
                    tune.with_resources(
                        tune.with_parameters(BasicCSTRTrainer, data=data),
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