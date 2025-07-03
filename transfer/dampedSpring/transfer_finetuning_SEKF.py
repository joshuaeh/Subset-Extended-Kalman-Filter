"""Finetune each scenario from the damped spring system"""
# imports
from dampedSpring import *

# constants
results_dir = os.path.join(DATA_DIR, "transfer_finetuning")
os.makedirs(results_dir, exist_ok=True)

# script
if __name__ == "__main__":
    for scenario in SCENARIOS:
        scenario_name = transfer_scenario_name(scenario)
        if not get_transfer_data(scenario):
            x, y = generate_transfer_data(scenario, N_TRANSFER_TOTAL)
        else:
            x, y = get_transfer_data(scenario)
        # add measurement noise
        x += rng.normal(0, MEASUREMENT_NOISE_PM/3, x.shape)
        y += rng.normal(0, MEASUREMENT_NOISE_PM/3, y.shape)
        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x, y, n_train=N_TRANSFER, n_validation=N_TRANSFER_VALIDATION, n_test=N_TRANSFER_TOTAL - N_TRANSFER - N_TRANSFER_VALIDATION, tensor_convert=True)

        data = {
        "train_x": x_train,
        "train_y": y_train,
        "val_x": x_val,
        "val_y": y_val,
        "test_x": x_test,
        "test_y": y_test,}
        
        config = {
            "batch_size": tune.choice([1, 16, 32, 64, 128, 256, 500, 1000]),
            "initialize_weights": "finetune",
            "lr": tune.loguniform(1e-6, 10),
            "lr_patience": 1000,
            "max_epochs": 1000,
            "mask_fn_quantile_thresh": tune.uniform(0.0, 1.0),
            "optimizer": "sekf",
            "sekf_q": tune.choice([0, 1e-6, 1e-5, 1e-4, 1e-3]),
            "sekf_save_path": os.path.join(DATA_DIR, "transfer_finetuning_SEKF", "SourceSystemPretraining_sekf_optimizer_state.npz")
        }

        # scheduler = ASHAScheduler(
        #     time_attr="training_iteration",
        #     max_t=config["max_epochs"],
        #     grace_period=50,
        #     reduction_factor=2)
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=config["max_epochs"],
            grace_period=50,
            reduction_factor=2)
        
        trial_stopper = TrialPlateauStopper(
            metric="val_loss",
            std=0.1e-6,
            num_results=10,
            grace_period=50,
            mode="min"
        )
        
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(
                    DampedSpringTrainer,
                    data=data
                ),
                    
                resources={"cpu": 1},
            ),
            tune_config=tune.TuneConfig(
                metric="val_loss",
                mode="min",
                scheduler=scheduler,
                max_concurrent_trials=N_CPUS,
                num_samples=N_HYPERPARAMETER_TRIALS,
                # reuse_actors=True
            ),
            param_space=config,
            run_config = tune.RunConfig(
                verbose=1,
                name=f"dampedSpring_transfer_finetuning_{scenario_name}",
                # storage_path=r"C:\Users\jhamm\Desktop\SEKF\transfer\dampedSpring\data\ray_results",
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=1000,
                    checkpoint_at_end=True
                ),
                stop=trial_stopper
            )
        )
        results = tuner.fit()

        best_result = results.get_best_result("val_loss", "min")
        
        print(f"Best trial config: {best_result.config}")
        print(f"Best trial final validation loss: {best_result.metrics}")
        metrics_df = results.get_dataframe()
        metrics_df.to_csv(os.path.join(results_dir, f"{scenario_name}_allTrials_metrics.csv"))
        best_result_df = best_result.metrics_dataframe
        best_result_df.to_csv(os.path.join(results_dir, f"{scenario_name}_bestResult_metrics.csv"))
        print(f"{best_result.path=}")
        print(f"{best_result.checkpoint=}")
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            model_path = os.path.join(checkpoint_dir, MODEL_FILENAME)
            target_path = os.path.join(results_dir, f"{scenario_name}_model_weights.pth")
            shutil.move(model_path, target_path)
            optimizer_state_checkpoint = os.path.join(checkpoint_dir, OPTIMIZER_STATE_SEKF)
            optimizer_state_target = os.path.join(results_dir, f"{scenario_name}_sekf_optimizer_state.npz")
            shutil.move(optimizer_state_checkpoint, optimizer_state_target)