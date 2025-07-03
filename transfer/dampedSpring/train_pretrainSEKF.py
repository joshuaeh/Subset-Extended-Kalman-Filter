"get SEKF Covariance Matrix P to converge"

from dampedSpring import *

results_dir = os.path.join(DATA_DIR, "transfer_finetuning_SEKF")
os.makedirs(results_dir, exist_ok=True)
scenario_name = "SourceSystemPretraining"

training_data = np.load(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME))
X = training_data["X"]
Y = training_data["Y"]
# add measurement noise
X += rng.normal(0, MEASUREMENT_NOISE_PM/3, X.shape)
Y += rng.normal(0, MEASUREMENT_NOISE_PM/3, Y.shape)
# split and convert to tensors
X_train, Y_train, X_validation, Y_validation, X_test, Y_test = \
    train_val_test_split(
        X, Y,
        n_train=N_TRAIN, n_validation=N_VALIDATION, n_test=N_TEST,
        tensor_convert=True
        )
    
# train to converge the covariance matrix
config = {
    "initialize_weights": "finetune",
    "batch_size": 16,
    "optimizer": "sekf",
    "sekf_q": 0.0001,
    "sekf_p0": 10,
    "lr": 1/(1e-1/3)
}

data = {
    # "train_x": X_train,
    # "train_y": Y_train,
    "train_x": X_train[:10000],
    "train_y": Y_train[:10000],
    "val_x": X_validation,
    "val_y": Y_validation,
    "test_x": X_test,
    "test_y": Y_test,
    }

scheduler = ASHAScheduler(
    time_attr="training_iteration",
    max_t=5,
    grace_period=5,
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
        num_samples=1,
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