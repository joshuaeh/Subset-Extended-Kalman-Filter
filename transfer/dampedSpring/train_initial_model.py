"""Train MLP Model for source task damped spring system"""
# imports
from dampedSpring import *

# constants

# script
if __name__ == "__main__":

    # --- generate training data
    if not os.path.exists(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME)):
        n_train_data_total = N_TRAIN + N_VALIDATION + N_TEST + N_TEST
        x0 = rng.uniform(-5, 5, n_train_data_total)
        x_dot0 = rng.uniform(-5, 5, n_train_data_total)
        X, Y = generate_dataset(x0, x_dot0, t_eval=np.arange(1, 21, 1))
        
        np.savez(os.path.join(DATA_DIR, TRAINING_DATA_FILENAME), X=X, Y=Y)

    # load training data
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
    
    # initialize trainable
    data = {
    "train_x": X_train,
    "train_y": Y_train,
    "val_x": X_validation,
    "val_y": Y_validation,
    "test_x": X_test,
    "test_y": Y_test,
    }
    
    config = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([16, 32, 64, 128, 256, 1028, 4096]),
        "lr_patience": tune.choice([10, 20, 50, 100]),
        "lr_factor": tune.uniform(0.1, 0.9),
        "initialize_weights": "random"
    }
    
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=1000,
        grace_period=50,
        reduction_factor=2)

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
            max_concurrent_trials=4,
            num_samples=100,
            # reuse_actors=True
        ),
        param_space=config,
        run_config = tune.RunConfig(
            verbose=1,
            name=f"dampedSpring_training",
            # storage_path=r"C:\Users\jhamm\Desktop\SEKF\transfer\dampedSpring\data\ray_results",
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=1000,
                checkpoint_at_end=True
            ),


        )
    )
    results = tuner.fit()

    best_result = results.get_best_result("val_loss", "min")

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics}")
    metrics_df = results.get_dataframe()
    metrics_df.to_csv(os.path.join(DATA_DIR, f"allTrials_metrics.csv"))
    best_result_df = best_result.metrics_dataframe
    best_result_df.to_csv(os.path.join(DATA_DIR, f"bestResult_metrics.csv"))
    print(f"{best_result.path=}")
    print(f"{best_result.checkpoint=}")
    with best_result.checkpoint.as_directory() as checkpoint_dir:
        model_path = os.path.join(checkpoint_dir, MODEL_FILENAME)
        target_path = os.path.join(DATA_DIR, "model_weights.pth")
        shutil.move(model_path, target_path)
    
         

    
    # plotting
    
    # fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # ax[0].plot(train_loss, label='Train Loss')
    # ax[0].plot(validation_loss, label='Validation Loss')
    # ax[0].set_yscale('log')
    # ax[0].set_ylabel('Loss (log scale)')
    # ax[0].legend()
    # ax[1].plot(learning_rate, label='Learning Rate')
    # ax[1].set_xlabel('Epoch')
    # ax[1].set_ylabel('Learning Rate')
    # ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # plt.show()

    # N_validation_examples = 5
    # examples_indices = np.random.choice(X_validation_tensor.size(0), N_validation_examples, replace=False)

    # t_plot = np.arange(1, 21)  # Time for plotting
    # fig, ax = plt.subplots(N_validation_examples, 1, figsize=(10, 2 * N_validation_examples), sharex=True)
    # for i, idx in enumerate(examples_indices):
    #     x0, x_dot0 = X_validation_tensor[idx]
    #     t_sim, states_sim = sim_mass_spring_damper(x0=x0.item(), x_dot0=x_dot0.item(), t_eval=np.linspace(0,20,200))
    #     y_pred = model(X_validation_tensor[idx:idx+1]).detach().numpy().flatten()
    #     # initial position
    #     ax[i].plot(0, x0.item(), "bo", label="Initial Position")
    #     # arrow showing initial velocity with length proportional to x_dot0 and number to the left showing its value
    #     ax[i].arrow(0, x0.item(), 0, x_dot0.item(), length_includes_head=True, head_width=0.25, head_length=0.25, fc='red', ec='red')
    #     ax[i].text(-0.1, x0.item() + x_dot0.item() / 2, f'{x_dot0.item():.2f}', color='red', fontsize=10, ha='right')
    #     ax[i].plot(t_sim, states_sim[0], label='True Position', color='blue')
    #     ax[i].plot(t_plot, y_pred, label='Predicted Position', marker="o", color='orange', linestyle='none')
    #     ax[i].hlines(0, 0, 21, color="k", linestyle='--', linewidth=0.5)
    #     ax[i].set_title(fr'Initial Conditions: $x_0$={x0.item():.2f}, $\dot{{x}}_0$={x_dot0.item():.2f}', pad=-14)
    #     ax[i].set_ylabel('Position (x)')
    #     # ax[i].legend()
    # ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3)
    # ax[i].set_xlabel('Time (t)')
    # fig.tight_layout()
    # plt.show()