import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler
import tclab
import torch
from tqdm import tqdm

from tclab_utils import *


# --- simulated data generation ---
if not Path("data").joinpath("tclab_sim_data.csv").exists():
    q1_time = 60
    q2_time = 300
    q1_setting = 0
    q2_setting = 0

    data = []

    Ta = 23
    T1 = Ta
    T2 = Ta

    for t in tqdm(
        range(0, 3600 * 24 * 365, 10)
    ):  # 24 hours in seconds, every 10 seconds
        if t >= q1_time:
            q1_time = t + rng.integers(6, 60) * 10  # 1-10 minutes
            q1_setting = rng.integers(0, 101)  # 0-100
            # 50% change of q1_setting being 0
            if rng.random() < 0.5:
                q1_setting = 0
        if t >= q2_time:
            q2_time = t + rng.integers(6, 60) * 10  # 1-10 minutes
            q2_setting = rng.integers(0, 101)
            # 50% change of q2_setting being 0
            if rng.random() < 0.5:
                q2_setting = 0
        Ta = np.clip(Ta + rng.normal(0, 0.05), 20, 26)
        X = np.array([T1, T2])
        U = np.array([q1_setting, q2_setting, Ta])
        sol = solve_ivp(dTdt_TCLab, [t, t + 10], X, args=(U,), t_eval=[t + 10])
        T1, T2 = sol.y[:, -1]
        data.append([t, T1, T2, Ta, q1_setting, q2_setting])
    df = pd.DataFrame(data, columns=["time", "T1", "T2", "Ta", "Q1", "Q2"])
    df.to_csv(Path("data").joinpath("tclab_sim_data.csv"), index=False)
else:
    df = pd.read_csv(Path("data").joinpath("tclab_sim_data.csv"))

# train model
if not MODEL0_PATH.exists():
    train_val_index = int(len(df) * 0.8)
    val_test_index = int(len(df) * 0.9)
    
    config = {
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([2**10, 2**12, 2**14, 2**16]),
        "lr_patience": tune.choice([10, 20, 50, 100]),
        "lr_factor": tune.uniform(0.1, 0.9),
        "scaling": tune.choice([True, False]),
        "train_begin_idx": 0,
        "train_end_idx": train_val_index,
        "val_begin_idx": train_val_index,
        "val_end_idx": val_test_index,
        "test_begin_idx": val_test_index,
        "test_end_idx": len(df),
        "train_dataset_stride": 6
    }
    
    scheduler = ASHAScheduler(
        max_t=500,
        grace_period=20,
        reduction_factor=2
    )
    
    trial_stopper = TrialPlateauStopper(
        metric="val_loss",
        std=0.00001,
        num_results=10,
        grace_period=20,
        mode="min",
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(TCLabTrainer, data=df),
            resources={"cpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="val_L2e",
            mode="min",
            scheduler=scheduler,
            max_concurrent_trials=2,
            num_samples=50,
        ),
        param_space=config,
        run_config=tune.RunConfig(
            verbose=0,
            name="TCLabTraining",
            # storage_path=RAY_STORAGE_PATH,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=100,
                checkpoint_at_end=True,
            ),
            stop=trial_stopper,
            # storage_path=RAY_STORAGE_PATH
            storage_path="/tmp/"
        ),
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("val_L2e", "min")
    
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
        target_path = MODEL0_PATH
        shutil.copy(str(model_path), str(target_path))

    # # compress all results directories and move to case_dir
    # shutil.move(
    #     results.experiment_path, RESULTS_DIR.joinpath("training", "ray_results")
    # )
    # compress all results directories and move to case_dir
    zip_dir(
        results.experiment_path, RESULTS_DIR.joinpath("training", "ray_results")
    )

