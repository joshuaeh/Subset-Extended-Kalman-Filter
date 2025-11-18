import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from scipy.integrate import solve_ivp
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import tclab
import torch
from torch import nn, optim
from tqdm import tqdm

from sekf.modeling import Exogenous_RkRNN, init_weights, get_jacobian, seed_worker, g, get_parameter_vector, cosine_similarity
from sekf.optimizers import maskedAdam, SEKF

rng = np.random.default_rng(42)
colors = sns.color_palette("colorblind", 10)
GENERATED_DATA_FILENAME = "tclab_sim_data.csv"
MEASURED_DATA_FILENAME = "tclab_measured_data.csv"
MODEL_FILENAME = "tclab_model.pth"
MODEL0_PATH = Path("results").joinpath(MODEL_FILENAME)
OPTIMIZER_FILENAME = "tclab_optimizer.pth"

# simulated model
Ua = 10  # W/m^2 K
m = 4e-3  # kg
Cp = 5e2  # J/kg K
A = 1e-3  # m^2 area not between heaters
As = 2e-4  # m^2 area between heaters
alpha1 = 1e-2  # W/% heater 1
alpha2 = 5e-3  # W/% heater 2
eps = 0.9  # emissivity
s = 5.67e-8  # Stefan-Boltzmann constant


def dTdt_TCLab(t, X, U):
    """Derivative function for TCLab model."""
    T1, T2 = X
    Q1, Q2, Ta = U

    Q_C12 = Ua * As * (T2 - T1)  # convection between heaters
    Q_R12 = eps * s * As * (T2**4 - T1**4)  # radiation between heaters
    dT1dt = (1.0 / (m * Cp)) * (
        Ua * A * (Ta - T1) + eps * s * A * (Ta**4 - T1**4) + Q_C12 + Q_R12 + alpha1 * Q1
    )
    dT2dt = (1.0 / (m * Cp)) * (
        Ua * A * (Ta - T2) + eps * s * A * (Ta**4 - T2**4) - Q_C12 - Q_R12 + alpha2 * Q2
    )
    return np.array([dT1dt, dT2dt])  # Ta is constant in this model

def sim_tclab(X0, U, dt):
    X = np.empty((U.shape[0]+1, X0.shape[-1]))
    X[0, :] = X0
    for i in range(0, U.shape[0]):
        t_span = [i * dt, (i + 1) * dt]
        sol = solve_ivp(dTdt_TCLab, t_span, X[i, :], args=(U[i, :],), t_eval=[t_span[1]])
        X[i + 1, :] = sol.y[:, -1]
    return X

class TCLabDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.length = data["y0"].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return {
            "y0": self.data["y0"][idx],
            "u": self.data["u"][idx],
            "y": self.data["y"][idx],
        }

def format_data(
    df,
    x_scaler,
    u_scaler,
    train=False,
    begin_index=0,
    end_index=None,
    input_cols=["Q1", "Q2"],
    state_cols=["T1_noisy", "T2_noisy", "Ta_noisy"],
    context_length=1,
    prediction_length=60,
    stride=1,
    name=None,
    device=None,
    dtype=None,
):
    """ """
    if end_index is None:
        end_index = len(df)
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    X = df[state_cols].iloc[begin_index:end_index].to_numpy()
    U = df[input_cols].iloc[begin_index:end_index].to_numpy()

    if train:
        X = x_scaler.fit_transform(X)
        U = u_scaler.fit_transform(U)
    else:
        X = x_scaler.transform(X)
        U = u_scaler.transform(U)

    # convert to torch tensors
    X = torch.tensor(X, device=device, dtype=dtype)
    U = torch.tensor(U, device=device, dtype=dtype)

    # convert to dataset/sequences
    idx = (
        torch.arange(begin_index, end_index)
        .unfold(0, context_length + prediction_length, stride)
        .cpu()
        .numpy()
    )
    X = X.unfold(0, context_length + prediction_length, stride).permute(0, 2, 1)
    U = U.unfold(0, context_length + prediction_length, stride).permute(0, 2, 1)

    data = {
        "idx": idx,
        "y": X[:, context_length:, :],
        "y0": X[:, :context_length, :],
        "u": U[:, :-context_length, :],
    }
    return data

class tclab_measured_dataset:
    def __init__(
        self,
        df,
        Xscaler,
        Uscaler,
        train=True,
        begin_idx=0,
        end_idx=None,
        input_cols=["Q1", "Q2", "Ta"],
        state_cols=["T1", "T2"],
        context_length=1,
        prediction_length=60,
        t_interval=10,
        stride=3,
        dtype=None,
        device=None,
    ):
        if end_idx is None:
            end_idx = len(df)
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()
        self.Xscaler = Xscaler
        self.Uscaler = Uscaler
        self.df = df.iloc[begin_idx:end_idx].reset_index(drop=True)
        t_span = (
            self.df["dt"].shift(-(prediction_length + context_length)) - self.df["dt"]
        )
        N_MEASUREMENTS = (prediction_length + context_length)
        valid = (t_span - N_MEASUREMENTS * pd.Timedelta(seconds=t_interval)).abs() < pd.Timedelta(seconds=2)
        self.valid_indices = self.df[valid].index.to_numpy()
        broadcasted_indices = self.valid_indices[:, np.newaxis] + np.arange(61).reshape(
            1, -1
        )

        X = self.df.loc[:, state_cols].to_numpy()[broadcasted_indices, :]
        U = self.df.loc[:, input_cols].to_numpy()[broadcasted_indices, :]

        X_shape = X.shape
        U_shape = U.shape

        if train:
            X = self.Xscaler.fit_transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)
            U = self.Uscaler.fit_transform(U.reshape(-1, U_shape[-1])).reshape(U_shape)
        else:
            X = self.Xscaler.transform(X.reshape(-1, X_shape[-1])).reshape(X_shape)
            U = self.Uscaler.transform(U.reshape(-1, U_shape[-1])).reshape(U_shape)

        X = torch.tensor(X, device=device, dtype=dtype)
        U = torch.tensor(U, device=device, dtype=dtype)

        # rolling horizon
        # X = X.unfold(0, prediction_length + context_length, stride).permute(0, 2, 1)
        # U = U.unfold(0, prediction_length + context_length, stride).permute(0, 2, 1)
        
        # self.X0 = X[:, :context_length, :]
        # self.X = X[:, context_length:, :]
        # self.U = U[:, context_length:, :]
        self.dataset = {
            "idx": broadcasted_indices,
            "y": X[:, context_length:, :],
            "y0": X[:, :context_length, :],
            "u": U[:, :-context_length, :],
        }
        return
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        return {
            "y0": self.dataset["y0"][idx],
            "u": self.dataset["u"][idx],
            "y": self.dataset["y"][idx],
        }
    
    def rescale_x(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = self.Xscaler.inverse_transform(x)
        return x.reshape(x_shape)
    
    def rescale_u(self, u):
        if isinstance(u, torch.Tensor):
            u = u.detach().cpu().numpy()
        u_shape = u.shape
        u = u.reshape(-1, u_shape[-1])
        u = self.Uscaler.inverse_transform(u)
        return u.reshape(u_shape)
    
    def model_predictions(self, model):
        y0 = self.dataset["y0"]
        u = self.dataset["u"]
        y_pred = model(y0, u)
        return y_pred
    
    def evaluate_model(self, model):
        y0 = self.dataset["y0"]
        u = self.dataset["u"]
        y = self.dataset["y"]
        y_pred = model(y0, u)

        y0 = y0.to("cpu")
        u = u.to("cpu")
        y = y.to("cpu")
        y_pred = y_pred.detach().to("cpu")

        y0_shape = y0.shape
        u_shape = u.shape
        y_shape = y.shape
        y_pred_shape = y_pred.shape

        y0 = self.rescale_x(y0.numpy().reshape(-1, y0_shape[-1])).reshape(-1, y0_shape[-2], y0_shape[-1])
        u = self.rescale_u(u.numpy().reshape(-1, u_shape[-1])).reshape(-1, u_shape[-2], u_shape[-1])
        y = self.rescale_x(y.numpy().reshape(-1, y_shape[-1])).reshape(-1, y_shape[-2], y_shape[-1])
        y_pred = self.rescale_x(y_pred.numpy().reshape(-1, y_pred_shape[-1])).reshape(-1, y_pred_shape[-2], y_pred_shape[-1])

        L1e = np.mean(np.abs(y - y_pred), axis=(1, 2))
        L2e = np.sqrt(np.mean((y - y_pred) ** 2, axis=(1, 2)))

        results = {
            "y0": y0,
            "u": u,
            "y": y,
            "y_pred": y_pred,
            "L1e": L1e,
            "L2e": L2e,
        }
        return results

def rescale_data(data, model, Xscaler, Uscaler):
    y0 = data["y0"]
    u = data["u"]
    y = data["y"]
    y_pred = model(y0, u)

    y0 = y0.to("cpu")
    u = u.to("cpu")
    y = y.to("cpu")
    y_pred = y_pred.detach().to("cpu")

    y0_shape = y0.shape
    u_shape = u.shape
    y_shape = y.shape
    y_pred_shape = y_pred.shape

    y0 = Xscaler.inverse_transform(y0.numpy().reshape(-1, y0_shape[-1])).reshape(
        -1, y0_shape[-2], y0_shape[-1]
    )
    u = Uscaler.inverse_transform(u.numpy().reshape(-1, u_shape[-1])).reshape(
        -1, u_shape[-2], u_shape[-1]
    )
    y = Xscaler.inverse_transform(y.numpy().reshape(-1, y_shape[-1])).reshape(
        -1, y_shape[-2], y_shape[-1]
    )
    y_pred = Xscaler.inverse_transform(
        y_pred.numpy().reshape(-1, y_pred_shape[-1])
    ).reshape(-1, y_pred_shape[-2], y_pred_shape[-1])
    return y0, u, y, y_pred


def get_results_dict(model, dataset, x_scaler, u_scaler, prefix=""):
    y0, u, y, y_pred = rescale_data(dataset, model, x_scaler, u_scaler)
    if prefix != "":
        prefix = prefix + "_"
    results = {
        f"{prefix}y0": y0,
        f"{prefix}u": u,
        f"{prefix}y": y,
        f"{prefix}y_pred": y_pred,
        f"{prefix}L1e": np.mean(np.abs(y - y_pred), axis=(1, 2)),
        f"{prefix}L2e": np.sqrt(np.mean((y - y_pred) ** 2, axis=(1, 2))),
    }
    return results

default_config = {
    "lr": 1e-3,
    "batch_size": 64,
    "lr_patience": 20,
    "lr_factor": 0.5,
    "initialize_weights": "random",
    "scaling": True, # bool
    "log_frequency": None,
}

class TCLabTrainer(tune.Trainable):
    
    def setup(self, config, data):
        """config must include:
        """
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config, data)
        
    def _set_config(self, config):
        self.config = default_config.copy()
        self.config.update(config)
    
    def _init_model(self, config):
        self.model = Exogenous_RkRNN(state_dim=2, input_dim=2, hidden_size=32)
        if config["initialize_weights"] == "random":
            self.model.apply(init_weights)
        elif config["initialize_weights"] == "finetune":
            self.model.load_state_dict(torch.load(MODEL0_PATH))
        self.model = self.model.to(torch.get_default_device())
    
    def _init_optimizer(self, config):
        self.optimizer = maskedAdam(
            self.model.parameters(),
            lr=config.get("lr"),
            mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 1.0),
        )
        
    def _scheduler(self, config):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.get("lr_factor"),
            patience=config.get("lr_patience"),
        )
        
    def _setup(self, config, data):
        """the portion of setup that will be the same in child classes."""
        self.x_scaler = StandardScaler()
        self.u_scaler = StandardScaler()
        if config.get("scaling"):
            # self.x_scaler.fit(data["train_y"])
            # self.u_scaler.fit(data["train_u"].reshape(-1, 1))
            pass
        else:
            self.x_scaler.scale_ = np.ones(2)
            self.x_scaler.mean_ = np.zeros(2)
            self.u_scaler.scale_ = np.ones(2)
            self.u_scaler.mean_ = np.zeros(2)
        if config.get("x_scale", False):
            self.x_scaler.scale_ = config.get("x_scale")
        if config.get("x_mean", False):
            self.x_scaler.mean_ = config.get("x_mean")
        if config.get("u_scale", False):
            self.u_scaler.scale_ = config.get("u_scale")
        if config.get("u_mean", False):
            self.u_scaler.mean_ = config.get("u_mean")
        self.train_dataset = tclab_measured_dataset(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["train_begin_idx"],
            end_idx=config["train_end_idx"],
            train=True,
            input_horizon=1,
            output_horizon=60,
            name="train",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.val_dataset = tclab_measured_dataset(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["val_begin_idx"],
            end_idx=config["val_end_idx"],
            train=False,
            input_horizon=1,
            output_horizon=60,
            name="val",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.test_dataset = tclab_measured_dataset(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["test_begin_idx"],
            end_idx=config["test_end_idx"],
            train=False,
            input_horizon=1,
            output_horizon=60,
            name="test",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset["dataset"],
            batch_size=config.get("batch_size"),
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.initial_weights = get_parameter_vector(self.model).detach().cpu().numpy()
        self.data = data
        self.train_steps = 0
        self.batches_per_epoch = len(self.train_dataloader)
        self.total_batches = 0
        self.total_epochs = 0
        
    def _optimizer_step(self, batch):
        self.optimizer.zero_grad()
        y_pred = self.model(batch["y0"], batch["u"])
        loss = self.loss_fn(y_pred, batch["y1"])
        loss.backward()
        self.optimizer.masked_step()
        
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            train_results = self.train_dataset.evaluate_model(self.model)
            val_results = self.val_dataset.evaluate_model(self.model)
            test_results = self.test_dataset.evaluate_model(self.model)
        results = {
            "train_L1e": np.mean(train_results["train_L1e"]),
            "train_L2e": np.mean(train_results["train_L2e"]),
            "val_L1e": np.mean(val_results["val_L1e"]),
            "val_L2e": np.mean(val_results["val_L2e"]),
            "test_L1e": np.mean(test_results["test_L1e"]),
            "test_L2e": np.mean(test_results["test_L2e"]),
            "cosine_sim": cosine_similarity(
                self.initial_weights, get_parameter_vector(self.model).detach().cpu().numpy()
            ),  
        }
        return results
    
    def _next_batch(self):
        try:
            batch = next(self.train_dataloader_iter)
        except StopIteration:
            self.train_dataloader_iter = iter(self.train_dataloader)
            batch = next(self.train_dataloader_iter)
        return batch
    
    def step(self):
        self.model.train()
        step_len = self.config.get("log_frequency")
        if step_len is None:
            step_len = self.batches_per_epoch
        for _ in range(step_len):
            batch = self._next_batch()
            self._optimizer_step(batch)
        metrics = self.eval()
        self.scheduler.step(metrics["val_L2e"])
        self.total_batches += step_len
        metrics.update(
            {
                "training_iteration": self.total_batches,
                "time": self.total_epochs + step_len / self.batches_per_epoch,
            }
        )
        return metrics
    
    def save_optimizer_state(self, tmp_checkpoint_dir):
        """Saves the state of the optimizer."""
        torch.save(
            self.optimizer.state_dict(),
            Path(tmp_checkpoint_dir).joinpath(OPTIMIZER_FILENAME),
        )
        return

    def load_optimizer_state(self, tmp_checkpoint_dir):
        """Loads the state of the optimizer."""
        self.optimizer.load_state_dict(
            Path(tmp_checkpoint_dir).joinpath(OPTIMIZER_FILENAME)
        )
        return

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir).joinpath(MODEL_FILENAME)
        torch.save(self.model.state_dict(), checkpoint_path)
        # Save optimizer state if needed
        self.save_optimizer_state(tmp_checkpoint_dir)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir).joinpath(MODEL_FILENAME)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.load_optimizer_state(tmp_checkpoint_dir)
        return

    def reset_config(self, new_config):
        """Reset the configuration of the trainer."""
        self.setup(new_config, self.data)
        return True

    def cleanup(self):
        """Cleanup resources."""
        # No specific cleanup needed for this trainer
        self.save()
