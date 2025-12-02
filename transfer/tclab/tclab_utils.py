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
from sekf.optimizers import PersistentSampler, maskedAdam, SEKF
from sekf.utils import zip_dir

rng = np.random.default_rng(42)
colors = sns.color_palette("colorblind", 10)
GENERATED_DATA_FILENAME = "tclab_sim_data.csv"
MEASURED_DATA_FILENAME = "tclab_measured_data.csv"
MODEL_FILENAME = "tclab_model.pth"
MODEL0_PATH = Path("results").joinpath(MODEL_FILENAME)
OPTIMIZER_FILENAME = "tclab_optimizer.pth"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
ALL_TRIALS_BASE_FILENAME = "all_trials_results.csv"
BEST_RESULT_BASE_FILENAME = "best_result.csv"
RAY_STORAGE_PATH = "/home1/08940/joshuaeh/SCRATCH/Subset-Extended-Kalman-Filter/transfer/tclab/data/ray_results"

MEASUREMENTS_PER_DAY = 6 * 60 * 24  # 8640
INITIALIZATION = {
    "retrain": "random",
    "finetune": "finetune",
}
DAYS = 5
DATA_DIM = {
    "0.25hr": 15 * 6,
    "1hr": 60 * 6,
    "4hr": 4 * 60 * 6,
    "12hr": 12 * 60 * 6,
    "24hr": 24 * 60 * 6,
}


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
            "y1": self.data["y1"][idx],
        }

def format_data(
    df,
    x_scaler,
    u_scaler,
    begin_idx=0,
    end_idx=None,
    input_cols=["Q1", "Q2", "Ta"],
    # state_cols=["T1_noisy", "T2_noisy", "Ta_noisy"],
    state_cols=["T1", "T2"],
    context_length=1,
    prediction_length=60,
    stride=1,
    name=None,
    device=None,
    dtype=None,
):
    """ """
    if end_idx is None:
        end_idx = len(df)
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()

    X = df[state_cols].iloc[begin_idx:end_idx].to_numpy()
    U = df[input_cols].iloc[begin_idx:end_idx].to_numpy()

    
    X = x_scaler.transform(X)
    U = u_scaler.transform(U)

    # convert to torch tensors
    X = torch.tensor(X, device=device, dtype=dtype)
    U = torch.tensor(U, device=device, dtype=dtype)

    # convert to dataset/sequences
    idx = (
        torch.arange(begin_idx, end_idx)
        .unfold(0, context_length + prediction_length, stride)
        .cpu()
        .numpy()
    )
    X = X.unfold(0, context_length + prediction_length, stride).permute(0, 2, 1)
    U = U.unfold(0, context_length + prediction_length, stride).permute(0, 2, 1)

    data = {
        "y1": X[:, context_length:, :],
        "y0": X[:, :context_length, :],
        "u": U[:, :-context_length, :],
    }
    return data

def rescale_data(data, model, Xscaler, Uscaler):
    y0 = data["y0"]
    u = data["u"]
    y = data["y1"]
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

@torch.no_grad()
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
    "batch_size": 64,
    "initialize_weights": "random",  # or "finetune"
    "max_epochs": 1000,
    "lr": 1e-3,
    "lr_patience": 20,
    "lr_factor": 0.5,
    "scaling": True,
    "mask_fn_quantile_thresh": None,
    "N_batches_per_step": 1,
}

class TCLabTrainer(tune.Trainable):
    
    def setup(self, config):
        """
        config must include:
        - initialize_weights: "random" or "finetune"
        - train_begin_idx: (int)
        - train_end_idx: (int)
        - val_begin_idx: (int)
        - val_end_idx: (int)
        - test_begin_idx: (int)
        - test_end_idx: (int)
        - data_ref: ray object ref to dataframe
        config may include:
        - lr: learning rate (float) (1e-3)
        - batch_size: (int) (64)
        - lr_patience: (int) (20)
        - lr_factor: (float) (0.5)
        - scaling: (bool) whether to scale data (True)
        - mask_fn_quantile_thresh: (float) quantile threshold for masking function (1.0)
        - N_batches_per_step: (int) number of batches per step (1)
        - train_dataset_stride: (int) stride for training dataset (1)
        """
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config)
        
    def _set_config(self, config):
        self.config = default_config.copy()
        self.config.update(config)
    
    def _init_model(self, config):
        self.model = Exogenous_RkRNN(state_dim=2, input_dim=3, hidden_size=32)
        if config["initialize_weights"] == "random":
            self.model.apply(init_weights)
        elif config["initialize_weights"] == "finetune":
            self.model.load_state_dict(torch.load(MODEL0_PATH))
        else:
            raise ValueError("initialize_weights must be 'random' or 'finetune'")
        self.model = self.model.to(torch.get_default_device())
    
    def _init_optimizer(self, config):
        self.optimizer = maskedAdam(
            self.model.parameters(),
            lr=config.get("lr", 1e-3),
            mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 1.0),
        )
        
    def _scheduler(self, config):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 20),
        )
        
    def _scaler_config(self, config, data):
        # initialize and fit scalers
        self.x_scaler = StandardScaler()
        self.u_scaler = StandardScaler()
        if config.get("scaling"):
            self.x_scaler.fit(data.iloc[config["train_begin_idx"]:config["train_end_idx"]][["T1", "T2"]].to_numpy())
            self.u_scaler.fit(data.iloc[config["train_begin_idx"]:config["train_end_idx"]][["Q1", "Q2", "Ta"]].to_numpy())
            pass
        else:
            self.x_scaler.scale_ = np.ones(2)
            self.x_scaler.mean_ = np.zeros(2)
            self.u_scaler.scale_ = np.ones(3)
            self.u_scaler.mean_ = np.zeros(3)
        if config.get("x_scale", False):
            self.x_scaler.scale_ = config.get("x_scale")
        if config.get("x_mean", False):
            self.x_scaler.mean_ = config.get("x_mean")
        if config.get("u_scale", False):
            self.u_scaler.scale_ = config.get("u_scale")
        if config.get("u_mean", False):
            self.u_scaler.mean_ = config.get("u_mean")
        
    def _setup(self, config, data):
        """the portion of setup that will be the same in child classes."""
        # create datasts and dataloaders
        data = ray.get(self.config["data_ref"])
        self._scaler_config(config, data)
        self.train_dataset = format_data(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["train_begin_idx"],
            end_idx=config["train_end_idx"],
            stride=config.get("train_dataset_stride", 1),
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.val_dataset = format_data(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["val_begin_idx"],
            end_idx=config["val_end_idx"],
            stride=config.get("val_dataset_stride", 1),
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.test_dataset = format_data(
            data,
            self.x_scaler,
            self.u_scaler,
            begin_idx=config["test_begin_idx"],
            end_idx=config["test_end_idx"],
            stride=config.get("test_dataset_stride", 1),
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.train_sampler = PersistentSampler(
            self.train_dataset["y0"].shape[0],
            seed=42
        )
        self.initial_weights = get_parameter_vector(self.model).detach().cpu().numpy()
        self.data = data
        
    def _optimizer_step(self, batch):
        self.optimizer.zero_grad()
        y_pred = self.model(batch["y0"], batch["u"])
        loss = self.loss_fn(y_pred, batch["y1"])
        loss.backward()
        self.optimizer.masked_step()
        
        
    def eval(self):
        self.model.eval()
        with torch.no_grad():
            train_results = get_results_dict(self.model, self.train_dataset, self.x_scaler, self.u_scaler, prefix="train")
            val_results = get_results_dict(self.model, self.val_dataset, self.x_scaler, self.u_scaler, prefix="val")
            test_results = get_results_dict(self.model, self.test_dataset, self.x_scaler, self.u_scaler, prefix="test")
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
    
    def step(self):
        self.model.train()
        for _ in range(self.config.get("N_batches_per_step", 1)):
            batch_idx = self.train_sampler.get_batch_indices(self.config.get("batch_size", 64))
            batch = {k:v[batch_idx].to(torch.get_default_device()) for k,v in self.train_dataset.items()}
            self._optimizer_step(batch)
        metrics = self.eval()
        self.scheduler.step(metrics["val_L2e"])
        metrics.update(
            {
                "effective_epochs": self.train_sampler.effective_epochs,
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
    
    def save_sampler_state(self, tmp_checkpoint_dir):
        """Saves the state of the sampler."""
        torch.save(
            self.train_sampler.state_dict(),
            Path(tmp_checkpoint_dir).joinpath("sampler_state.pth")
        )
        return
    
    def load_sampler_state(self, tmp_checkpoint_dir):
        """Loads the state of the sampler."""
        self.train_sampler.load_state_dict(
            torch.load(
                Path(tmp_checkpoint_dir).joinpath("sampler_state.pth")
            )
        )
        return

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir).joinpath(MODEL_FILENAME)
        torch.save(self.model.state_dict(), checkpoint_path)
        # Save optimizer state if needed
        self.save_optimizer_state(tmp_checkpoint_dir)
        self.save_sampler_state(tmp_checkpoint_dir)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = Path(tmp_checkpoint_dir).joinpath(MODEL_FILENAME)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.load_optimizer_state(tmp_checkpoint_dir)
        self.load_sampler_state(tmp_checkpoint_dir)
        return

    def reset_config(self, new_config):
        """Reset the configuration of the trainer."""
        self.setup(new_config, self.data)
        return True

    def cleanup(self):
        """Cleanup resources."""
        # No specific cleanup needed for this trainer
        self.save()

default_config_SEKF = {
    "batch_size": 1,
    "initialize_weights": "finetune",  # or "random"
    "max_epochs": 1000,
    "mask_fn_quantile_thresh": None,
    "R": 0.1,
    "Q": 0.1,
    "p0": 100.0,
    "log_frequency": None,
}

class TCLabTrainer_SEKF(TCLabTrainer):
    """Trainer for SEKF specifically."""

    def _set_config(self, config):
        self.config = default_config_SEKF | config
        return

    def _init_optimizer(self, config):
        self.optimizer = SEKF(
            self.model.parameters(),
            R=config.get("R"),
            p0=config.get("p0"),
            q=config.get("Q"),
            mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh"),
        )

    def _scheduler(self, config):
        return SEKF_scheduler(self.optimizer)

    def setup(self, config):
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config)

    def _optimizer_step(self, batch):
        """Performs a single step of the SEKF optimizer."""
        y_pred = self.model(batch["y0"], batch["u"])
        e = batch["y1"] - y_pred
        if torch.isnan(e).any():
            self.reset()

        if self.config.get("mask_fn_quantile_thresh", None) is not None:
            loss = self.loss_fn(y_pred, batch["y1"])
            loss.backward()
            grad_loss = get_parameter_gradient_vector(self.model)
            mask = mask_fn(grad_loss, self.config.get("mask_fn_quantile_thresh"))
        else:
            mask = None
        J = get_jacobian(self.model, (batch["y0"], batch["u"]))
        self.optimizer.step(e, J, mask=mask)

default_config_LBFGS = {
    "batch_size": 1,
    "initialize_weights": "finetune",  # or "random"
    "max_epochs": 500,
    "lr": 1.0,
    "lr_max_iter": 20,
    "lr_history_size": 10,
    "lr_patience": 20,
    "lr_factor": 0.5,
}

class TCLabTrainer_LBFGS(TCLabTrainer):
    """Trainer for LBFGS specifically."""

    def _set_config(self, config):
        self.config = default_config_LBFGS | config
        return

    def _init_optimizer(self, config):
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=config.get("lr", 1.0),
            max_iter=config.get("lr_max_iter", 20),
            history_size=config.get("lr_history_size", 10),
        )

    def _scheduler(self, config):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=self.config.get("lr_patience"),
            factor=self.config.get("lr_factor"),
        )

    def setup(self, config):
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config)

    def _optimizer_step(self, batch):
        """Performs a single step of the LBFGS optimizer."""

        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(batch["y0"], batch["u"])
            loss = self.loss_fn(y_pred, batch["y1"])
            if torch.isnan(loss).any():
                self.reset()
            loss.backward()
            return loss

        self.optimizer.step(closure)


class SEKF_scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lowest_val_loss = float("inf")
        self.patience_counter = 0

    def step(self, val_loss):
        if val_loss < self.lowest_val_loss:
            self.lowest_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= 20:
            pass
            # Implement logic to reduce Q or adjust optimizer parameters

    def get_last_lr(self):
        return [1.0]
