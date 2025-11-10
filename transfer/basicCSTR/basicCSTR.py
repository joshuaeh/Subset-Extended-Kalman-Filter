from pathlib import Path
import shutil
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from scipy.integrate import solve_ivp
import scipy.stats as stats

import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from tqdm import tqdm

# torch.set_default_dtype(torch.float64)
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# )
# # device = torch.device("cpu")
# print(f"Using device: {device}")
# torch.set_default_device(device)

from sekf.modeling import Exogenous_RkRNN, init_weights, get_jacobian, seed_worker, g, get_parameter_vector
from sekf.optimizers import SEKF, maskedAdam
from sekf.utils import zip_dir

default_rng = np.random.default_rng(42)
# ----- plotting parameters -----
default_rng = np.random.default_rng(42)
colors = sns.color_palette("colorblind", 10)
color_palette = sns.color_palette("colorblind")
sns.set_palette(color_palette)
# mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color_palette)
# sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_style({"xtick.bottom": True, "ytick.left": True})
plt.rcParams["figure.constrained_layout.use"] = True

# plt.rc("font", family="Arial")
# plt.rc("font", family="sans-serif", size=12)
# plt.rc("axes", labelsize=7)
# plt.rc("legend", fontsize=7)
# plt.rc("xtick", labelsize=5)
# plt.rc("ytick", labelsize=5)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = 7
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
# Set font as TrueType
plt.rcParams["pdf.fonttype"] = 42

# plt.rc("savefig", dpi=1_000, bbox="tight", pad_inches=0.01)
plt.rc("savefig", dpi=1_000)
# ----- end plotting parameters -----

# constants
RAY_STORAGE_PATH = "/home1/08940/joshuaeh/SCRATCH/Subset-Extended-Kalman-Filter/transfer/basicCSTR/data/ray_results"

CASE_DIR = Path(__file__).parent
DATA_DIR = CASE_DIR.joinpath("data")
RESULTS_DIR = CASE_DIR.joinpath("results")
TRAINING_DATA_PATH = DATA_DIR.joinpath("training_data.npz")
TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
MODEL_FILENAME = "model.pth"
OPTIMIZER_FILENAME = "optimizer.pth"
ALL_TRIALS_BASE_FILENAME = "allTrials_metrics.csv"
BEST_RESULT_BASE_FILENAME = "bestResult_metrics.csv"
MODEL0_PATH = RESULTS_DIR.joinpath("training", MODEL_FILENAME)
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

FILE_ENDING_MAPPING = {
    "metrics": "_metrics.csv",
    "weights": "_weights.npz",
    "model": ".pth",
}
CASE_NICENAME_MAPPING = {
    "NN_retrain_adam": "",
    "NN_transfer_adam": "",
    "NN_transfer_SEKF": "",
}

constants = {
    "F": 0.6,  # Feed flow rate (m^3/s)
    "V": 15.0,  # Volume of the reactor (m^3)
    "k1": 2e-1,  # Reaction rate constant for A -> B (1/s)
    "k2f": 5e-1,  # Forward reaction rate constant for B -> C (1/s)
    "k2r": 1e-1,  # Reverse reaction rate constant for B -> C (1/s)
}

x_ss = np.array([0.3333, 0.4802, 0.39548609])  # SS for Caf=2.0

# plotting kwargs
SPECIES_LABELS = [
    r"$C_A$",
    r"$C_B$",
    r"$C_C$",
]
C0_KWARGS = dict(
    facecolor="none",
    s=20,
    linewidth=0.5,
    marker=".",
)
CA0_KWARGS = {
    **C0_KWARGS,
    "edgecolor": colors[0],
    "label": r"$C_{A0}$",
}
CB0_KWARGS = {
    **C0_KWARGS,
    "edgecolor": colors[1],
    "label": r"$C_{B0}$",
}
CC0_KWARGS = {
    **C0_KWARGS,
    "edgecolor": colors[2],
    "label": r"$C_{C0}$",
}
C_KWARGS = dict(
    s=2,
)
CA_KWARGS = {
    **C_KWARGS,
    "color": colors[0],
    "label": r"$C_A$",
}
CB_KWARGS = {
    **C_KWARGS,
    "color": colors[1],
    "label": r"$C_B$",
}
CC_KWARGS = {
    **C_KWARGS,
    "color": colors[2],
    "label": r"$C_C$",
}
CPRED_KWARGS = {}
CA_PRED_KWARGS = {
    **CPRED_KWARGS,
    "color": colors[0],
    "label": r"$\hat{C}_A$",
}
CB_PRED_KWARGS = {
    **CPRED_KWARGS,
    "color": colors[1],
    "label": r"$\hat{C}_B$",
}
CC_PRED_KWARGS = {
    **CPRED_KWARGS,
    "color": colors[2],
    "label": r"$\hat{C}_C$",
}


def sim(
    func,
    t_span,
    X0,
    U,
    constants,
    noise=None,
):
    if isinstance(U, (int, float)):
        U = np.full((len(t_span), 1), U)
    X = np.zeros((len(t_span), len(X0)))
    X[0, :] = X0
    for i in range(1, len(t_span)):
        X[i] = solve_ivp(
            func,
            (t_span[i - 1], t_span[i]),
            X[i - 1],
            args=(U[i - 1], constants),
            t_eval=[t_span[i]],
        ).y.T[0]
    if noise is not None:
        assert noise.shape == X0.shape, (
            "Noise shape must match state shape, got {} and {}".format(
                noise.shape, X0.shape
            )
        )
        Y = X + default_rng.normal(np.zeros_like(X0), noise, size=X.shape)
    else:
        Y = X
    return {
        "X0": X0,
        "X": X,
        "Y": Y,
        "U": U,
    }


def get_U(t_span, umin, umax, len_step):
    U = np.zeros(len(t_span))
    for i in range(0, len(t_span), len_step):
        U[i : i + len_step] = default_rng.uniform(umin, umax)
    return U


def dXdt(t, X, U, constants):
    """
    Compute the time derivative of the state vector X.

    Parameters:
    X : np.ndarray
        State vector [Ca, Cb, Cc].
    U : np.ndarray
        Input vector [Caf].

    Returns:
    dXdt : np.ndarray
        Time derivative of the state vector.
    """

    Ca, Cb, Cc = X

    r1 = constants["k1"] * Ca
    r2 = constants["k2f"] * Cb**3 - constants["k2r"] * Cc
    dCadt = constants["F"] / constants["V"] * (U - Ca) - r1
    dCbdt = -constants["F"] / constants["V"] * Cb + r1 - 3 * r2
    dCcdt = -constants["F"] / constants["V"] * Cc + r2
    return [dCadt, dCbdt, dCcdt]


def rescale_data(data, model, Xscaler, Uscaler):
    y0 = data["y0"]
    u = data["u"]
    y1 = data["y1"]
    y_pred = model(y0, u)

    y0 = y0.to("cpu")
    u = u.to("cpu")
    y1 = y1.to("cpu")
    y_pred = y_pred.to("cpu")

    y0 = Xscaler.inverse_transform(y0.numpy().reshape(-1, 3))
    u = Uscaler.inverse_transform(u.squeeze().numpy())
    y1 = Xscaler.inverse_transform(y1.numpy().reshape(-1, 3)).reshape(-1, 60, 3)
    y_pred = Xscaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 3)).reshape(
        -1, 60, 3
    )
    t = np.arange(0, y_pred.shape[0], 1)
    return y0, u, y1, y_pred, t

class CSTRDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for CSTR data.
    
    Args:
        data (dict):
        dict containing "y0", "y", and "u" as keys with corresponding numpy arrays or tensors.
    """
    def __init__(self, data):
        self.data = data
        self.length = data["y0"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {
            "y0": self.data["y0"][index],
            "u": self.data["u"][index],
            "y1": self.data["y1"][index],
        }


def format_dataset(
    data,
    Xscaler,
    Uscaler,
    train=True,
    begin_index=0,
    end_index=-1,
    input_horizon=2,
    output_horizon=60,
    stride=1,
    name=None,
    device=None,
    dtype=None,
):
    # take advantage of all data, don't throw boundary between train/validate/test away
    # TODO: finish implementation
    _end_index = (
        end_index
        if end_index == -1
        else min(end_index + output_horizon, data["Y"].shape[0])
    )
    # U needs to be 2d
    if data["U"].ndim == 1:
        data["U"] = data["U"].reshape(-1, 1)
    # scaling
    if train:
        X = torch.tensor(
            Xscaler.fit_transform(data["Y"][begin_index:end_index, :]),
            dtype=torch.float32,
        )
        U = torch.tensor(
            Uscaler.fit_transform(data["U"][begin_index:end_index, :]),
            dtype=torch.float32,
        )
    else:
        X = torch.tensor(
            Xscaler.transform(data["Y"][begin_index:end_index, :]), dtype=torch.float32
        )
        U = torch.tensor(
            Uscaler.transform(data["U"][begin_index:end_index, :]), dtype=torch.float32
        )
    # rolling horizon
    X = X.unfold(0, output_horizon + input_horizon, stride).permute(
        0, 2, 1
    )  # Nbatches, Nsteps, Nx
    U = U.unfold(0, output_horizon + input_horizon, stride).permute(0, 2, 1)[
        :, :, :
    ]  # Nbatches, Nsteps, Nu
    # format into dictionary
    dataset = {
        "y1": X[:, input_horizon:, :],
        "y0": X[:, :input_horizon, :],
        "u": U[:, :-input_horizon, :],
    }
    # send to device
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()
    for k, v in dataset.items():
        if isinstance(v, torch.Tensor):
            dataset[k] = v.to(device=device, dtype=dtype)
        else:
            dataset[k] = torch.tensor(v, device=device, dtype=dtype)
    if name is not None:
        dataset["name"] = name
        
    dataset["dataset"] = CSTRDataset(dataset)
    return dataset


def np_load_safe(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as data:
            if isinstance(data, dict):
                return {k: v for k, v in data.items()}
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, np.lib.npyio.NpzFile):
                return dict(data)
            else:
                return data
    elif path.endswith(".npy"):
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(
            "File must be a .npz or .npy file, got {}".format(os.path.basename(path))
        )


def load_model(case=None, model_path=None, device=None, dtype=None):
    """ """
    model = Exogenous_RkRNN(state_dim=3, input_dim=1, hidden_size=16)
    if device is None:
        device = torch.get_default_device()
    if dtype is None:
        dtype = torch.get_default_dtype()
    if case is not None and model_path is not None:
        raise ValueError("Provide either 'case' or 'model_path', or none. not both.")
    if case is None and model_path is None:
        model_path = MODEL0_PATH
    if case is not None:
        model_path = case + ".pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device=device, dtype=dtype)
    return model


def cosine_similarity(a, b):
    """computes row-wise cosine similarity."""
    assert a.ndim == b.ndim
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if a.ndim == 2:
        return np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )


def get_results_dict(model, dataset, x_scaler, u_scaler, prefix=""):
    xn, u, y, y_pred, t = rescale_data(dataset, model, x_scaler, u_scaler)
    if prefix != "":
        prefix = prefix + "_"
    results = {
        f"{prefix}xn": xn,
        f"{prefix}u": u,
        f"{prefix}y": y,
        f"{prefix}y_pred": y_pred,
        f"{prefix}t": t,
        f"{prefix}L1e": np.mean(np.abs(y - y_pred), axis=(1, 2)),
        f"{prefix}L2e": np.sqrt(np.mean((y - y_pred) ** 2, axis=(1, 2))),
    }

    return results


def get_results(model, x_scaler, u_scaler, *datasets):
    results = {}
    for dataset in datasets:
        results.update(
            get_results_dict(
                model,
                dataset,
                x_scaler,
                u_scaler,
                prefix=dataset["name"],
            )
        )
    return results


def get_case_results(case, x_scaler, u_scaler, *datasets):
    """ """
    model = load_model(case=case)
    results = get_results(model, x_scaler, u_scaler, *datasets)
    results["final_weights"] = (
        torch.cat([v.reshape(-1) for k, v in model.named_parameters()])
        .detach()
        .cpu()
        .numpy()
    )
    if os.path.exists(case + "_metrics.csv"):
        df = pd.read_csv(case + "_metrics.csv")
        results["metrics"] = df
        results["time"] = df["time"].values[-1]
    if os.path.exists(case + "_weights.npz"):
        weights = np_load_safe(case + "_weights.npz")
        weights = next(iter(weights.values()))
        results["weights"] = weights
    return results

# --- Ray tune stuff ---
default_config = {
    "lr": 1e-3,
    "batch_size": 64,
    "lr_patience": 20,
    "lr_factor": 0.5,
    "initialize_weights": "random",
    "scaling": True, # bool
    "log_frequency": None,
}

class BasicCSTRTrainer(tune.Trainable):
    """Trainer for basic CSTR model using Ray Tune."""
    
    def setup(self, config, data):
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
        self.model = Exogenous_RkRNN(state_dim=3, input_dim=1, hidden_size=32)
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
            self.x_scaler.fit(data["train_y"])
            self.u_scaler.fit(data["train_u"].reshape(-1, 1))
        else:
            self.x_scaler.scale_ = np.ones(3)
            self.x_scaler.mean_ = np.zeros(3)
            self.u_scaler.scale_ = np.ones(1)
            self.u_scaler.mean_ = np.zeros(1)
        if config.get("x_scale", False):
            self.x_scaler.scale_ = config.get("x_scale")
        if config.get("x_mean", False):
            self.x_scaler.mean_ = config.get("x_mean")
        if config.get("u_scale", False):
            self.u_scaler.scale_ = config.get("u_scale")
        if config.get("u_mean", False):
            self.u_scaler.mean_ = config.get("u_mean")
        self.train_dataset = format_dataset(
            {
                "Y": data["train_y"],
                "U": data["train_u"],
            },
            self.x_scaler,
            self.u_scaler,
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="train",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.val_dataset = format_dataset(
            {
                "Y": data["val_y"],
                "U": data["val_u"],
            },
            self.x_scaler,
            self.u_scaler,
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="val",
            device=torch.get_default_device(),
            dtype=torch.get_default_dtype(),
        )
        self.test_dataset = format_dataset(
            {
                "Y": data["test_y"],
                "U": data["test_u"],
            },
            self.x_scaler,
            self.u_scaler,
            train=False,
            input_horizon=1,
            output_horizon=60,
            stride=5,
            name="test",
            device=torch.get_default_device(),
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
            train_results = get_results_dict(
                self.model,
                self.train_dataset,
                self.x_scaler,
                self.u_scaler,
                prefix="train",
            )
            val_results = get_results_dict(
                self.model,
                self.val_dataset,
                self.x_scaler,
                self.u_scaler,
                prefix="val",
            )
            test_results = get_results_dict(
                self.model,
                self.test_dataset,
                self.x_scaler,
                self.u_scaler,
                prefix="test",
            )
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

class CSTRTrainer_SEKF(BasicCSTRTrainer):
    """Trainer for SEKF specifically."""

    def _set_config(self, config):
        self.config = default_config_SEKF | config
        return

    def _init_optimizer(self, config):
        self.optimizer = SEKF(
            self.model.parameters(),
            R=config.get("R"),
            p0=config.get("p0", 100),
            q=config.get("Q"),
            mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 0.0),
        )

    def _scheduler(self, config):
        return SEKF_scheduler(self.optimizer)

    def setup(self, config, data):
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config, data)

    def _optimizer_step(self, x_batch, y_batch):
        """Performs a single step of the SEKF optimizer."""
        y_pred = self.model(x_batch)
        e = y_batch - y_pred
        if torch.isnan(e).any():
            self.reset()

        if self.config.get("mask_fn_quantile_thresh", None) is not None:
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            grad_loss = get_parameter_gradient_vector(self.model)
            mask = mask_fn(grad_loss, self.config.get("mask_fn_quantile_thresh"))
        else:
            mask = None
        J = get_jacobian(self.model, (x_batch))
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

class CSTRTrainer_LBFGS(BasicCSTRTrainer):
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

    def setup(self, config, data):
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config, data)

    def _optimizer_step(self, x_batch, y_batch):
        """Performs a single step of the LBFGS optimizer."""

        def closure():
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_pred, y_batch)
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