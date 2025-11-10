# Damped Spring Model utility functions and constants

# TODO:

##### Imports
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from scipy.integrate import solve_ivp
from sekf.modeling import (
    cosine_similarity,
    get_jacobian,
    get_parameter_vector,
    get_parameter_gradient_vector,
    init_weights,
    mask_fn,
    seed_worker,
)
from sekf.optimizers import SEKF, maskedAdam
from tqdm import tqdm

##### Constants
os.environ["TUNE_WARN_EXCESSIVE_EXPERIMENT_CHECKPOINT_SYNC_THRESHOLD_S"] = "0"

torch.manual_seed(42)
np.random.seed(42)
g = torch.Generator()
g.manual_seed(42)

RAY_STORAGE_PATH = "/home1/08940/joshuaeh/SCRATCH/Subset-Extended-Kalman-Filter/transfer/dampedSpring/data/ray_results"

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR.joinpath("data")
# DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = EXPERIMENT_DIR.joinpath("results")
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILENAME = "model_weights.pth"
OPTIMIZER_FILENAME = "optimizer_state.pth"
# SEKF_FILENAME = "sekf_optimizer_state.npz"
MODEL0_WEIGHTS_PATH = RESULTS_DIR.joinpath("training", MODEL_FILENAME)
TRAINING_DATA_PATH = DATA_DIR.joinpath("training_data.npz")
TRANSFER_DATA_PATH = DATA_DIR.joinpath("transfer_data.npz")
ALL_TRIALS_BASE_FILENAME = "allTrials_metrics.csv"
BEST_RESULT_BASE_FILENAME = "bestResult_metrics.csv"


N_SOURCE_TRAIN = 80_000
N_SOURCE_VALIDATION = 10_000
N_SOURCE_TEST = 10_000
N_SOURCE_TOTAL = N_SOURCE_TRAIN + N_SOURCE_VALIDATION + N_SOURCE_TEST
N_TARGET_TOTAL = 110_000

MEASUREMENT_NOISE_STD = 0.1

INIT_METHODS = ["retrain", "finetune", "noMaintenance"]
OPTIMIZERS = {
    "adam": maskedAdam,
    "sekf": SEKF,
    "bfgs": torch.optim.LBFGS,
}
SCENARIOS = [
    {"m": 1.1},
    {"m": 0.9},
    {"c": 0.55},
    {"c": 0.45},
    {"k": 1.1},
    {"k": 0.9},
    {"u": 1.0},
    {"u": -1.0},
]
TARGET_DATA_DIM = [10, 50, 100, 500, 1000]
N_ITERATIONS = 10


N_CPUS = 4
N_HYPERPARAMETER_TRIALS = 50

# ray.init(ignore_reinit_error=True)
# assert ray.is_initialized()

rng = np.random.default_rng(42)


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 20)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(x)
        x = self.fc4(x)
        return x


default_config = {
    "batch_size": 64,
    "initialize_weights": "random",  # or "finetune"
    "lr": 1e-3,
    "lr_patience": 20,
    "lr_factor": 0.5,
    "max_epochs": 1000,
    "mask_fn_quantile_thresh": None,
    "log_frequency": None,
}

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


default_config_LBFGS = {
    "batch_size": 1,
    "initialize_weights": "finetune",  # or "random"
    "max_epochs": 1000,
    "lr": 1.0,
    "lr_max_iter": 20,
    "lr_history_size": 10,
    "lr_patience": 20,
    "lr_factor": 0.5,
    "log_frequency": None,
}


class DampedSpringTrainer(tune.Trainable):
    """Trainer for the Damped Spring model using Ray Tune.
    config (dict): Configuration dictionary containing hyperparameters.
        - "batch_size": Batch size for training.
        - "initialize_weights": Method to initialize weights, can be "random" or "finetune".
        - "lr": Learning rate for the optimizer.
        - "lr_patience": Patience for the learning rate scheduler.
        - "lr_factor": Factor by which the learning rate is reduced. to not change lr, set factor to 1.0
        - "max_epochs": Maximum number of epochs for training.
        - "mask_fn_quantile_thresh": Threshold for the masked Adam optimizer.
        - "optimizer": Type of optimizer to use, can be "adam" or "sekf".
        - "sekf_q": Parameter q for the SEKF optimizer.
        - "sekf_p0": Initial value for the SEKF optimizer.
        - "sekf_save_path": Path to save the SEKF optimizer state.

    data (dict): Dictionary containing training, validation, and test data.

    """

    def setup(self, config, data):
        self._set_config(config)
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = self._scheduler(self.config)
        self._setup(config, data)

    def _set_config(self, config):
        self.config = default_config | config
        return

    def _init_model(self, config):
        self.model = MLP()
        match config.get("initialize_weights"):
            case "random":
                self.model.apply(init_weights)
            case "finetune":
                # load pre-trained weights
                weights_dict = torch.load(MODEL0_WEIGHTS_PATH, weights_only=True)
                self.model.load_state_dict(weights_dict)

    def _init_optimizer(self, config):
        self.optimizer = maskedAdam(
            self.model.parameters(),
            lr=config.get("lr"),
            mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 1.0),
        )

    def _scheduler(self, config):
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            "min",
            patience=self.config.get("lr_patience"),
            factor=self.config.get("lr_factor"),
        )

    def _setup(self, config, data):
        """The portion that is common to all optimizers."""
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data["train_x"], data["train_y"]),
            batch_size=self.config["batch_size"],
            shuffle=True,
            # num_workers=1,
            # persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        self.train_dataloader_iter = iter(self.train_dataloader)
        self.initial_weights = get_parameter_vector(self.model).detach().numpy()
        self.data = data
        self.train_steps = 0
        self.batches_per_epoch = len(self.train_dataloader)
        self.total_batches = 0
        self.total_epochs = 0

    def _optimizer_step(self, x_batch, y_batch):
        """Performs a single step of the masked Adam optimizer."""
        self.optimizer.zero_grad()
        y_pred = self.model(x_batch)

        loss = self.loss_fn(y_pred, y_batch)
        # end experiment if loss is nan
        if torch.isnan(y_pred).any():
            self.reset()
        loss.backward()
        self.optimizer.masked_step()

    def eval(self, data):
        self.model.eval()
        with torch.no_grad():
            metrics = {
                "train_loss": self.loss_fn(
                    self.model(data["train_x"]), data["train_y"]
                ).item(),
                "val_loss": self.loss_fn(
                    self.model(data["val_x"]), data["val_y"]
                ).item(),
                "test_loss": self.loss_fn(
                    self.model(data["test_x"]), data["test_y"]
                ).item(),
                "cosine_similarity_weights": cosine_similarity(
                    self.initial_weights,
                    get_parameter_vector(self.model).detach().numpy(),
                ),
                "lr": self.scheduler.get_last_lr()[0],
            }
        self.scheduler.step(metrics["val_loss"])
        return metrics

    def _next_batch(self):
        try:
            x_batch, y_batch = next(self.train_dataloader_iter)
        except StopIteration:
            self.train_dataloader_iter = iter(self.train_dataloader)
            x_batch, y_batch = next(self.train_dataloader_iter)
        return x_batch, y_batch

    def step(self):
        self.model.train()
        step_len = self.config.get("log_frequency", None)
        if step_len is None:
            step_len = self.batches_per_epoch
        for _ in range(step_len):
            x_batch, y_batch = self._next_batch()
            self._optimizer_step(x_batch, y_batch)
            # self.total_batches += 1
        metrics = self.eval(self.data)
        self.total_batches += step_len
        self.scheduler.step(metrics["val_loss"])
        metrics.update(
            {
                "total_batches": self.total_batches,
                "total_epochs": self.total_batches // self.batches_per_epoch,
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
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        torch.save(self.model.state_dict(), checkpoint_path)
        # Save optimizer state if needed
        self.save_optimizer_state(tmp_checkpoint_dir)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
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


class DampedSpringTrainer_SEKF(DampedSpringTrainer):
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


class DampedSpringTrainer_LBFGS(DampedSpringTrainer):
    """Trainer for LBFGS specifically."""

    def _set_config(self, config):
        self.config = default_config_LBFGS | config
        return

    def _init_optimizer(self, config):
        self.optimizer = optim.LBFGS(
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


# Mass-Spring-Damper system simulation
# m dxdx + c dx + kx = u
def mass_spring_damper(t, state, m, c, k, u):
    x, x_dot = state
    x_ddot = (u - c * x_dot - k * x) / m
    return [x_dot, x_ddot]


def sim_mass_spring_damper(
    x0=0.0, x_dot0=0.0, m=1.0, c=0.5, k=1.0, u=0.0, t_end=20.0, t_eval=None
):
    initial_state = [x0, x_dot0]  # Initial position and velocity
    params = (m, c, k, u)
    # Integrate the ODE
    res = solve_ivp(
        mass_spring_damper, [0, t_end], initial_state, args=params, t_eval=t_eval
    )
    return res.t, res.y


def generate_dataset(x0_samples, x_dot0_samples, **kwargs):
    X = []
    Y = []
    for x0, x_dot0 in tqdm(zip(x0_samples, x_dot0_samples)):
        t_sim, states_sim = sim_mass_spring_damper(x0=x0, x_dot0=x_dot0, **kwargs)
        X.append([x0, x_dot0])
        Y.append(states_sim[0])  # Position at each time step
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def transfer_scenario_name(transfer_params):
    """Generates a scenario name based on the transfer parameters."""
    return "".join([f"{k}({v})" for k, v in transfer_params.items()])


def data_dim_name(data_dim):
    """Generates a data dimension name based on the data dimension."""
    return f"dataDim({data_dim})"


def get_data_iteration_name(data_iteration):
    """Generates a data iteration name based on the data iteration."""
    return f"dataIter({data_iteration})"


def get_data_indices(data_dim, data_iteration):
    train_begin = 10_000 * data_iteration
    trainVal_end = train_begin + data_dim
    test_begin = train_begin + 1_000
    test_end = test_begin + 9_000
    return np.concatenate(
        [
            np.arange(train_begin, trainVal_end),
            np.arange(test_begin, test_end),
        ],
        dtype=int,
    )


def get_transfer_data(scenario, indices=None):
    """Retrieve transfer data for a specific scenario and optional indices."""
    transfer_data = np.load(TRANSFER_DATA_PATH)
    scenario_name = transfer_scenario_name(scenario)
    if scenario_name not in transfer_data:
        raise ValueError(f"Scenario {scenario_name} not found in transfer data.")
    X = transfer_data["X"]
    Y = transfer_data[scenario_name]
    if indices is not None:
        X = X[indices]
        Y = Y[indices]
    return X, Y


def train_val_test_split(
    x, y, n_train, n_validation, n_test, begin_index=0, tensor_convert=True
):
    train_begin_index = begin_index
    train_end_index = train_begin_index + n_train
    val_begin_index = train_end_index
    val_end_index = val_begin_index + n_validation
    test_begin_index = val_end_index
    test_end_index = test_begin_index + n_test
    if not tensor_convert:
        x_train = x[train_begin_index:train_end_index]
        y_train = y[train_begin_index:train_end_index]
        x_val = x[val_begin_index:val_end_index]
        y_val = y[val_begin_index:val_end_index]
        x_test = x[test_begin_index:test_end_index]
        y_test = y[test_begin_index:test_end_index]
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        x_train = torch.tensor(
            x[train_begin_index:train_end_index], dtype=torch.float32
        )
        y_train = torch.tensor(
            y[train_begin_index:train_end_index], dtype=torch.float32
        )
        x_val = torch.tensor(x[val_begin_index:val_end_index], dtype=torch.float32)
        y_val = torch.tensor(y[val_begin_index:val_end_index], dtype=torch.float32)
        x_test = torch.tensor(x[test_begin_index:test_end_index], dtype=torch.float32)
        y_test = torch.tensor(y[test_begin_index:test_end_index], dtype=torch.float32)
        return x_train, y_train, x_val, y_val, x_test, y_test


def zip_dir(dir, zip_filename):
    dir = Path(dir)
    for file in dir.rglob("*"):
        if file.is_file():
            with zipfile.ZipFile(zip_filename, "a") as zipf:
                zipf.write(file, file.relative_to(dir))
    return
