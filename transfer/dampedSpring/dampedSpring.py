""""""
##########https://update.code.visualstudio.com/1.101.1/cli-linux-x64/stable
# TODO ###
##########
# checkpoint last model -> trainable class, also set frequency to just last or infrequently
# get the model to save the weights of the best, delete others
# same summary stats

#############################
########## Imports ##########
#############################
import os
import shutil

from dtaidistance import dtw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import CombinedStopper, ExperimentPlateauStopper, TrialPlateauStopper
from scipy.integrate import solve_ivp
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.autonotebook import tqdm

from sekf.modeling import init_weights
from sekf.optimizers import maskedAdam, SEKF

########## Config ##########

# Randomly sample initial conditions for training and testing
N_TRAIN = 100_000
N_VALIDATION = 10_000
N_TEST = 10_000
N_TRANSFER = 1000
N_TRANSFER_VALIDATION = 1000
N_TRANSFER_TOTAL = 100_000

MEASUREMENT_NOISE_PM = 0.1

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "model_weights.pth"
OPTIMIZER_STATE_SEKF = "sekf_optimizer_state.npz"
DATA_DIR = os.path.join(CASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
MODEL0_WEIGHTS_PATH = os.path.join(DATA_DIR, MODEL_FILENAME)
TRAINING_DATA_FILENAME = "training_data.npz"
TRAINING_METRICS_FILENAME = "training_metrics.csv"
TRANSFER_DIR = os.path.join(DATA_DIR, "transfer")
os.makedirs(TRANSFER_DIR, exist_ok=True)
RESULTS_DIRS = {
    "training": DATA_DIR,
    "transfer_finetuning": os.path.join(DATA_DIR, "transfer_finetuning"),
    "transfer_finetuning_SEKF": os.path.join(DATA_DIR, "transfer_finetuning_SEKF"),
    "transfer_retraining": os.path.join(DATA_DIR, "transfer_retraining"),
}
ALL_TRIALS_BASE_FILENAME = "allTrials_metrics.csv"
BEST_RESULT_BASE_FILENAME = "bestResult_metrics.csv"


SCENARIOS = [
    {"m":1.1},
    {"m":0.9},
    {"c":0.55},
    {"c":0.45},
    {"k":1.1},
    {"k":0.9},
    {"u":1.0},
    {"u":-1.0},
    ]

# configs to change

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
    "mask_fn_quantile_thresh": 0.0,
    "optimizer": "adam",
    "sekf_q": 0.1,
    "sekf_p0": 100.0,
    "sekf_save_path": None,  
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
        self.config = default_config | config
        self._init_model(self.config)
        self._init_optimizer(self.config)
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=self.config.get("lr_patience"), factor=self.config.get("lr_factor"))
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data["train_x"], data["train_y"]),
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        self.initial_weights = self.optimizer._get_flat_params().detach().numpy()
        self.data = data
        
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
        match config.get("optimizer"):
            case "adam":
                self.optimizer = maskedAdam(
                    self.model.parameters(),
                    lr=config.get("lr"),
                    mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 1.0),
                )
                
            case "sekf":
                self.optimizer = SEKF(
                    self.model.parameters(),
                    lr=config.get("lr"),
                    q=config.get("sekf_q", 0.1),
                    p0=config.get("sekf_p0", 100),
                    mask_fn_quantile_thresh=config.get("mask_fn_quantile_thresh", 1.0),
                    save_path=config.get("sekf_save_path", None),
                    
                )
        
    def _maskedAdam_step(self, x_batch, y_batch):
        """Performs a single step of the masked Adam optimizer."""
        self.optimizer.zero_grad()
        y_pred = self.model(x_batch)
        loss = self.loss_fn(y_pred, y_batch)
        loss.backward()
        self.optimizer.masked_step()
        
    def _sekf_step(self, x_batch, y_batch):
        """Performs a single step of the SEKF optimizer."""
        y_pred = self.model(x_batch)
        e = y_batch - y_pred
        
        if self.config.get("mask_fn_quantile_thresh",0.0) > 0.0:
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            grad_loss = self.optimizer._get_flat_grads()
            mask = self.optimizer.mask_fn(grad_loss)
        else:
            mask = None
        J = get_jacobian(self.model, (x_batch))
        self.optimizer.step(e, J, mask=mask)
        
        
        
    def _optimizer_step(self, x_batch, y_batch):
        match self.config.get("optimizer"):
            case "adam":
                self._maskedAdam_step(x_batch, y_batch)
            case "sekf":
                self._sekf_step(x_batch, y_batch)
        
    def eval(self, data):
        self.model.eval()
        with torch.no_grad():
            metrics = {
                "train_loss": self.loss_fn(self.model(data["train_x"]), data["train_y"]).item(),
                "val_loss": self.loss_fn(self.model(data["val_x"]), data["val_y"]).item(),
                "test_loss": self.loss_fn(self.model(data["test_x"]), data["test_y"]).item(),
                "cosine_similarity_weights": cosine_similarity(self.initial_weights, self.optimizer._get_flat_params().detach().numpy()),
                "lr": self.scheduler.get_last_lr()[0],
            }
        self.scheduler.step(metrics["val_loss"])
        return metrics
        

    def step(self):
        self.model.train()
        for x_batch, y_batch in self.train_dataloader:
            self._optimizer_step(x_batch, y_batch)

        metrics = self.eval(self.data)
        self.scheduler.step(metrics["val_loss"])
        return metrics

    def save_checkpoint(self, tmp_checkpoint_dir, model_fname=MODEL_FILENAME):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, model_fname)
        torch.save(self.model.state_dict(), checkpoint_path)
        # TODO: save optimizer state if needed
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir, model_fname=MODEL_FILENAME):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, model_fname)
        self.model.load_state_dict(torch.load(checkpoint_path))
        # TODO: Load optimizer state if needed
        return 
    
    def reset_config(self, new_config):
        """Reset the configuration of the trainer."""
        self.setup(new_config, self.data)
        return True

# Mass-Spring-Damper system simulation
def mass_spring_damper(t, state, m, c, k, u):
    x, x_dot = state
    x_ddot = (u - c*x_dot - k*x) / m
    return [x_dot, x_ddot]

def sim_mass_spring_damper(x0=0.0, x_dot0=0.0, m=1.0, c=0.5, k=1.0, u=0.0, t_end=20.0, t_eval=None):
    initial_state = [x0, x_dot0]  # Initial position and velocity
    params = (m, c, k, u)
    # Integrate the ODE
    res = solve_ivp(mass_spring_damper, [0,t_end], initial_state, args=params, t_eval=t_eval)
    return res.t, res.y

def generate_dataset(x0_samples, x_dot0_samples, **kwargs):
    X = []
    Y = []
    for x0, x_dot0 in tqdm(zip(x0_samples, x_dot0_samples)):
        t_sim, states_sim = sim_mass_spring_damper(x0=x0, x_dot0=x_dot0, **kwargs)
        X.append([x0, x_dot0])
        Y.append(states_sim[0])
    return np.array(X), np.array(Y)

## cosine similarity
# use einsum to compute Y_train @ Y_transfer.T / ||Y_train|| * ||Y_transfer||
def cosine_similarity(a, b):
    """computes row-wise cosine similarity."""
    assert a.ndim == b.ndim
    if a.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if a.ndim == 2:
        return np.einsum('ij,ij->i', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

def cosine_distance(a, b):
    """computes row-wise cosine distance."""
    return (1 - cosine_similarity(a, b)) / 2

def transfer_scenario_name(transfer_params):
    """Generates a scenario name based on the transfer parameters."""
    return "".join([f"{k}({v})" for k, v in transfer_params.items()])

def get_transfer_data(transfer_params, data_dir=os.path.join(DATA_DIR, "transfer")):
    scenario_filename = transfer_scenario_name(transfer_params) + ".npz"
    if not os.path.exists(os.path.join(data_dir, scenario_filename)):
        return False
    data = np.load(os.path.join(data_dir, scenario_filename))
    data_x = np.load(os.path.join(data_dir, "X.npz"))["X"]
    data_y = data["Y"]
    return data_x, data_y

def generate_transfer_data(transfer_params, n_samples=1000, data_dir=os.path.join(DATA_DIR, "transfer")):
    """Generates transfer data for the given parameters."""
    if not os.path.exists(os.path.join(data_dir, "X.npy")):
        rng = np.random.default_rng(42)
        X_transfer = rng.uniform(-5, 5, (n_samples, 2), dtype=np.float32)
        np.savez(os.path.join(data_dir, "X"), X=X_transfer)
    else:
        X_transfer = np.load(os.path.join(data_dir, "X.npy"))["X"]
    
    _, Y_transfer = generate_dataset(X_transfer[:, 0], X_transfer[:, 1], **transfer_params)
    # ensure float32 type
    Y_transfer = np.array(Y_transfer, dtype=np.float32)
    
    scenario_filename = transfer_scenario_name(transfer_params) + ".npz"
    np.savez(os.path.join(data_dir, scenario_filename), Y=Y_transfer)
    return X_transfer, Y_transfer

def train_val_test_split(x,y, n_train=None, n_validation=None, n_test=None, p_train=None, p_validation=None, p_test=None, tensor_convert=False):
    """Splits the data into train, validation and test sets."""
    assert len(x) == len(y), "x and y must have the same length"
    # give either n_train, n_validation or n_test, or p_train, p_validation, p_test
    if p_train is not None:
        assert p_train + p_validation + p_test == 1, "p_train, p_validation and p_test must sum to 1"
        assert all((n_train is None, n_validation is None, n_test is None)), "Either use p_train, p_validation, p_test or n_train, n_validation, n_test"
        n_train = int(p_train * len(x))
        n_validation = int(p_validation * len(x))
        n_test = len(x) - n_train - n_validation
    if n_train is None:
        n_train = int(0.8 * len(x))
    if n_validation is None:
        n_validation = int(0.1 * len(x))
    if n_test is None:
        n_test = len(x) - n_train - n_validation

    x_train = x[:n_train]
    y_train = y[:n_train]
    x_validation = x[n_train:n_train+n_validation]
    y_validation = y[n_train:n_train+n_validation]
    x_test = x[n_train+n_validation:n_train+n_validation+n_test]
    y_test = y[n_train+n_validation:n_train+n_validation+n_test]
    if tensor_convert:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        x_validation = torch.tensor(x_validation, dtype=torch.float32)
        y_validation = torch.tensor(y_validation, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    return x_train, y_train, x_validation, y_validation, x_test, y_test

