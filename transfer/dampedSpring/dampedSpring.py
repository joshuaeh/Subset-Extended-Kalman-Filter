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

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL0_FILENAME = "modelv0.pth"
MODEL0_PATH = os.path.join(CASE_DIR, MODEL0_FILENAME)
MODEL0_WEIGHTS_FILENAME = "modelv0_weights.pth"
MODEL0_WEIGHTS_PATH = os.path.join(CASE_DIR, MODEL0_WEIGHTS_FILENAME)
MODEL_FILENAME = "model.pth"
DATA_DIR = os.path.join(CASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
TRAINING_DATA_FILENAME = "training_data.npz"
TRAINING_METRICS_FILENAME = "training_metrics.csv"
TRANSFER_DIR = os.path.join(DATA_DIR, "transfer")
os.makedirs(TRANSFER_DIR, exist_ok=True)

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

class DampedSpringTrainer(tune.Trainable):
    def setup(self, config):
        self.config = config
        
        self.train_x = config["train_x"]
        self.train_y = config["train_y"]
        self.val_x = config["val_x"]
        self.val_y = config["val_y"]
        self.test_x = config["test_x"]
        self.test_y = config["test_y"]
        
        self._initialize_model(config)
        self.optimizer = maskedAdam(self.model.parameters(), lr=config["lr"], mask_fn_quantile_thresh=config["mask_fn_quantile_thresh"])
        self.loss_fn = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=20, factor=0.5)
        self.train_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.train_x, self.train_y),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        self.initial_weights = self.optimizer._get_flat_params().detach().numpy()
        
    def _initialize_model(self, config):
        self.model = MLP()
        # load pre-trained weights
        weights_dict = torch.load(MODEL0_WEIGHTS_PATH, weights_only=True)
        self.model.load_state_dict(weights_dict)
        

    def step(self, train_x, train_y, val_x, val_y, test_x, test_y):
        self.model.train()
        for x_batch, y_batch in self.train_dataloader:
            self.optimizer.zero_grad()
            y_pred = self.model(x_batch)
            loss = self.loss_fn(y_pred, y_batch)
            loss.backward()
            self.optimizer.masked_step()

        with torch.no_grad():
            epoch_loss = self.loss_fn(self.model(train_x), train_y).item()
            validation_loss = self.loss_fn(self.model(val_x), val_y).item()
            test_loss = self.loss_fn(self.model(test_x), test_y).item()
        self.scheduler.step(epoch_loss)
        # Log the training loss
        metrics = {
            "train_loss": epoch_loss,
            "val_loss": validation_loss,
            "test_loss": test_loss,
            "cosine_distance_weights": cosine_distance(self.initial_weights.reshape(1, -1), self.optimizer._get_flat_params().detach().numpy().reshape(1, -1)),
        }
        # tune.report(metrics)
        # checkpoint
        return metrics

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir
    
    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model.load_state_dict(torch.load(checkpoint_path))
        return 


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
        return np.einsum('ij,ik->i', a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))

def cosine_distance(a, b):
    """computes row-wise cosine distance."""
    return (1 - cosine_similarity(a, b)) / 2

def get_transfer_data(k, v, data_dir=os.path.join(DATA_DIR, "transfer")):
    scenario_filename = f"{k}({v}).npz"
    if not os.path.exists(os.path.join(data_dir, scenario_filename)):
        return False
    data = np.load(os.path.join(data_dir, scenario_filename))
    data_x = np.load(os.path.join(data_dir, "X.npz"))["X"]
    data_y = data["Y"]
    return data_x, data_y

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

