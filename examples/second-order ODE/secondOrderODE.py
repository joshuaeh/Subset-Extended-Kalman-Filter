# usr/bin/env python
# -*- coding: utf-8 -*-
"""functions for second-order ODE"""
import numpy as np
import torch

from sekf.modeling import AbstractNN

rng = np.random.default_rng(42)

class NN(AbstractNN):
    def __init__(self):
        super(AbstractNN, self).__init__()
        self.fc1 = torch.nn.Linear(1, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

##### function definitions #####
# Plant Equations
def analytical_solution(x, epsilon):
    """numpy function for analytical solutions in scalar/numpy form"""
    lambda1 = (-1 + np.sqrt(1 - 4 * epsilon)) / (2 * epsilon)
    lambda2 = (-1 - np.sqrt(1 - 4 * epsilon)) / (2 * epsilon)
    y = (np.exp(lambda1 * x) - np.exp(lambda2 * x)) / (np.exp(lambda1) - np.exp(lambda2))
    return y

def F(x, epsilon):
    """PyTorch function for analytical solutions in PyTorch form"""
    lambda1 = (-1 + (1 - 4 * epsilon)**0.5) / (2 * epsilon)
    lambda2 = (-1 - (1 - 4 * epsilon)**0.5) / (2 * epsilon)
    return (torch.exp(lambda1 * x) - torch.exp(lambda2 * x)) / (np.e**lambda1 - np.e**lambda2)

# input domain
def random_walk_step(x0, step_size=0.1, min_value=0, max_value=1):
    """Generate a random step for the random walk"""
    step = rng.uniform(-step_size, step_size)
    x_new = x0 + step
    # Ensure the new value stays within the specified bounds
    x_new = np.clip(x_new, min_value, max_value)
    return x_new

def random_walk(walk_length, min_value=0, max_value=1, step_size=0.1):
    """Generate a random walk"""
    walk = np.zeros(walk_length)
    for i in range(1, walk_length):
        walk[i] = random_walk_step(walk[i - 1], step_size, min_value, max_value)
    return walk

def plot_results(plot_path):
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    
    with open("config.json", "r") as f:
        config = json.load(f)
    df = pd.read_csv(config["UPDATING_MODEL_DATA_SAVE_PATH"], header=0)
    x = df["x"].values
    ym = df["y_measured"].values
    yp = df["y_pred"].values
    
    e = ym - yp
    ze = e / config["NOISE_STD"]
    t = np.arange(len(df))

    fig, ax = plt.subplots(2,2,figsize=(10, 6))

    ax[0,0].scatter(t, x, s=0.25, c=ym, cmap="viridis", alpha=0.25, )
    ax[0,0].set_xlabel("Measurement Number")
    ax[0,0].set_ylabel(r"$x$")
    ax[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax[0,0].set_title("Plant Measurements", y=0.96, va="top")
    cb00 = plt.colorbar(ax[0,0].collections[0], ax=ax[0,0], label="y")
    cb00.set_alpha(1)
    cb00.solids.set(alpha=1)

    ax[0,1].scatter(t, x, s=0.25, c=ze, cmap="RdBu_r", vmin=-5, vmax=5, alpha=0.25)
    ax[0,1].set_xlabel("Measurement Number")
    ax[0,1].set_ylabel(r"$x$")
    ax[0,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax[0,1].set_title("Measurement Z-score", y=0.96, va="top")
    cb01 = plt.colorbar(ax[0,1].collections[0], ax=ax[0,1], label=r"$z_e$")
    cb01.set_alpha(1)
    cb01.solids.set(alpha=1)

    rolling_error = np.convolve(ze, np.ones(50)/50, mode='valid')
    ax[1,0].plot(t[24:-25], rolling_error, c="k", alpha=0.5)
    ax[1,0].set_xlabel("Measurement Number")
    ax[1,0].set_ylabel("Rolling $z_e$ (50)")
    ax[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax[1,0].set_title("Rolling Z-score", y=0.96, va="top")

    rolling_abs_error = np.convolve(np.abs(ze), np.ones(50)/50, mode='valid')
    ax[1,1].plot(t[24:-25], rolling_abs_error, c="k", alpha=0.5)
    ax[1,1].set_xlabel("Measurement Number")
    ax[1,1].set_ylabel("Rolling $|z_e|$ (50)")
    ax[1,1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
    ax[1,1].set_title("Rolling Absolute Z-score", y=0.96, va="top")
    
    fig.savefig(plot_path, dpi=1_000)
    plt.close(fig)
    return
        