# usr/bin/env python
# -*- coding: utf-8 -*-
"""train nn on second-order ODE"""
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from sekf.modeling import AbstractNN, init_weights

from secondOrderODE import F, NN

# read configuration from JSON file
with open("config.json") as f:
    config = json.load(f)
SEED = config.get("RANDOM_SEED", 42)
NOISE_STD = config.get("NOISE_STD", 0.1)
EPSILON = config.get("TRAIN_EPSILON", 0.1)
TRAINING_SAMPLES = config.get("TRAINING_SAMPLES", 1000)
TESTING_SAMPLES = config.get("TESTING_SAMPLES", 200)
TRAINING_EPOCHS = config.get("TRAINING_EPOCHS", 1_000)
MODEL_SAVE_PATH = config.get("MODEL_SAVE_PATH", "model.pth")
TRAINING_DATA_SAVE_PATH = config.get("TRAINING_DATA_SAVE_PATH", "training_data.npz")
PLOT_PATH = config.get("PLOT_PATH", None)

# set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# create datasets
x_train_ = torch.rand(TRAINING_SAMPLES, 1)
x_test_ = torch.rand(TESTING_SAMPLES, 1)
x_sweep = torch.linspace(0, 1, 100).reshape(-1, 1)
y_train_ = F(x_train_, EPSILON) + torch.normal(0, NOISE_STD, (TRAINING_SAMPLES, 1))
y_test_ = F(x_test_, EPSILON) + torch.normal(0, NOISE_STD, (TESTING_SAMPLES, 1))

model = NN()
model.apply(init_weights)
    
opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

# train model
training_losses = []
testing_losses = []
for epoch in range(TRAINING_EPOCHS):
    model.train()
    opt.zero_grad()
    y_pred = model(x_train_)
    loss = loss_fn(y_pred, y_train_)
    loss.backward()
    opt.step()
    training_losses.append(loss.item())
    with torch.no_grad():
        y_test_pred = model(x_test_)
        test_loss = loss_fn(y_test_pred, y_test_)
        testing_losses.append(test_loss.item())
    if epoch % 100 == 0:
        model.eval()
        print(f"Epoch {epoch}, Test Loss: {test_loss.item()}", end="\r")

# save model
torch.save(model, MODEL_SAVE_PATH)
np.savez(TRAINING_DATA_SAVE_PATH, dict(
    train_x=x_train_.numpy(),
    train_y=y_train_.numpy(),
    test_x=x_test_.numpy(),
    test_y=y_test_.numpy(),
    train_loss=np.array(training_losses),
    test_loss=np.array(testing_losses)
))

if PLOT_PATH:
    # plot training and testing losses
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=1_000)
    ax[0].plot(np.arange(TRAINING_EPOCHS), training_losses, label="Training Loss")
    ax[0].plot(np.arange(TRAINING_EPOCHS), testing_losses, label="Testing Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend(loc="best")
    ax[1].plot(x_sweep.numpy(), F(x_sweep, EPSILON).numpy(), "k-", label="Analytical Solution")
    ax[1].scatter(x_train_.numpy(), y_train_.numpy(), 1, marker=".", color="b", label="Training Data")
    ax[1].scatter(x_test_.numpy(), y_test_.numpy(), 1, marker=".", color="r", label="Testing Data")
    ax[1].plot(x_sweep.numpy(), model(x_sweep).detach().numpy(), "g:", linewidth=3, label="Model Prediction")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend(loc="best")
    fig.savefig(PLOT_PATH, dpi=1_000)
    # plt.show()
    plt.close(fig)