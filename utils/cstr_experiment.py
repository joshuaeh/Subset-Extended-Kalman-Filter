import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

from .__init__ import device, default_rng, model_colors, true_colors
from .modeling import AbstractNN
from .optimizers import GEKF, maskedAdam, maskedSGD

torch.device(device)

# constants
F = 0.6
V = 15.0
k1 = 2e-1
k2f = 5e-1
k2r = 1e-1

# functions
def dXdt(t, X, U):
    Ca, Cb, Cc = X
    Caf = U
    
    r1 = k1 * Ca
    r2 = k2f * Cb ** 3 - k2r * Cc
    dCa_dt = F / V * (Caf - Ca) - r1
    dCb_dt = -F / V * Cb + r1 - 3 * r2
    dCc_dt = -F / V * Cc + r2
    return [dCa_dt, dCb_dt, dCc_dt]

def dXdt_drifting(t, X, U):
    Ca, Cb, Cc = X
    Caf = U
    k2r = 1e-1 + 3e-5 * t
    
    r1 = k1 * Ca
    r2 = k2f * Cb ** 3 - k2r * Cc
    dCa_dt = F / V * (Caf - Ca) - r1
    dCb_dt = -F / V * Cb + r1 - 3 * r2
    dCc_dt = -F / V * Cc + r2
    return [dCa_dt, dCb_dt, dCc_dt]

def sim(func, t_span, X0, U, noise=None):
    X = np.zeros((len(t_span), len(X0)))
    X[0,:] = X0
    for i in range(1, len(t_span)):
        X[i] = solve_ivp(func, (t_span[i - 1], t_span[i]), X[i - 1], args=(U[i],), t_eval=[t_span[i]]).y.T[0]
    if noise is not None:
        assert noise.shape == X0.shape, "Noise shape must match state shape, got {} and {}".format(noise.shape, X0.shape)
        Y = X + default_rng.normal(np.zeros_like(X0), noise, size=X.shape)
    else:
        Y = X
    return {
        "X": X,
        "Y": Y,
        "U": U,
    }

def get_U(t_span, umin, umax, len_step):
    U = np.zeros(len(t_span))
    for i in range(0, len(t_span), len_step):
        U[i:i + len_step] = default_rng.uniform(umin, umax)
    return U

def format_dataset(
    data,
    Xscaler,
    Uscaler,
    train=True,
    begin_index=0,
    end_index=-1,
    nx=-1,
    input_horizon=2,
    output_horizon=60,
    name=None,
    device=None,
):
    # take advantage of all data, don't throw boundary between train/validate/test away
    # TODO: finish implementation
    _end_index = end_index if end_index == -1 else min(end_index + output_horizon, data["Y"].shape[0])
    # U needs to be 2d
    if data["U"].ndim == 1:
        data["U"] = data["U"].reshape(-1, 1)
    # scaling
    if train:
        X = torch.tensor(Xscaler.fit_transform(data["Y"][begin_index:end_index, :nx]), dtype=torch.float32)
        U = torch.tensor(Uscaler.fit_transform(data["U"][begin_index:end_index, :]), dtype=torch.float32)
    else:
        X = torch.tensor(Xscaler.transform(data["Y"][begin_index:end_index, :nx]), dtype=torch.float32)
        U = torch.tensor(Uscaler.transform(data["U"][begin_index:end_index, :]), dtype=torch.float32)
    # rolling horizon
    X = X.unfold(0, output_horizon + input_horizon, 1).permute(0, 2, 1)  # Nbatches, Nsteps, Nx
    U = U.unfold(0, output_horizon + input_horizon, 1).permute(0, 2, 1)[:, :, :]  # Nbatches, Nsteps, Nu
    # format into dictionary
    dataset = {
        "y": X[:, input_horizon:, :],
        "xn": X[:, :input_horizon, :],
        "u": U[:, :-input_horizon, :],
    }
    # send to device
    if device is not None:
        [v.to(device) for k, v in dataset.items()]
    return dataset

def rescale_data(data, model, Xscaler, Uscaler):
    xn = data["xn"]
    u = data["u"]
    y = data["y"]
    y_pred = model(xn, u)

    xn = xn.to("cpu")
    u = u.to("cpu")
    y = y.to("cpu")
    y_pred = y_pred.to("cpu")

    u = Uscaler.inverse_transform(u.squeeze().numpy())
    y = Xscaler.inverse_transform(y.numpy().reshape(-1, 3)).reshape(-1, 60, 3)
    y_pred = Xscaler.inverse_transform(y_pred.detach().numpy().reshape(-1, 3)).reshape(-1, 60, 3)
    t = np.arange(0, y_pred.shape[0], 1)
    return u, y, y_pred, t

def graph_concentration_predictions(u, y, y_pred, t, steps_ahead=0, dpi=300, ax=None, figsize=(13, 6)):
    n_steps_out = y.shape[1]
    species_labels = [r"$C_A$", r"$C_B$", r"$C_C$"]
    y_ticks_formatter = mpl.ticker.StrMethodFormatter("{x:,.2f}")

    if ax is None:
        fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True, dpi=dpi)
        ax_was_none = True
    for species in range(3):
        pred, = ax[species].plot(t + steps_ahead, y_pred[:, steps_ahead, species], color=model_colors[0], linestyle="-", label=f"Predicted", zorder=0)
        true1 = ax[species].scatter(t + steps_ahead, y[:, steps_ahead, species], color=true_colors[0], marker="o", s=5, label=f"True", zorder=1)

        ax[species].set_ylabel(species_labels[species])
        ax[species].yaxis.set_major_formatter(y_ticks_formatter)
    if ax_was_none:
        true2, = ax[-1].plot(t, u[:, 0], color=true_colors[0], linestyle="-", label="u")
        ax[-1].yaxis.set_major_formatter(y_ticks_formatter)
        # ax[-1].plot(t[-1] + np.arange(n_steps_out - 1), u[-1, 1:], "b-", label="u")
        ax[-1].set_ylabel(r"U ($C_{Af}$)")
        ax[-1].set_xlabel("Hours")
        ax[-1].set_xticks(np.arange(0, t[-1] + n_steps_out, 60), labels=np.arange(0, t[-1] + n_steps_out, 60) / 60)
        ax[-1].set_xticks(np.arange(0, t[-1] + n_steps_out, 15), minor=True)
        
        ax[-1].legend([pred, (true1, true2)], ["Predicted", "True"], handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
            loc="upper left", ncol=2, bbox_to_anchor=(0.65, -.13))

        # for a in range(3):
        #     pass
        for a in range(4):
            ax[a].set_xlim(0, t[-1])
            
    fig.tight_layout()
    return fig, ax

def gradient_descent_step(xn, u, y, model, optimizer, loss_fn, loss_threshold, gradient_threshold):
    optimizer.zero_grad()
    y_pred = model(xn, u)
    loss = loss_fn(y_pred, y)
    if loss.item() > loss_threshold:
        loss.backward()
        gradients = optimizer._get_flat_grads()
        mask = gradients.abs() > gradient_threshold
        optimizer.masked_step(mask)
    else:
        gradients = torch.zeros_like(optimizer._get_flat_params())
    return loss, gradients

