#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Update model according to last plant measurement"""

import json
import logging
import os
 
import numpy as np
import pandas as pd
import torch

from sekf.modeling import AbstractNN
from sekf.optimizers import SEKF

from secondOrderODE import NN

# read config from JSON file
with open("config.json") as f:
    config = json.load(f)
NOISE_STD = config["NOISE_STD"]
UPDATING_MODEL_SAVE_PATH = config["UPDATING_MODEL_SAVE_PATH"]
UPDATING_PLANT_DATA_SAVE_PATH = config["UPDATING_PLANT_DATA_SAVE_PATH"]
UPDATING_MODEL_DATA_SAVE_PATH = config["UPDATING_MODEL_DATA_SAVE_PATH"]
UPDATING_OPTIMIZER_SAVE_PATH = config["UPDATING_OPTIMIZER_SAVE_PATH"]
UPDATING_PARAM_SELECTION = config["UPDATING_PARAM_SELECTION"]

def main():
    # get data (read last row of csv file)
    # with open(UPDATING_PLANT_DATA_SAVE_PATH, "r") as f:
    #     lines = f.readlines()
    #     n_lines = len(lines)
    # df = pd.read_csv(UPDATING_PLANT_DATA_SAVE_PATH, header=0, skiprows=range(2, n_lines-1))
    plant_data_df = pd.read_csv(UPDATING_PLANT_DATA_SAVE_PATH, header=0)
    x = plant_data_df["x"].values[-1]
    y = plant_data_df["y_measured"].values[-1]

    # initialize model, optimizer, loss function
    # model
    if os.path.exists(UPDATING_MODEL_SAVE_PATH):
        model = torch.load(UPDATING_MODEL_SAVE_PATH, weights_only=False)
    else:
        model = torch.load(config["TRAINING_MODEL_SAVE_PATH"], weights_only=False)
    # optimizer
    param_selection_method = UPDATING_PARAM_SELECTION["method"]
    assert param_selection_method.lower() in ["proportion", "magnitude", None], f"Invalid method: {param_selection_method}. Choose 'proportion', 'magnitude', or 'none'."
    if param_selection_method.lower() == "proportion":
        mask_func_kwargs = {"mask_fn_quantile_thresh": UPDATING_PARAM_SELECTION["param"]}
    elif param_selection_method.lower() == "magnitude":
        mask_func_kwargs = {"mask_fn_thresh": UPDATING_PARAM_SELECTION["param"]}
    else:
        mask_func_kwargs = {"mask_fn_thresh": None, "mask_fn_quantile_thresh": None}
    opt = SEKF(model.parameters(),
        lr=1/NOISE_STD**2,
        save_path=UPDATING_OPTIMIZER_SAVE_PATH,
        **mask_func_kwargs
    )
    # loss function
    loss_fn = torch.nn.MSELoss()

    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(1, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(1, 1)
    # update step
    yp,_ = opt.easy_step(
        model, (x_tensor,), y_tensor, loss_fn
    )

    # save
    torch.save(model, UPDATING_MODEL_SAVE_PATH)
    opt.save_params()
    if not os.path.exists(UPDATING_MODEL_DATA_SAVE_PATH):
        with open(UPDATING_MODEL_DATA_SAVE_PATH, "w") as f:
            f.write("x,y_measured,y_pred\n")
    with open(UPDATING_MODEL_DATA_SAVE_PATH, "a") as f:
        f.write(f"{x},{y},{yp.detach().item()}\n")
    return
if __name__ == "__main__":
    main()