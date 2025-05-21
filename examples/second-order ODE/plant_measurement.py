#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Make the plant measurement"""

import json
import os

import numpy as np
import pandas as pd
import torch

from secondOrderODE import analytical_solution, NN, rng, random_walk_step

# read config from JSON file
with open("config.json") as f:
    config = json.load(f)
RANDOM_SEED = config["RANDOM_SEED"]
NOISE_STD = config["NOISE_STD"]
TRAIN_EPSILON = config["TRAINING_EPSILON"]
FINAL_EPSILON = config["UPDATING_FINAL_EPSILON"]
UPDATING_N_MEASUREMENTS = config["UPDATING_N_MEASUREMENTS"]
UPDATING_PLANT_DATA_SAVE_PATH = config["UPDATING_PLANT_DATA_SAVE_PATH"]

def main():
    # init csv if not exist
    if not os.path.exists(UPDATING_PLANT_DATA_SAVE_PATH):
        with open(UPDATING_PLANT_DATA_SAVE_PATH, "w") as f:
            f.write("x,y,y_measured\n")
        N_MEASUREMENTS = 0
        x0 = 0
    else:
        df = pd.read_csv(UPDATING_PLANT_DATA_SAVE_PATH)
        N_MEASUREMENTS = df.shape[0]
        x0 = df["x"].values[-1]

    epsilon = TRAIN_EPSILON + (FINAL_EPSILON - TRAIN_EPSILON) * N_MEASUREMENTS / UPDATING_N_MEASUREMENTS
    x1 = random_walk_step(x0, step_size=0.1, min_value=0, max_value=1)
    y1 = analytical_solution(x1, epsilon)
    y1_measured = y1 + rng.normal(0, NOISE_STD)

    # save measurement
    with open(UPDATING_PLANT_DATA_SAVE_PATH, "a") as f:
        f.write(f"{x1},{y1},{y1_measured}\n")
    return

if __name__ == "__main__":
    main()