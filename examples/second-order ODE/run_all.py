#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all steps of the second-order ODE example"""
import json

import train_model
import plant_measurement
import update_model

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)
    train_model.main()
    for i in range(config["UPDATING_N_MEASUREMENTS"]):
        print(f"Measurement {i+1:0>6,}/{config['UPDATING_N_MEASUREMENTS']:0>7,}", end="\r")
        plant_measurement.main()
        update_model.main()