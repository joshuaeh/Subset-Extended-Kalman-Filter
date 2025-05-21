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


    
    