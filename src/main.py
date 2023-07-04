from scipy.integrate import solve_ivp
from shapely.geometry import Polygon
from pathlib import Path
from PIL import Image
from globals import *
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
from tools import reshape_state
from cvtODE import cvtODE

"""
MAE 598 Multi robot systems
Project - MATLAB Version
Distributed Adaptive coverage control
By: Ravi Pipaliya

Collaborative Multi UAV Aided Smart Farming
Project - Python Version
Distributed Adaptive coverage control
By: Gowrisankar Chandraskeharan

"""

# Initial Positions
x0 = np.random.rand(n, 1)  # Initial x
y0 = np.random.rand(n, 1)  # Initial y

# Initialize state
z0 = np.concatenate([x0.flatten(), y0.flatten(), ai.flatten(), li.flatten(), Li.flatten()])  # Initial state

# Define the ODE solver
sol = solve_ivp(cvtODE, [tspan[0], tspan[-1]], z0, method='RK45', t_eval=tspan)

# TODO: Save/Load Workspace

# Decompose state
par_err = np.zeros(sol.y.shape[1])
pxi = np.zeros((n, sol.y.shape[1]))
pyi = np.zeros((n, sol.y.shape[1]))
for i in range(sol.y.shape[1]):
    # TODO: Define the reshape_state function that matches your problem. Replace `reshape_state` with your function
    pxi[:, i], pyi[:, i], ain = reshape_state(sol.y[:, i])
    par_err[i] = np.mean(np.linalg.norm(a - ain, axis=0))

# TODO: Plots and GIFs

# TODO: Comparing Position error (Basic non consensus vs consensus controller)
