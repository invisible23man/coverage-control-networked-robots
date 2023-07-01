import numpy as np
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import os
from tools import reshape_state
import cvtODE

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


# Global Variables
n = 20  # Number of robots
K = 3 * np.eye(2)  # Control gain matrix
Tau = np.eye(9)
g = 130  # Learning rate <100
psi = 0  # Consensus weight <20
h = 0.01  # ode step size
sens_info_flag = 0  # Set 1 to run the case when sensory info is known

# Initial Positions
x0 = np.random.rand(n, 1)  # Initial x
y0 = np.random.rand(n, 1)  # Initial y

# Model parameters
amin = 0.1  # minimum weight
a = np.array([100] + [amin]*7 + [100]).reshape(-1, 1)  # True weights
ai = amin * np.ones((9, n))  # parameter estimate with each robot
li = np.zeros((9, n))
Li = np.zeros((9, 9, n))
Fi = np.zeros((9, 9, n))

# Simulation
tspan = np.arange(0, 30 + h, h)
est_pos_err = np.zeros(len(tspan))
tru_pos_err = np.zeros(len(tspan))

# Initialize state
z0 = np.concatenate([x0, y0, ai.flatten(), li.flatten(), Li.flatten()])  # Initial state

# Define the ODE solver
# Note: You need to define the cvtODE function that matches your problem. Replace `cvtODE` with your function
sol = solve_ivp(cvtODE, (tspan[0], tspan[-1]), z0, method='RK45', t_eval=tspan)

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
