import numpy as np

# Global Variables

n = 20  # Number of robots
K = 3 * np.eye(2)  # Control gain matrix
Tau = np.eye(9)
g = 130  # Learning rate <100
psi = 0  # Consensus weight <20
h = 0.01  # ode step size
sens_info_flag = 0  # Set 1 to run the case when sensory info is known

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
