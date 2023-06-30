# MAE 598 Multi robot systems
# Project - Ravi Pipaliya
# Distributed Adaptive coverage control
# 11/22/2020
# ----------------------------------------------------------------------- %

# Python Version - Gowrisankar Chandrasekaran 
# 30/06/2023

#%%
import numpy as np
from scipy.integrate import solve_ivp

np.random.seed(1000)

global n, K, Tau, g, psi, h
global amin, Fi, kappa, a, sens,_info_flag
global est_pos_err, tru_pos_err 
#%% Initialize ---- %
n = 20  # no. of robots
K = 3*np.eye(2) # Control gain matrix
Tau = np.eye(9) 
g = 130 # Learning rate <100
psi = 0 # Consensus weight <20
h = 0.01 # ode step size
sens_info_flag = 0 # Set 1 to run the case when sensory info is known

## Initial Positions ----#
x0 = np.random.rand(n,1)  # Initial x
y0 = np.random.rand(n,1)  # Initial y

## Model parameters ----#
amin = 0.1 # minimum weight
a = np.array([100, amin*np.ones((1,7)), 100]).T # True weights
ai = amin * np.ones((9,n)) # parameter estimate with each robot
li = np.zeros((9,n)) 
Li = np.zeros((9,9,n))
Fi = np.zeros((9,9,n))



# %%Sensory basis function ----%
sigma = 0.18 # Gaussian sd
mu = np.array([[1, 1, 1, 3, 3, 3, 5, 5, 5], 
            [1, 3, 5, 1, 3, 5, 1, 3, 5]])/6; # Gaussian means

def get_kappa(qx,qy):

    kappa = \
        [1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,0],2) + np.power(qy-mu[1,0],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,1],2) + np.power(qy-mu[1,1],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,2],2) + np.power(qy-mu[1,2],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,3],2) + np.power(qy-mu[1,3],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,4],2) + np.power(qy-mu[1,4],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,5],2) + np.power(qy-mu[1,5],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,6],2) + np.power(qy-mu[1,6],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,7],2) + np.power(qy-mu[1,7],2)) /(2*np.power(sigma,2))),
         1/(np.power(sigma,2)*(2*np.pi))*np.exp(-(np.power(qx-mu[0,8],2) + np.power(qy-mu[1,8],2)) /(2*np.power(sigma,2)))]

    return kappa

get_kappa(1,3)

#%% Simulation ----%
tspan = list(range(0,h,30))
est_pos_err = np.zeros(len(tspan),1)
tru_pos_err = np.zeros(len(tspan),1)

z0 = np.array([x0, y0, ai[:], li[:], Li[:]])  # Initial state
# z = ode1(@cvtODE,tspan,z0); # Fixed time step ode
