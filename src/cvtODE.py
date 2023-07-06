import numpy as np
from tools import reshape_state, kappa
from compute_centroid import compute_centroid

# Define the function cvtODE
def cvtODE(t, z, args):

    (n, K,Tau, g, psi, h, sens_info_flag, amin, a, ai, li, Li, Fi, tspan, est_pos_err, tru_pos_err) = args
    
    # ------------------------------------------------- #
    # ---- The Adaptive Coverage Control Algorithm ---- #
    # ------------------------------------------------- #
    px, py, ai, li, Li = reshape_state(z) 
    dai = np.zeros_like(ai)
    dli = np.zeros_like(li)
    dLi = np.zeros_like(Li)

    # Voronoi regions & Centroid calcuation
    Cv, Cv_true, L, sens_info_flag, a, Fi, K = compute_centroid(px, py, ai, n, sens_info_flag, a, Fi, K)  

    # Control law: ui = -K*(Cvi - pi)
    dx = K[0, 0] * (Cv[0, :].T - px)
    dy = K[1, 1] * (Cv[1, :].T - py)

    # est_pos_err(round(t/h+1)) = .... 
    est_pos_err = np.append(est_pos_err, np.mean(np.linalg.norm([(Cv[0, :].T - px), (Cv[1, :].T - py)], axis=0)))
    tru_pos_err = np.append(tru_pos_err, np.mean(np.linalg.norm([(Cv_true[0, :].T - px), (Cv_true[1, :].T - py)], axis=0)))

    # Adaption laws for paramter estimate
    s = ai @ L.T

    for i in range(n):
        dai_pre = -(Fi[:, :, i] @ ai[:, i]) - g * (Li[:, :, i] @ ai[:, i]) - li[:, i] - psi * s[:, i]
        Iproji = np.zeros(9)
        Iproji[ai[:, i] + dai_pre * h <= amin] = 1
        #Iproji[ai[:,i] > amin] = 0
        #Iproji[ai[:,i] == amin & dai_pre >= 0] = 0
        dai[:, i] = Tau @ (dai_pre - np.diag(Iproji) @ dai_pre).T

    # Update li and Li
    # w_t = np.exp(-t) 
    for i in range(n):
        w_t = np.linalg.norm([dx[i], dy[i]]) / np.linalg.norm(K)  # Data weighting function
        ki = kappa(px[i], py[i])  # You need to define the kappa function
        phi_t = ki @ a
        dLi[:, :, i] = w_t * (ki.T@ki)
        dli[:, i] = w_t * phi_t * ki.T

    # State update
    dz = np.concatenate([dx, dy, dai.ravel(), dli.ravel(), dLi.ravel()], axis=None)

    # Debugging ---------------------------- #
    print(t)
    print(np.mean(np.linalg.norm(a - ai)))  # parameter error
    # print(np.linalg.norm([dx,dy])) # distance to centroid

    return dz
