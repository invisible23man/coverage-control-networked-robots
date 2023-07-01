import numpy as np
import reshape_state
import compute_centroid

# Initialize global variables
n, h, K, Fi, amin, Tau, g, psi, a, kappa = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # Replace with actual values or definitions
est_pos_err = []
tru_pos_err = []

# Define the function cvtODE
def cvtODE(t, z):
    global n, h, K, Fi, amin, Tau, g, psi, a, kappa
    global est_pos_err, tru_pos_err

    # ------------------------------------------------- #
    # ---- The Adaptive Coverage Control Algorithm ---- #
    # ------------------------------------------------- #
    px, py, ai, li, Li = reshape_state(z, n) 
    dai = np.zeros_like(ai)
    dli = np.zeros_like(li)
    dLi = np.zeros_like(Li)

    # Voronoi regions & Centroid calcuation
    Cv, Cv_true, L = compute_centroid(px, py, ai)  

    # Control law: ui = -K*(Cvi - pi)
    dx = K[0, 0] * (Cv[0, :].T - px)
    dy = K[1, 1] * (Cv[1, :].T - py)
    est_pos_err.append(np.mean(np.linalg.norm([(Cv[0, :].T - px), (Cv[1, :].T - py)], axis=0)))
    tru_pos_err.append(np.mean(np.linalg.norm([(Cv_true[0, :].T - px), (Cv_true[1, :].T - py)], axis=0)))

    # Adaption laws for paramter estimate
    s = np.dot(ai, L)

    for i in range(n):
        dai_pre = -(np.dot(Fi[:, :, i], ai[:, i])) - g * (np.dot(Li[:, :, i], ai[:, i]) - li[:, i]) - psi * s[:, i]
        Iproji = np.zeros(9)
        Iproji[ai[:, i] + dai_pre * h <= amin] = 1
        #Iproji[ai[:,i] > amin] = 0
        #Iproji[ai[:,i] == amin & dai_pre >= 0] = 0
        dai[:, i] = Tau * (dai_pre - np.diag(Iproji) @ dai_pre)

    # Update li and Li
    # w_t = np.exp(-t) 
    for i in range(n):
        w_t = np.linalg.norm([dx[i], dy[i]]) / np.linalg.norm(K)  # Data weighting function
        ki = kappa(px[i], py[i])  # You need to define the kappa function
        phi_t = ki * a
        dLi[:, :, i] = w_t * (np.matmul(ki.T,ki))
        dli[:, i] = w_t * phi_t * ki.T

    # State update
    dz = np.concatenate([dx, dy, dai.ravel(), dli.ravel(), dLi.ravel()], axis=None)

    # Debugging ---------------------------- #
    print(t)
    print(np.mean(np.linalg.norm(a - ai)))  # parameter error
    # print(np.linalg.norm([dx,dy])) # distance to centroid

    return dz
