import numpy as np

# Sensory basis function
def kappa(qx, qy):
    sigma = 0.18  # Gaussian sd
    mu = np.array([1, 1, 1, 3, 3, 3, 5, 5, 5, 
                    1, 3, 5, 1, 3, 5, 1, 3, 5]) / 6  # Gaussian means
    return np.array([1 / (sigma**2 * 2 * np.pi) * 
        np.exp(-((qx - mu[i])**2 + (qy - mu[j])**2) / (2 * sigma**2)) 
        for i, j in zip(range(9), range(9, 18))])

def reshape_state(z, n):
    sz = [n, n, 9*n, 9*n, 9*9*n]
    en = 0

    st = en; en = sum(sz[:1])
    px_rs = z[st:en, 0]

    st = en; en = sum(sz[:2])
    py_rs = z[st:en, 0]

    st = en; en = sum(sz[:3])
    ai_rs = np.reshape(z[st:en, 0], (9, n))

    st = en; en = sum(sz[:4])
    li_rs = np.reshape(z[st:en, 0], (9, n))

    st = en; en = sum(sz[:5])
    Li_rs = np.reshape(z[st:en, 0], (9, 9, n))

    return px_rs, py_rs, ai_rs, li_rs, Li_rs
