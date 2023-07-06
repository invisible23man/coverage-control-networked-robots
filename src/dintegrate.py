import numpy as np
from shapely.geometry import Point, Polygon
from tools import kappa


def dintegrate(vx, vy, pi_t, ai_t, pos,
                sens_info_flag, a, Fi, K):
    """
    Performs integration over the voronoi region.

    Parameters:
    vx, vy: x and y coordinates of the voronoi vertices
    pi_t: ith position
    ai_t: ith 'ai' parameter
    pos: position index

    Returns:
    Cvi: estimated centroid
    Cvi_true: true centroid
    """

    # Sample points
    cnt = 10000  # Sample count
    xp = np.random.rand(cnt, 1) * (np.max(vx) - np.min(vx)) + np.min(vx)
    yp = np.random.rand(cnt, 1) * (np.max(vy) - np.min(vy)) + np.min(vy)

    # create a polygon using vertices
    polygon = Polygon(zip(vx, vy))

    # check if points are inside the polygon
    in_region = [polygon.contains(Point(x, y)) for x, y in zip(xp, yp)]
    xp = xp[in_region]
    yp = yp[in_region]

    # Integrals over voronoi region
    kq = kappa(xp, yp)
    if sens_info_flag:
        # optimal configuration case when sensory info is known
        phi_est = kq @ a
    else:
        # Case when sensory info is unknown and estimate is used
        phi_est = kq @ ai_t

    mv = np.sum(phi_est)
    cx = np.sum(xp * phi_est) / mv
    cy = np.sum(yp * phi_est) / mv
    phi = kq @ a
    cx_true = np.sum(xp * phi) / np.sum(phi)
    cy_true = np.sum(yp * phi) / np.sum(phi)
    Cvi_true = np.array([cx_true, cy_true])

    # Check for negative mass
    if mv < 0:
        print(f'Negative mass calculated: {mv}')
        print(pos)
        print(ai_t.T)

    # Check for calculated centroid in voronoi region
    if not polygon.contains(Point(cx, cy)):
        print('Centroid outside the region')
        print(ai_t.T)
        Cvi = pi_t.T
    else:
        Cvi = np.array([cx, cy])

    # Update parameter Fi
    k1 = np.zeros((9, 2))
    for i in range(len(xp)):
        q_pi = np.array([xp[i], yp[i]]).T[0] - pi_t
        k1 += kq[i, :].reshape(-1, 1) * q_pi

    Fi[:, :, pos] = (1 / mv) * (k1 @ K @ k1.T)

    return Cvi, Cvi_true, sens_info_flag, a, Fi, K
