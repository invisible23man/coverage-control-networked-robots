import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from globals import n

# Sensory basis function
def kappa(qx, qy):
    sigma = 0.18  # Gaussian sd
    mu = np.array([1, 1, 1, 3, 3, 3, 5, 5, 5, 
                    1, 3, 5, 1, 3, 5, 1, 3, 5]) / 6  # Gaussian means
    kappa_ =  np.array([1 / (sigma**2 * 2 * np.pi) * 
        np.exp(-((qx - mu[i])**2 + (qy - mu[j])**2) / (2 * sigma**2)) 
        for i, j in zip(range(9), range(9, 18))]).squeeze().T

    return kappa_

# def kappa(qx,qy):
#     sigma = 0.18  # Gaussian sd
#     mu = np.array([[1, 1, 1, 3, 3, 3, 5, 5, 5], 
#                 [1, 3, 5, 1, 3, 5, 1, 3, 5]])/6; # Gaussian means    
#     kappa = \
#         [1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,0]) ** 2 + (qy-mu[1,0]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,1]) ** 2 + (qy-mu[1,1]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,2]) ** 2 + (qy-mu[1,2]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,3]) ** 2 + (qy-mu[1,3]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,4]) ** 2 + (qy-mu[1,4]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,5]) ** 2 + (qy-mu[1,5]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,6]) ** 2 + (qy-mu[1,6]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,7]) ** 2 + (qy-mu[1,7]) ** 2) /(2*sigma**2)),
#          1/(sigma**2 * 2 * np.pi) * np.exp(-((qx-mu[0,8]) ** 2 + (qy-mu[1,8]) ** 2) /(2*sigma**2))]

#     return kappa

def reshape_state(z:np.array):
    z = z.reshape(-1,1)
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

def compute_voronoi_with_boundaries(points, boundary_points, plot=False):
    # Add the boundary points to the input points
    points_with_boundary = np.concatenate([points, boundary_points])

    # Compute Voronoi diagram
    vor = Voronoi(points_with_boundary)

    # Filter out infinite regions and regions associated with boundary points
    # finite_regions = [region for region in vor.regions if -1 not in region and len(region) > 0 and not any(idx >= len(points) for idx in region)]
    finite_regions = [region for region in vor.regions if -1 not in region and len(region) > 0 ]
    finite_points = [vor.vertices[region] for region in finite_regions]

    if plot:
        # Plot
        for region in finite_points:
            plt.fill(*zip(*region), alpha=0.4)

        plt.plot(points[:, 0], points[:, 1], 'ko')
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)

        # Plot the points
        plt.scatter(points[:, 0], points[:, 1], color='black', label='Points')

        # Plot the vertices
        plt.scatter(vor.vertices[:, 0], vor.vertices[:, 1], color='blue', marker='s', label='Vertices')

        # Add a legend
        plt.legend()

        # Add a caption
        caption = f"Bounded Voronoi Diagram of {points.shape[0]} Random Points"
        plt.figtext(0.5, 0.01, caption, ha='center', fontsize=10)

        plt.show()

    return vor, finite_points, finite_regions

