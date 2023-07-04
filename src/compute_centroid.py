from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import pdist
from tools import compute_voronoi_with_boundaries
import numpy as np
from dintegrate import dintegrate
from globals import *

def compute_centroid(px, py, ai):

    # Compute local Voronoi region Vi
    bs_ext = np.array([[0, 1, 1, 0], [0, 0, 1, 1]]).T  # Environment boundary 
    points = np.vstack((px, py)).T
    vor, finite_points, finite_regions  = \
        compute_voronoi_with_boundaries(points, bs_ext, plot=False)  
    Cv = np.zeros((2, n))
    Cv_true = np.zeros((2, n))
    order = np.zeros(n)
    for i in range(n):
        region_index = vor.point_region[i]
        try:
            vertices = finite_regions[region_index]
            vx = vor.vertices[vertices, 0]
            vy = vor.vertices[vertices, 1]
            pos_x = np.where(px==vor.points[i, 0])[0][0]
            pos_y = np.where(py==vor.points[i, 1])[0][0]
            order[pos_x] = i

            if pos_x == pos_y:
                # Reorder centroids based on robot order
                Cv[:, pos_x], Cv_true[:, pos_x] = dintegrate(vx, vy, vor.points[i, :], ai[:, pos_x], pos_x)
            else:
                print('Mismatch in position found')
                Cv[:, pos_x] = vor.points[i, :].T
                Cv_true[:, pos_x] = vor.points[i, :].T

        except Exception as e:
            print(e)

    # Laplacian: Shared edge length as weight
    tri = Delaunay(points)
    ed = tri.simplices
    L = np.zeros((n, n))
    for ind in range(len(ed)):
        r1 = ed[ind, 0]
        r2 = ed[ind, 1]
        points = list(set(finite_regions[int(order[r1])]).intersection(finite_regions[int(order[r2])]))
        if len(points) == 2:
            edge_len = pdist(vor.vertices[points, :], 'euclidean')
        else:
            edge_len = 0
        L[r1, r2] = -edge_len
        L[r2, r1] = -edge_len
    L = L + np.diag(-1 * np.sum(L, 1))

    return Cv, Cv_true, L
