import numpy as np
from sklearn.neighbors import KernelDensity

# Generate sample weed concentration data
weed_concentration = np.concatenate([
    np.random.normal(loc=[0.2, 0.2], scale=[0.1, 0.05], size=(100, 2)),
    np.random.normal(loc=[0.7, 0.7], scale=[0.2, 0.15], size=(100, 2)),
    np.random.normal(loc=[0.5, 0.5], scale=[0.3, 0.1], size=(100, 2))
])

# Define the grid for estimation
grid_resolution = 0.01
x_min, x_max = 0, 1
y_min, y_max = 0, 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_resolution),
                     np.arange(y_min, y_max, grid_resolution))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Perform kernel density estimation
bandwidth = 0.035  # Adjust this value to control the spread/variance
kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
kde.fit(weed_concentration)
weed_density = np.exp(kde.score_samples(grid_points))

# Reshape the density values to match the grid shape
density_map = weed_density.reshape(xx.shape)

# Visualization (example using Matplotlib)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, density_map, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Weed Concentration')
ax.set_title('Estimated Weed Concentration with Diverse Distribution')
plt.show()
