import matplotlib.pyplot as plt

def plot_polygon_with_samples(vx, vy, xp, yp):
    # Plot the polygon
    plt.fill(vx, vy, alpha=0.2, edgecolor='black', label='Polygon')

    # Plot the sample points
    plt.scatter(xp, yp, color='red', label='Sample Points')

    # Add labels and title to the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Polygon with Sample Points')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()
