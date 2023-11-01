matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the data
X = np.genfromtxt('train_x.csv', delimiter=',', skip_header=1)
y = np.genfromtxt('train_y.csv', delimiter=',', skip_header=1)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(X[:, 0], X[:, 1], y)

ax.set_xlabel('Left vector of X')
ax.set_ylabel('Right vector of X')
ax.set_zlabel('Labels y')

plt.show()