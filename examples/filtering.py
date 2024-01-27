import numpy as np
import matplotlib.pyplot as plt

n_points, end_x = 600, 8
x = np.linspace(0, end_x, n_points)
n = np.random.normal(scale=0.5, size=(n_points))
s = np.sin(2 * np.pi * x)
y = s + n

plt.plot(x, s)
plt.plot(x, y, ls="none", marker=".")
plt.show()
