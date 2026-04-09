import matplotlib.pyplot as plt
import numpy as np
"""
8.1 question 9
"""

act_scores = np.array([28, 25, 28, 27, 28, 33, 28, 29, 23, 27, 29, 28, 27, 29, 21, 28, 28, 26, 30, 24] )
gpas = np.array([3.84, 3.21, 3.23, 3.63, 3.75, 3.2, 3.41, 3.38, 3.53, 2.03, 3.75, 3.65, 3.87, 3.75, 1.66, 3.12, 2.96, 2.92, 3.1, 2.81])

"""
Construct linear lest squares approximation 
"""
x_squared = np.sum(act_scores ** 2)
x_i = np.sum(act_scores)
n = act_scores.size
x_y = np.sum(act_scores * gpas)
y_i = np.sum(gpas)

a = np.array([
            [x_squared, x_i],
            [x_i, n]
            ])
b = np.array([[x_y],[y_i]])

result = np.linalg.solve(a, b)
a_1 = result.item(0)
a_0 = result.item(1)

print(f"linear lest squares approximation:\n\ty = {a_1}x + {a_0}")

x_plot = np.linspace(20, 34, 500)
y_plot = a_1 * x_plot + a_0

plt.figure(figsize=(8,5))
plt.scatter(act_scores, gpas, label="Data points")
plt.xlabel("ACT Scores")
plt.ylabel("College GPAs")
plt.title("ACT Score vs. College GPA")
plt.plot(x_plot, y_plot, label="Least squares approximation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("8.1 question 9.png", dpi=200)
plt.show()