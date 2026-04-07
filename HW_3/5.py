import numpy as np
"""
Home work #3 question 5 calculations 
"""
x = np.array([4.0, 4.2, 4.5, 4.7, 5.1, 5.5, 5.9, 6.3, 6.8, 7.1])
y = np.array([102.56, 113.18, 130.11, 142.05, 167.53, 195.14, 224.87, 256.73, 299.5, 326.72])

num_data_points = len(x)
sum_x = np.sum(x)
sum_x_squared = np.sum(x**2)
sum_x_cubed = np.sum(x**3)
sum_x_fourth = np.sum(x**4)
sum_y = np.sum(y)
sum_x_y = np.sum(x*y)
sum_x_squared_y = np.sum((x**2) * y)

print(f"number of element:\n{len(x)}")
print(f"sumation of x_i:\n{sum_x}")    
print(f"sumation of x_i squared:\n{sum_x_squared}")
print(f"sumation of x_i cubed:\n{sum_x_cubed}")
print(f"sumation of x_i fourth:\n{sum_x_fourth}")
print(f"sumation of y_i:\n{sum_y}")
print(f"sumation of y_i * x_i:\n{sum_x_y}")
print(f"sumation of y_i * x_i^2:\n{sum_x_squared_y}")

a_1_matrix = np.array([
    [sum_x_squared, sum_x],
    [sum_x, num_data_points]
])

b_2_matrix = np.array([sum_x_y, sum_y])

result_x_a = np.linalg.solve(a_1_matrix, b_2_matrix)
print(f"Result for 5 A:\n{result_x_a}")

a = result_x_a[0]
b = result_x_a[1]

error = np.sum((y - (a * x + b)) ** 2)

print(f"Mean squarred error:\n{error}")

a_matrix = np.array([
                [num_data_points, sum_x, sum_x_squared],
                [sum_x, sum_x_squared, sum_x_cubed],
                [sum_x_squared, sum_x_cubed, sum_x_fourth]
            ])

b_matrix = np.array([sum_y, sum_x_y, sum_x_squared_y])

result = np.linalg.solve(a_matrix, b_matrix)
print(f"Result for 5 B:\n{result}")

a_0 = result[0]
a_1 = result[1]
a_2 = result[2]

B_error = np.sum((y - (a_2*x**2 + a_1*x + a_0))**2)
print(f"Error for 5 B:\n{B_error}")