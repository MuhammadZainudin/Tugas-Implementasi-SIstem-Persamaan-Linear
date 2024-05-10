import numpy as np

def solve_linear_system(A, b):
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    return x

# Kode Testing
A = np.array([[2, -1, 1],
              [-1, 3, 2],
              [1, 1, 3]])
b = np.array([3, 5, 1])

solution = solve_linear_system(A, b)
print("Solusi sistem persamaan linear:")
print(solution)
