import numpy as np

def crout_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for j in range(n):
        U[j][j] = 1
        for i in range(j, n):
            s1 = sum(L[i][k] * U[k][j] for k in range(i))
            L[i][j] = A[i][j] - s1
        for i in range(j+1, n):
            s2 = sum(L[j][k] * U[k][i] for k in range(j))
            U[j][i] = (A[j][i] - s2) / L[j][j]
    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))) / U[i][i]
    return x

def solve_crout(A, b):
    L, U = crout_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

# Kode Testing
A = np.array([[2, -1, 1],
              [-1, 3, 2],
              [1, 1, 3]])
b = np.array([3, 5, 1])

solution = solve_crout(A, b)
print("Solusi sistem persamaan linear:")
print(solution)
