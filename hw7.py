import numpy as np
from scipy.sparse.linalg import cg

# Coefficient matrix A and vector b
A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
], dtype=float)

b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

def jacobi(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()
    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i + 1
        x = x_new
    return x, max_iter

def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for itr in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, itr + 1
        x = x_new
    return x, max_iter

def sor(A, b, omega=1.1, x0=None, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b) if x0 is None else x0.copy()

    for itr in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - omega) * x[i] + omega * (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, itr + 1
        x = x_new
    return x, max_iter

# Jacobi
x_jacobi, iter_jacobi = jacobi(A, b)
print("Jacobi Solution:", x_jacobi)
#print("Iterations:", iter_jacobi)

# Gauss-Seidel
x_gs, iter_gs = gauss_seidel(A, b)
print("\nGauss-Seidel Solution:", x_gs)
#print("Iterations:", iter_gs)

# SOR
x_sor, iter_sor = sor(A, b, omega=1.1)
print("\nSOR Solution (ω=1.1):", x_sor)
#print("Iterations:", iter_sor)

# Conjugate Gradient
x_cg, info = cg(A, b)
print("\nConjugate Gradient Solution:", x_cg)
#print("Conjugate Gradient info (0 = success):", info)
