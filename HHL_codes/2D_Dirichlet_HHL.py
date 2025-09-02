import numpy as np
from qiskit.primitives import Estimator, Sampler
from hhl2 import HHL
import matplotlib.pyplot as plt

#===== DOMAIN =====
x_inf, x_sup = 0.0, 1.0
y_inf, y_sup = 0.0, 1.0
N = 5  # (n_interior = N-1)
dx = (x_sup - x_inf) / (N - 1)
dy = dx
x = np.linspace(x_inf, x_sup, N + 1)
y = np.linspace(y_inf, y_sup, N + 1)
X, Y = np.meshgrid(x, y, indexing='ij')

#===== BCs =====
p = np.zeros((N + 1, N + 1))
for i in range(N + 1):
    p[i, 0] = 0.0
    p[i, N] = 0.0
for j in range(N + 1):
    p[0, j] = 0.0
    p[N, j] = 0.0

#===== MATRIX A =====
N_inner = (N - 1) * (N - 1)
A = np.zeros((N_inner, N_inner))

for j in range(N - 1):
    for i in range(N - 1):
        row_idx = i + j * (N - 1)
        A[row_idx, row_idx] = -4.0
        if i > 0:
            A[row_idx, row_idx - 1] = 1.0
        if i < N - 2:
            A[row_idx, row_idx + 1] = 1.0
        if j > 0:
            A[row_idx, row_idx - (N - 1)] = 1.0
        if j < N - 2:
            A[row_idx, row_idx + (N - 1)] = 1.0

#===== MATRIX b =====
b = np.zeros(N_inner)
for j in range(N - 1):
    for i in range(N - 1):
        row_idx = i + j * (N - 1)
        ii, jj = i + 1, j + 1  
        f_ij = np.sin(2*np.pi * X[ii, jj]) * np.sin(2*np.pi * Y[ii, jj]) * (dx ** 2)
        b[row_idx] = f_ij

        if i == 0:
            b[row_idx] -= p[0, j + 1]
        if i == N - 2:
            b[row_idx] -= p[N, j + 1]     
        if j == 0:
            b[row_idx] -= p[i + 1, 0]        
        if j == N - 2:
            b[row_idx] -= p[i + 1, N]      

#===== HHL SOLUTION =====
solver = HHL(estimator=Estimator(), sampler=Sampler())
result = solver.solve(A, b)
x_exp = np.array(result.solution).reshape(-1, 1)

#===== CLASSICAL SOLUTION =====
x_theo = np.linalg.solve(A, b).reshape(-1, 1)

#===== REAL SOLUTION =====
x_real = -(np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)) / (4 * np.pi ** 2)

#===== RECONSTRUCTION =====
p_exp = p.copy()
p_theo = p.copy()
for j in range(N - 1):
    for i in range(N - 1):
        row_idx = i + j * (N - 1)
        p_exp[i + 1, j + 1] = x_exp[row_idx, 0]
        p_theo[i + 1, j + 1] = x_theo[row_idx, 0]

#===== ERROR =====
error_grid = np.zeros_like(p)
for j in range(N - 1):
    for i in range(N - 1):
        ii, jj = i + 1, j + 1
        denom = x_real[ii, jj]
        if abs(denom) > 1e-12:
            error_grid[ii, jj] = 100.0 * abs((p_exp[ii, jj] - denom) / denom)
        else:
            error_grid[ii, jj] = 0.0

#===== PLOTTING =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

contour1 = axes[0].contourf(X, Y, x_real, levels=50, cmap='plasma')
cbar1 = fig.colorbar(contour1, ax=axes[0])
cbar1.set_label('p_exact(x,y)', rotation=270, labelpad=15)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title('Exact (analytical)')

contour2 = axes[1].contourf(X, Y, p_exp, levels=50, cmap='plasma')
cbar2 = fig.colorbar(contour2, ax=axes[1])
cbar2.set_label('p_HHL(x,y)', rotation=270, labelpad=15)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_title('HHL')

plt.tight_layout(); plt.show()

plt.figure(figsize=(6, 5))
contour3 = plt.contourf(X, Y, error_grid, levels=50, cmap='viridis')
cbar3 = plt.colorbar(contour3)
cbar3.set_label('Error (%)', rotation=270, labelpad=15)
plt.tight_layout(); plt.show()
