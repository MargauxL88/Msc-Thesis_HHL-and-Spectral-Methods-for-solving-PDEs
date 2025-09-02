import numpy as np
from qiskit.primitives import Estimator, Sampler
from hhl2 import HHL
import matplotlib.pyplot as plt
import time

# ==== PARAMETERS ====
x_inf, x_sup = 0,1
y_inf, y_sup = 0,1
N = 5                            
dx = (x_sup - x_inf) / N
dy = (y_sup - y_inf) / N

x = np.linspace(x_inf, x_sup, N + 1)
y = np.linspace(y_inf, y_sup, N + 1)
X, Y = np.meshgrid(x, y, indexing='ij')

alpha = beta = gamma = delta = 0.0

N2 = (N - 1) * (N - 1)

# ====== MATRIX A ======
A = np.zeros((N2, N2))
for i in range(N - 1):
    for j in range(N - 1):
        idx = i + (N - 1) * j

        is_corner = (i in (0, N-2)) and (j in (0, N-2))
        is_edge   = ((i in (0, N-2)) or (j in (0, N-2))) and not is_corner

        A[idx, idx] = -2.0 if is_corner else (-3.0 if is_edge else -4.0)

        if i > 0:       A[idx, idx - 1]         = 1.0
        if i < N - 2:   A[idx, idx + 1]         = 1.0
        if j > 0:       A[idx, idx - (N - 1)]   = 1.0
        if j < N - 2:   A[idx, idx + (N - 1)]   = 1.0

# ====== MATRIX b ======
b = np.zeros(N2)
for i in range(N - 1):
    for j in range(N - 1):
        row_idx = i + j * (N - 1)
        ii, jj = i + 1, j + 1                   
        f_ij= np.cos(2*np.pi * X[ii, jj]) * np.cos(2*np.pi * Y[ii, jj]) * (dx ** 2)
        b[row_idx] = f_ij

# ===== NEUMANN BCs =====
for j in range(1, N - 2):
    idx_left  = 0 + (N - 1) * j
    idx_right = (N - 2) + (N - 1) * j
    b[idx_left]  += alpha * dx
    b[idx_right] -= beta  * dx

for i in range(1, N - 2):
    idx_bottom = i + (N - 1) * 0
    idx_top    = i + (N - 1) * (N - 2)
    b[idx_bottom] += gamma * dy
    b[idx_top]    -= delta * dy

idx_bl = 0
idx_br = (N - 2)
idx_tl = (N - 1) * (N - 2)
idx_tr = (N - 2) + (N - 1) * (N - 2)
b[idx_bl] += alpha * dx + gamma * dy
b[idx_br] += (-beta * dx + gamma * dy)
b[idx_tl] += (alpha * dx - delta * dy)
b[idx_tr] += (-beta * dx - delta * dy)

# ===== MU  =====
mu_list = [100.0]   

for mu_val in mu_list:

    A_mu = A.copy()
    A_mu[np.arange(N2), np.arange(N2)] += (2.0 * mu_val) * (dx ** 2)

    # === HHL ===
    solver = HHL(estimator=Estimator(), sampler=Sampler())
    t0 = time.time()
    result = solver.solve(A_mu, b)
    t_hhl = time.time() - t0
    x_exp = np.array(result.solution).reshape(-1, 1)   # état normalisé

    # === CLASSICAL SOL ===
    t0 = time.time()
    x_theo = np.linalg.solve(A_mu, b).reshape(-1, 1)
    t_classic = time.time() - t0

    # ==== RECONSTRUCTION =====
    p_exp  = np.zeros((N + 1, N + 1))
    p_theo = np.zeros((N + 1, N + 1))
    p_exp[1:N, 1:N]  = x_exp.reshape((N - 1, N - 1))
    p_theo[1:N, 1:N] = x_theo.reshape((N - 1, N - 1))

    # ==== REAL SOLUTION ====
    p_real = 1 / (-8.0 * np.pi ** 2 + 2.0 * mu_val) * np.cos(2*np.pi * X) * np.cos(2*np.pi * Y)

    # ===== ERROR ======
    error_grid = np.zeros_like(p_real)
    for i in range(1, N):
        for j in range(1, N):
            ref = p_real[i, j]
            if abs(ref) > 1e-14:
                error_grid[i, j] = 100.0 * abs((p_exp[i, j] - ref) / ref)

    mean_error = float(np.mean(error_grid[1:N, 1:N]))

    # ===== PLOTTING =====
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    
    vmin = min(p_exp.min(), p_real.min())
    vmax = max(p_exp.max(), p_real.max())
    levels = np.linspace(vmin, vmax, 50)
    
    cf0 = axes[0].contourf(X, Y, p_exp, levels=levels, cmap='plasma')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_title(f'HHL')
    
    cf1 = axes[1].contourf(X, Y, p_real, levels=levels, cmap='plasma')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
    axes[1].set_title('Exact (analytical)')
    
    cbar = fig.colorbar(cf1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label('p(x,y)', rotation=270, labelpad=15)
    
    plt.show()
    
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(X, Y, error_grid, levels=50, cmap='viridis')
    cbar = plt.colorbar(contour)
    cbar.set_label('Error (%)', rotation=270, labelpad=15)
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout(); plt.show()
