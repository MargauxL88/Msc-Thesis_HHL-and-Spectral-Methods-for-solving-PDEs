import numpy as np
from qiskit.primitives import Estimator, Sampler
from hhl2 import HHL
import matplotlib.pyplot as plt
import time

# ==== PARAMETERS =====
x_inf, x_sup = 0, 1
y_inf, y_sup = 0, 1
N = 3
dx = (x_sup - x_inf) / (N - 1)
dy = (y_sup - y_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N + 1)
y = np.linspace(y_inf, y_sup, N + 1)
X, Y = np.meshgrid(x, y, indexing='ij')

alpha = beta = gamma = delta = 0.0

N2 = (N - 1) * (N - 1)

# ====== MATRIX A =======
A = np.zeros((N2, N2))
for i in range(N - 1):
    for j in range(N - 1):
        idx = i + (N - 1) * j

        is_corner_bottom_left = (i == 0 and j == 0)
        is_corner_bottom_right = (i == N - 2 and j == 0)
        is_corner_top_left = (i == 0 and j == N - 2)
        is_corner_top_right = (i == N - 2 and j == N - 2)
        is_corner = (is_corner_bottom_left or is_corner_bottom_right or
                     is_corner_top_left or is_corner_top_right)

        is_edge_left = (i == 0 and not is_corner)
        is_edge_right = (i == N - 2 and not is_corner)
        is_edge_bottom = (j == 0 and not is_corner)
        is_edge_top = (j == N - 2 and not is_corner)
        is_edge = is_edge_left or is_edge_right or is_edge_bottom or is_edge_top

        if is_corner:
            A[idx, idx] = -2.0
        elif is_edge:
            A[idx, idx] = -3.0
        else:
            A[idx, idx] = -4.0

        if i > 0:
            A[idx, idx - 1] = 1.0
        if i < N - 2:
            A[idx, idx + 1] = 1.0
        if j > 0:
            A[idx, idx - (N - 1)] = 1.0
        if j < N - 2:
            A[idx, idx + (N - 1)] = 1.0

# ====== MATRIX b ======
b = np.zeros(N2)
for i in range(N - 1):
    for j in range(N - 1):
        row_idx = i + j * (N - 1)
        ii, jj = i + 1, j + 1
        f_ij = np.cos(2*np.pi * X[ii, jj]) * np.cos(2*np.pi * Y[ii, jj]) * (dx ** 2)
        b[row_idx] = f_ij

# ===== NEUMANN BCs =====
b = b.copy()
for j in range(1, N - 2):
    idx_left = 0 + (N - 1) * j
    b[idx_left] += alpha * dx
    idx_right = (N - 2) + (N - 1) * j
    b[idx_right] -= beta * dx

for i in range(1, N - 2): 
    idx_bottom = i + (N - 1) * 0
    b[idx_bottom] += gamma * dx
    idx_top = i + (N - 1) * (N - 2)
    b[idx_top] -= delta * dx

idx_bl = 0
b[idx_bl] += alpha * dx + gamma * dx
idx_br = (N - 2)
b[idx_br] += (-beta * dx + gamma * dx)  
idx_tl = (N - 1) * (N - 2)
b[idx_tl] += (alpha * dx - delta * dx)
idx_tr = (N - 2) + (N - 1) * (N - 2)
b[idx_tr] += (-beta * dx - delta * dx)

# ===== LAGRANGE =====
e = np.ones((N2, 1))
A_aug = np.block([
    [A, e],
    [e.T, np.zeros((1, 1))]
])
b = b.reshape(-1, 1)
b_aug = np.vstack([b, [[0.0]]])

# ====== PADDING =======
def next_power_of_two(x):
    return 1 if x == 0 else 2 ** ((x - 1).bit_length())

n = A_aug.shape[0]           
m = next_power_of_two(n)
epsilon = 5e-2

if m > n:
    A_padded = np.zeros((m, m))
    A_padded[:n, :n] = A_aug
    A_padded[n:, n:] = epsilon * np.eye(m - n)   
    b_padded = np.zeros((m, 1))
    b_padded[:n, 0] = b_aug.flatten()
else:
    A_padded = A_aug
    b_padded = b_aug

# ===== HHL ======
solver = HHL(estimator=Estimator(), sampler=Sampler())

t0 = time.time()
result = solver.solve(A_padded, b_padded)
t_hhl = time.time() - t0

x_exp = np.array(result.solution).reshape(-1, 1)   
x_exp_p = x_exp[:n - 1]                            

# ===== CLASSICAL =====
t0 = time.time()
x_theo = np.linalg.solve(A_padded, b_padded).reshape(-1, 1)
t_classic = time.time() - t0
x_theo_p = x_theo[:n - 1]


# ====== RECONSTRUCTION =======
p_exp = np.zeros((N + 1, N + 1))
p_exp[1:N, 1:N] = x_exp_p.reshape((N - 1, N - 1))

p_theo = np.zeros((N + 1, N + 1))
p_theo[1:N, 1:N] = x_theo_p.reshape((N - 1, N - 1))

# ===== REAL SOLUTION =====
p_real = -(1.0 / (8.0 * np.pi ** 2)) * np.cos(2*np.pi * X) * np.cos(2*np.pi * Y)

# ====== ERROR =======
error_grid = np.zeros_like(p_real)
for i in range(1, N):
    for j in range(1, N):
        ref = p_real[i, j]
        if abs(ref) > 1e-14:
            rel_err = 100.0 * abs((p_exp[i, j] - ref) / ref)
        else:
            rel_err = 0.0
        error_grid[i, j] = rel_err

mean_error = np.mean(error_grid[1:N, 1:N])

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
plt.title('Error grid')
plt.tight_layout(); plt.show()
