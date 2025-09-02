import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Estimator, Sampler
from hhl2 import HHL

#===== DOMAIN =====
x_inf = 0
x_sup = 1
N = 18
dx = (x_sup - x_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N)

alfa = 0
beta = 0
mu = 10

#===== MATRIX A =====
A = np.zeros((N - 2, N - 2))
np.fill_diagonal(A, -2)
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)
A[0, 0] = -1
A[0, 1] = 1
A[-1, -2] = 1
A[-1, -1] = -1

#===== MATRIX B =====
b = np.zeros((N - 2, 1))
for i in range(1, N - 1):
    b[i - 1] = (np.cos(np.pi * x[i])) * dx**2
b[0] += alfa * dx
b[-1] -= beta * dx

for i in range(1):
    
    A_diff = A.copy()
    A_diff[np.arange(A.shape[0]), np.arange(A.shape[0])] += (2*mu) * dx**2
    
    #===== HHL SOLUTION =====
    solver = HHL(estimator=Estimator(), sampler=Sampler())
    result = solver.solve(A_diff, b)
    x_exp = np.array(result.solution).reshape(-1, 1)
    
    #===== CLASSICAL SOLUTION =====
    x_theo = np.linalg.solve(A_diff, b)
    
    #===== RECONSTRUCTION =====
    x_full_exp = np.zeros((N, 1))
    x_full_exp[1:-1] = x_exp
    x_full_exp[0] = x_exp[0] + alfa * dx
    x_full_exp[-1] = x_exp[-1] - beta * dx
    
    x_full_theo = np.zeros((N, 1))
    x_full_theo[1:-1] = x_theo
    x_full_theo[0] = x_theo[0] + alfa * dx
    x_full_theo[-1] = x_theo[-1] - beta * dx
    
    #===== EXACT SOLUTION =====
    # x_real = 5 * x**2 * (x / 3 - 1 / 2) + 5/12 #10x-5
    x_real = -1/(2.0 * np.pi ** 2)* np.cos(np.pi * x) #cos(pix)
    x_real = x_real.reshape(-1, 1)
    
    #===== ERROR =====
    tol = 1e-12  
    mask = np.abs(x_full_exp[:, 0]) > tol
    error = 100 * np.abs((x_real[mask] - x_full_exp[mask]) / x_real[mask])
    x_masked = x[mask]
    
    eTx = float(np.sum(x_exp))
    
    #===== PLOTTING =====
    fig, ax1 = plt.subplots()
    
    ax1.plot(x, x_full_exp, 'o-', label='HHL')
    ax1.plot(x, x_full_theo, '--', label='Classique')
    ax1.plot(x, x_real, 'x-', label='Exacte')
    ax1.set_xlabel('x')
    ax1.set_ylabel('p(x)')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(x_masked, error, 'o--', color='red', label='Erreur HHL (%)')
    ax2.set_ylabel('Erreur (%)')
    
    ax1.text(0.05, 0.95, f"$e^T x$ = {eTx:.2e}", transform=ax1.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    fig.suptitle(
    f'Plot of HHL, classical and real solutions for the penalty method with '
    rf'$\mu = {mu[i]:.2f}$, $\kappa = {np.linalg.cond(A_diff):.2f}$ for $N={N-2}$'
    )