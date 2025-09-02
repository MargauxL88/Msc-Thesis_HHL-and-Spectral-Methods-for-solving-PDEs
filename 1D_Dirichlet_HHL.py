#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:37:22 2025

@author: Margaux
"""

import numpy as np
from qiskit.primitives import Estimator, Sampler
import hhl2
from hhl2 import HHL
from tridiagonal_toeplitz import TridiagonalToeplitz
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import time

#===== DOMAIN =====
x_inf = 0
x_sup = 1
N =34 # n = N-2
dx = (x_sup - x_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N)

alfa = 0
beta = 0

# =====  MATRIX A =====
A = np.zeros((N - 2, N - 2))
np.fill_diagonal(A, -2)
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:, 1:], 1)

#===== b =====
b = np.zeros((N - 2, 1))
for i in range(1, N - 1):
    b[i - 1] = (10*x[i]) * (dx ** 2)
b[0] -= alfa
b[-1] -= beta

#===== HHL =====
solver = HHL(estimator=Estimator(), sampler=Sampler())
t_0 = time.time()
result = solver.solve(A, b)
t_hhl = time.time() - t_0
x_exp = np.array(result.solution).reshape(-1, 1)

#===== CLASSICAL SOL =====
x_theo = np.linalg.solve(A, b)

#===== EXACT SOLUTION =====
x_real = 5/3*(x**3-x)

#===== RECONSTRUCTION =====
x_full_exp = np.zeros((N, 1))
x_full_exp[0] = alfa
x_full_exp[-1] = beta
x_full_exp[1:-1] = x_exp

x_full_theo = np.zeros((N, 1))
x_full_theo[0] = alfa
x_full_theo[-1] = beta
x_full_theo[1:-1] = x_theo

#===== ERROR =====
mask = np.abs(x_real) > 1e-8
x_masked = x[mask]
error = 100 * np.abs((x_full_exp[mask].ravel() - x_real[mask]) / x_real[mask])

mean_error = np.mean(error)

#===== PLOTTING ======
fig, ax1 = plt.subplots(figsize=(8,6))

ax1.plot(x, x_full_exp, 'o-', label='HHL')
ax1.plot(x, x_full_theo, '--', label='Classical')
ax1.plot(x, x_real, 'x-', label='Real solution')
ax1.set_xlabel('x')
ax1.set_ylabel('p(x)')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(x, error, 'o--', color='red', label='Error of HHL')
ax2.set_ylabel('Error (%)')
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.title(f'Comparaison : HHL vs Classical Resolution for n={N-2}')
plt.tight_layout()
plt.show()

print(f'Mean Error = {mean_error:.4f}%')
