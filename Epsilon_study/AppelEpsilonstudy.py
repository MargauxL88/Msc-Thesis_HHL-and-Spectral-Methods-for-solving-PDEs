#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import Statevector

from hhl2 import HHL
from tridiagonal_toeplitz import TridiagonalToeplitz  

# ========================= Parameters =========================
x_inf, x_sup = 0.0, 1.0
N = 10                       
alfa, beta = -0.5, 0.5     
EPSILONS = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
RESCALE_LS = True        

estimator = Estimator()
sampler = Sampler()

dx = (x_sup - x_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N)
m = N - 2
NB = int(np.log2(m))

# ========================= Discrete operators =========================
A_delta = np.zeros((m, m))
np.fill_diagonal(A_delta, -2.0)
np.fill_diagonal(A_delta[1:], 1.0)
np.fill_diagonal(A_delta[:, 1:], 1.0)
A_spd = -A_delta

b0 = np.zeros((m, 1))
for i in range(1, N - 1):
    b0[i - 1] = (10.0 * x[i]) * (dx ** 2)

# ===== Analytical solution =====
u_exact_full = (5.0/3.0) * x**3 + (beta - alfa - 5.0/3.0) * x + alfa
u_exact_int = u_exact_full[1:-1].reshape(-1, 1)

# ========================= Helpers =========================
def qpe_size(qc):
    try:
        return qc.qregs[1].size
    except Exception:
        return None

def trotter_steps_for_eps(eps, nl):
    if not nl or nl <= 0:
        return []
    return [max(1, int(np.ceil(np.sqrt(2**j / eps)))) for j in range(nl)]

def build_indexer(qc, roles, nb):
    sys_qs = roles["sys"]
    sys_pos = {q: i for i, q in enumerate(sys_qs)}  
    set_sys, set_qpe = set(sys_qs), set(roles["qpe"])
    set_flag, set_aux = set(roles["flag"]), set(roles["aux"])
    qubits = list(qc.qubits)

    def idx_for(k: int) -> int:
        bits = []
        for q in qubits:
            if q in set_sys:
                b = '1' if ((k >> sys_pos[q]) & 1) else '0'
            elif q in set_qpe or q in set_aux:
                b = '0'
            elif q in set_flag:
                b = '1'
            else:
                b = '0'
            bits.append(b)
        return int(''.join(bits[::-1]), 2)  
    return idx_for

def extract_solution(qc, nb, nl, A_ref, b_ref):
    psi = Statevector(qc).data
    roles = find_registers_simple(qc, nb)
    idx_for = build_indexer(qc, roles, nb)
    vec = np.zeros(1 << nb, dtype=complex)
    for k in range(1 << nb):
        vec[k] = psi[idx_for(k)]
    num = np.linalg.norm((A_ref @ vec).ravel())
    den = np.linalg.norm(b_ref.ravel()) + 1e-300
    scale = num / den
    return (vec / scale).reshape(-1, 1)

# ========================= ε-study =========================
def run_sweep():
    out = {}
    for eps in EPSILONS:
        solver = HHL(estimator=estimator, sampler=sampler, epsilon=eps)
        solver._exact_reciprocal = False

        #Noiseless solution
        qc_tmp = solver.construct_circuit(A_spd, (-b0).ravel())
        nl = qpe_size(qc_tmp)

        steps_vec = trotter_steps_for_eps(eps, nl)

        #Trotter-Suzuki decomposition
        main_diag, off_diag = (-2.0, 1.0)
        H_toep = TridiagonalToeplitz(NB, main_diag, off_diag, trotter_steps=steps_vec)
        
        qc = solver.construct_circuit(H_toep, (-b0).ravel())
        v_int = extract_solution(qc, NB, nl, A_ref=A_spd, b_ref=(-b0)).real

        #Reconstruction
        u_int = v_int + u_lift_int
        u_full = np.zeros((N, 1))
        u_full[0, 0] = alfa
        u_full[-1, 0] = beta
        u_full[1:-1] = u_int

        #Error and residuals between the noise and noiseless simu
        residual = np.linalg.norm((A_delta @ u_int - b0).ravel())
        mask = np.abs(u_exact_full) > 1e-12
        err_rel = 100 * np.abs((u_full.ravel()[mask] - u_exact_full[mask]) / u_exact_full[mask])
        mean_err = float(np.mean(err_rel))

        out[eps] = dict(nl=nl, steps_vec=steps_vec, u_full=u_full,
                        residual=residual, mean_err=mean_err)

        print(f"[ε={eps:.0e}] nl={nl:>2} steps={steps_vec}  "
              f"||Δu-b0||={residual:.3e}  mean err%={mean_err:.2f}")
    return out
    
results = run_sweep()

# ========================= Plots =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for j, eps in enumerate(EPSILONS):
    d = results[eps]
    ax1.plot(x, d['u_full'].ravel(), 'x--', ms=3, alpha=0.9,
             color=colors[j % len(colors)], label=f"ε={eps:.0e}, steps={d['steps_vec']}")
ax1.plot(x, u_exact_full, 'k:', lw=2, label='Exact')
ax1.set_xlabel('x'); ax1.set_ylabel('u(x)')
ax1.grid(True); ax1.legend(fontsize=8)

eps_ticks = list(range(len(EPSILONS)))
ax2.plot(eps_ticks, [results[e]['mean_err'] for e in EPSILONS],
         'o--', label='Relative mean error (%)')
ax2.set_xticks(eps_ticks); ax2.set_xticklabels([f"{e:.0e}" for e in EPSILONS])
ax2.set_xlabel('ε'); ax2.set_ylabel('Error (%)'); ax2.set_title('Error vs ε')
ax2.grid(True); ax2.legend()

plt.tight_layout()
plt.show()
