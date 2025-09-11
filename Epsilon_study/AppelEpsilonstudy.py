#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from qiskit.primitives import Estimator, Sampler
from qiskit.quantum_info import Statevector

from hhl2 import HHL
from tridiagonal_toeplitz import TridiagonalToeplitz  # (num_state_qubits, main, off, trotter_steps)

# ========================= Paramètres =========================
x_inf, x_sup = 0.0, 1.0
N = 10                       # m = N-2 = 8 = 2^3
alfa, beta = -0.5, 0.5       # Dirichlet NON homogènes
EPSILONS = [5e-1, 1e-1, 1e-2, 1e-3, 1e-4]
RESCALE_LS = True            # Mise à l'échelle LS (recommandé)

estimator = Estimator()
sampler = Sampler()

dx = (x_sup - x_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N)
m = N - 2
assert int(np.log2(m)) == np.log2(m), "Choisir N tel que m=N-2 soit une puissance de 2."
NB = int(np.log2(m))

# ========================= Opérateurs discrets =========================
# Δ (non-SPD) sur la grille intérieure: schéma 1D [1, -2, 1]
A_delta = np.zeros((m, m))
np.fill_diagonal(A_delta, -2.0)
np.fill_diagonal(A_delta[1:], 1.0)
np.fill_diagonal(A_delta[:, 1:], 1.0)

# SPD pour HHL : -Δ
A_spd = -A_delta

# RHS brut: b0 = dx^2 * f(x_i), i=1..N-2, f(x)=10x (PDE: Δu = b0/dx^2)
b0 = np.zeros((m, 1))
for i in range(1, N - 1):
    b0[i - 1] = (10.0 * x[i]) * (dx ** 2)

# ===== Lifting (Dirichlet non homogènes) =====
# u_g(x) = alfa*(1-x) + beta*x  impose u(0)=alfa, u(1)=beta, et Δu_g = 0 (linéaire)
u_lift_full = alfa * (1 - x) + beta * x
u_lift_int = u_lift_full[1:-1].reshape(-1, 1)

# ===== Solution analytique (Δu = 10x, u(0)=alfa, u(1)=beta) =====
# u(x) = (5/3)x^3 + C1 x + C2, C2=alfa, C1 = beta - alfa - 5/3
u_exact_full = (5.0/3.0) * x**3 + (beta - alfa - 5.0/3.0) * x + alfa
u_exact_int = u_exact_full[1:-1].reshape(-1, 1)

# Sanity check (ordre O(dx^2) attendu)
fd_res_exact = float(np.linalg.norm((A_delta @ u_exact_int - b0).ravel()))
print(f"[Sanity] ||Δ u_exact - b0|| = {fd_res_exact:.3e}")

# ========================= Helpers =========================
def qpe_size(qc):
    try:
        return qc.qregs[1].size
    except Exception:
        return None

def trotter_steps_for_eps(eps, nl):
    """Heuristique simple: plus eps est petit, plus on augmente les pas."""
    if not nl or nl <= 0:
        return []
    return [max(1, int(np.ceil(np.sqrt(2**j / eps)))) for j in range(nl)]

def find_registers_simple(qc, nb):
    """Identifie système/QPE/flag/aux par tailles de registres."""
    regs = {reg.name: list(reg) for reg in qc.qregs}
    sizes = {n: len(qs) for n, qs in regs.items()}
    sys_name = next((n for n, s in sizes.items() if s == nb), None)
    if sys_name is None:
        raise RuntimeError("Registre système introuvable.")
    others = {n: s for n, s in sizes.items() if n != sys_name}
    qpe_name = max(others, key=others.get) if others else None
    cand1 = [n for n, s in sizes.items() if s == 1 and n not in (sys_name, qpe_name)]
    flag_name = cand1[0] if cand1 else None
    aux_names = [n for n in regs if n not in (sys_name, qpe_name, flag_name)]
    return dict(sys=regs[sys_name],
                qpe=(regs[qpe_name] if qpe_name else []),
                flag=(regs[flag_name] if flag_name else []),
                aux=sum((regs[n] for n in aux_names), []))

def build_indexer(qc, roles, nb):
    """
    Index du statevector pour |k>_sys avec post-sélection:
    flag=1, qpe=0..0, aux=0..0.
    """
    sys_qs = roles["sys"]
    sys_pos = {q: i for i, q in enumerate(sys_qs)}  # little-endian local
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
        return int(''.join(bits[::-1]), 2)  # base |q_{n-1}...q_0>
    return idx_for

def extract_solution(qc, nb, nl, A_ref, b_ref):
    """
    Post-sélection (flag=1, qpe=0..0, aux=0..0) puis mise à l'échelle par ||A vec||/||b||.
    Retourne un vecteur colonne (m,1).
    """
    psi = Statevector(qc).data
    roles = find_registers_simple(qc, nb)
    idx_for = build_indexer(qc, roles, nb)
    vec = np.zeros(1 << nb, dtype=complex)
    for k in range(1 << nb):
        vec[k] = psi[idx_for(k)]
    num = np.linalg.norm((A_ref @ vec).ravel())
    den = np.linalg.norm(b_ref.ravel()) + 1e-300
    scale = num / den
    if scale == 0:
        raise RuntimeError("Échelle nulle à l'extraction (post-sélection invalide ?).")
    return (vec / scale).reshape(-1, 1)

# ========================= Balayage en ε =========================
def run_sweep():
    out = {}
    for eps in EPSILONS:
        solver = HHL(estimator=estimator, sampler=sampler, epsilon=eps)
        solver._exact_reciprocal = False

        # (1) Taille QPE estimée avec (-Δ, -b0)
        qc_tmp = solver.construct_circuit(A_spd, (-b0).ravel())
        nl = qpe_size(qc_tmp)

        # (2) Étapes de Trotter (si backend les utilise)
        steps_vec = trotter_steps_for_eps(eps, nl)

        # (3) Hamiltonien Toeplitz pour -Δ (Dirichlet homogènes internalisées)
        main_diag, off_diag = (-2.0, 1.0)
        H_toep = TridiagonalToeplitz(NB, main_diag, off_diag, trotter_steps=steps_vec)

        # (4) Circuit HHL sur (-Δ, -b0) ⇒ solution v (CL homogènes)
        qc = solver.construct_circuit(H_toep, (-b0).ravel())
        v_int = extract_solution(qc, NB, nl, A_ref=A_spd, b_ref=(-b0)).real

        # (5) Mise à l'échelle LS (optionnelle) pour minimiser ||Δ(α v) - b0||
        if RESCALE_LS:
            Av = A_delta @ v_int
            num = (Av.conj().T @ b0).real.item()
            den = (Av.conj().T @ Av).real.item() + 1e-300
            alpha_ls = num / den
            v_int *= alpha_ls

        # (6) Reconstruction u = v + u_g (remise des CL non homogènes)
        u_int = v_int + u_lift_int
        u_full = np.zeros((N, 1))
        u_full[0, 0] = alfa
        u_full[-1, 0] = beta
        u_full[1:-1] = u_int

        # (7) Résidu & erreur (PDE cible Δu = b0)
        residual = np.linalg.norm((A_delta @ u_int - b0).ravel())
        mask = np.abs(u_exact_full) > 1e-12
        err_rel = 100 * np.abs((u_full.ravel()[mask] - u_exact_full[mask]) / u_exact_full[mask])
        mean_err = float(np.mean(err_rel))

        out[eps] = dict(nl=nl, steps_vec=steps_vec, u_full=u_full,
                        residual=residual, mean_err=mean_err)

        print(f"[ε={eps:.0e}] nl={nl:>2} steps={steps_vec}  "
              f"||Δu-b0||={residual:.3e}  mean err%={mean_err:.2f}")
    return out

print(f"[INFO] N={N}, m={m}=2^{NB} | PDE: Δu=f (Dirichlet non homog.) | HHL: -Δ avec RHS -b0")
results = run_sweep()

# ========================= Tracés =========================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Solutions pour chaque ε
for j, eps in enumerate(EPSILONS):
    d = results[eps]
    ax1.plot(x, d['u_full'].ravel(), 'x--', ms=3, alpha=0.9,
             color=colors[j % len(colors)], label=f"ε={eps:.0e}, steps={d['steps_vec']}")
ax1.plot(x, u_exact_full, 'k:', lw=2, label='Exact')
ax1.set_xlabel('x'); ax1.set_ylabel('u(x)')
ax1.grid(True); ax1.legend(fontsize=8)

# Erreur vs ε
eps_ticks = list(range(len(EPSILONS)))
ax2.plot(eps_ticks, [results[e]['mean_err'] for e in EPSILONS],
         'o--', label='Relative mean error (%)')
ax2.set_xticks(eps_ticks); ax2.set_xticklabels([f"{e:.0e}" for e in EPSILONS])
ax2.set_xlabel('ε'); ax2.set_ylabel('Error (%)'); ax2.set_title('Error vs ε')
ax2.grid(True); ax2.legend()

plt.tight_layout()
plt.show()
