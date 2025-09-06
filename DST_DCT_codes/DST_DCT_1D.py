#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit.library import QFT, UnitaryGate
from scipy.fft import dct, dst, fft, fftfreq, ifft

def safe_norm(x):
    """||x||_2 renvoyée comme scalaire NumPy, sans cast Python ni or/and."""
    return np.linalg.norm(np.asarray(x).ravel())

def normalize(v, eps=1e-15):
    n = safe_norm(v)
    return v if not np.isfinite(n) or n <= eps else (v / n)

# ===== PARAMETERS ======
num_qubits = 6
N  = 2**num_qubits
N2 = 2*N
x  = np.linspace(0.0, 1.0, N, endpoint=False)

function = 'cos'          
BCs      = 'neumann'   
APPLY_BANDPASS = True        

# ===== FUNCTIONS ======
if BCs == 'dirichlet':
    if function == 'linear':
        f_x_raw = 10*x
        x_real  = (5/3)*(x**3 - x)
    elif function == 'sin':
        f_x_raw = -4*np.pi**2*np.sin(2*np.pi*x)
        x_real  =  np.sin(2*np.pi*x)
    elif function == 'Heaviside':
        f_x_raw = 2*np.heaviside(x-0.5, 0.5) - 1
        x_real  = np.where(x<=0.5, -0.5*x**2 + 0.75*x, 0.5*x**2 - 0.25*x + 0.25)
    else:
        raise ValueError("Unsupported function for Dirichlet.")
elif BCs == 'neumann':
    if function == 'cos':
        f_x_raw = np.cos(2*np.pi*x)
        x_real  = -(1/4)*np.pi**2*np.cos(2*np.pi*x)
    elif function == 'linear':
        f_x_raw = 10*x - 5
        x_real  = 5/3*x**3 - 5/2*x**2 + 5/12
    else:
        raise ValueError("Unsupported function for Neumann.")
else:
    raise ValueError("BCs must be 'neumann' or 'dirichlet'.")

# ===== EMBEDING ======
f_x  = normalize(f_x_raw)
f_pad = np.zeros(N2, dtype=complex)
if BCs == 'dirichlet':  
    f_pad[N:] = f_x

# ===== BUILDING BLOCS ======
def B_gate_user():
    B = (1/np.sqrt(2.0)) * np.array([[1,  1j],
                                     [1, -1j]], dtype=complex)
    return UnitaryGate(B, label="B_user")

def inc_le(qc: QuantumCircuit, le_bits, control=None):
    if not le_bits: return
    if control is None: qc.x(le_bits[0])
    else:               qc.cx(control, le_bits[0])
    for i in range(1, len(le_bits)):
        ctrls = ([control] if control is not None else []) + list(le_bits[:i])
        qc.mcx(ctrls, le_bits[i])

def twos_complement_ctrl_le(qc: QuantumCircuit, le_bits, control):
    for q in le_bits:
        qc.cx(control, q)
    inc_le(qc, le_bits, control=control)

def permutation_matrix_Pn(size):
    P = np.zeros((size, size), dtype=complex)
    for i in range(size-1): P[i, i+1] = 1
    P[size-1, 0] = 1
    return P

def permutation_matrix_Pn_inv(size):
    Pinv = np.zeros((size, size), dtype=complex)
    for i in range(1, size): Pinv[i, i-1] = 1
    Pinv[0, size-1] = 1
    return Pinv

# ===== QUANTUM CIRCUITS ======
def create_transform(nq_plus_1: int) -> QuantumCircuit:

    qr = QuantumRegister(nq_plus_1, 'q')
    qc = QuantumCircuit(qr, name="U_Tdag_F_T_MSB")

    anc     = qr[-1]            
    data    = list(qr[:-1])   
    data_le = list(qr[:-1])    

    qc.append(B_gate_user(), [anc])
    twos_complement_ctrl_le(qc, data_le, control=anc)

    size_sub = 2**len(data)
    qc.append(UnitaryGate(permutation_matrix_Pn(size_sub),    "P"),    data)
    qc.append(QFT(nq_plus_1, do_swaps=True).to_gate(label="F_2N"), qr[:])
    qc.append(UnitaryGate(permutation_matrix_Pn_inv(size_sub), "P^-1"), data)

    twos_complement_ctrl_le(qc, data_le, control=anc)
    qc.append(B_gate_user().adjoint(), [anc])
    return qc

def qc_apply_U(state_vec, nq_plus_1: int) -> QuantumCircuit:
    qr = QuantumRegister(nq_plus_1, 'q'); qc = QuantumCircuit(qr)
    state_vec = normalize(state_vec)
    qc.initialize(state_vec, qr)
    return qc.compose(create_transform(nq_plus_1))

def qc_apply_U_inv(state_vec, nq_plus_1: int) -> QuantumCircuit:
    qr = QuantumRegister(nq_plus_1, 'q'); qc = QuantumCircuit(qr)
    state_vec = normalize(state_vec)
    qc.initialize(state_vec, qr)
    return qc.compose(create_transform(nq_plus_1).inverse())

# ===== DST/DCT ======
backend = Aer.get_backend('statevector_simulator')
qc_fwd  = qc_apply_U(f_pad, num_qubits+1)
psi_U   = np.array(execute(qc_fwd, backend=backend).result().get_statevector(qc_fwd))

C = np.real(psi_U[:N]) 
S = np.real((-1j) * psi_U[N:])

# ===== DIVISION ======

if BCs == 'neumann':
    k   = np.arange(N, dtype=float)
    lam = (np.pi*k)**2
    C_sol = np.zeros_like(C); C_sol[1:] = - C[1:] / lam[1:]
    S_sol = np.zeros_like(S); which='C'
else:
    k1   = np.arange(1, N+1, dtype=float)
    lamS = (np.pi*k1)**2
    S_sol = - S / lamS
    C_sol = np.zeros_like(C); which='S'

# ===== IDST/IDCT ======
spec = np.zeros(N2, dtype=complex)
spec[:N] = C_sol
spec[N:] = 1j * S_sol
spec = normalize(spec)

qc_inv   = qc_apply_U_inv(spec, num_qubits+1)
psi_back = np.array(execute(qc_inv, backend=backend).result().get_statevector(qc_inv))

# ===== EXTRACTION ======
if BCs == 'neumann':
    u_num    = np.real(psi_back[:N])  
else :
    u_num    = np.real(psi_back[N:])

# ===== FILTER ======
def bandpass_first_peak(x, dx, width=0.3, oversample=20, kmin=1, rel_floor=1e-6):
    """
    Band-pass autour du PREMIER pic non nul (fondamental), pas du plus fort.
    kmin=1 saute le DC; rel_floor évite des micro-pics de bruit.
    Retourne (signal_filtré, f0_estime).
    """
    L = oversample * len(x)
    X = fft(x - x.mean(), n=L)
    f = fftfreq(L, dx)
    pos = f > 0
    fpos = f[pos]
    Apos = np.abs(X[pos])

    if Apos.size == 0:
        return x.copy(), 0.0

    Amax = Apos.max()
    k0 = None
    start = max(kmin, 1)
    for k in range(start, len(Apos) - 1):
        if Apos[k] > Apos[k-1] and Apos[k] >= Apos[k+1] and Apos[k] >= rel_floor * Amax:
            k0 = k
            break
    if k0 is None:  # fallback
        idx = np.where(Apos >= rel_floor * Amax)[0]
        k0 = int(idx[0]) if idx.size else int(np.argmax(Apos))

    # interpolation quadratique (optionnelle)
    if 1 <= k0 < len(Apos) - 1:
        a, b, c = Apos[k0-1], Apos[k0], Apos[k0+1]
        denom = (a - 2*b + c)
        delta = 0.0 if denom == 0 else 0.5 * (a - c) / denom
        f0 = fpos[k0] + delta * (fpos[1] - fpos[0])
    else:
        f0 = fpos[k0]

    lo, hi = (1.0 - width) * f0, (1.0 + width) * f0
    M = (np.abs(f) >= lo) & (np.abs(f) <= hi)
    Y = np.zeros_like(X); Y[M] = X[M]
    y = np.real(ifft(Y))[:len(x)]
    return y, float(f0)

if APPLY_BANDPASS:
    dx = 1.0 / N
    u_num, f0 = bandpass_first_peak(u_num, dx, width=0.3, oversample=20, kmin=1)

# ===== NORMALISATION ======
nx = safe_norm(x_real)
nu = safe_norm(u_num)
if np.isfinite(nx) and np.isfinite(nu) and nu > 0:
    u_num *= nx / nu

#===== ERROR (relative %) =======
mask = (x_real != 0)
error = np.zeros_like(x_real)
error[mask] = 100.0 * np.abs((u_num[mask] - x_real[mask]) / x_real[mask])

# ===== PLOTTING =======
fig, ax1 = plt.subplots()
ax1.plot(x, x_real, '-',  label='Exact')
ax1.plot(x, u_num,  '--', label='DST/DCT')
ax1.set_xlabel('x')
ax1.set_ylabel('p(x)')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(x, error, '-', color='red', label='Error (%)')
ax2.set_ylabel('Error (%)')
ax2.set_yscale('log')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.abs(fft(u_num)),label='unum')
plt.plot(np.abs(fft(x_real)),label='xreal')
plt.legend()
plt.tight_layout()
plt.show()
