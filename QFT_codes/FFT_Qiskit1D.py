from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit.library import QFT
import numpy as np
import matplotlib.pyplot as plt

# ===== PARAMETERS =====
num_qubits = 4
N = 2**num_qubits       
x_sup = 1
x_inf = 0
dx = (x_sup - x_inf) / (N - 1)
x = np.linspace(x_inf, x_sup, N)
freq = np.fft.fftfreq(N, d=dx)

# ===== FUNCTION =====
f_x_raw = 10*x
norm_factor = np.linalg.norm(f_x_raw)
f_x = f_x_raw / norm_factor             

# ===== QUANTUM CIRCUIT QFT =====
qr = QuantumRegister(num_qubits, name='x')
qc = QuantumCircuit(qr)
qc.initialize(f_x, qr)
qc.append(QFT(num_qubits=num_qubits, do_swaps=True).to_gate(), qr)

backend = Aer.get_backend('statevector_simulator')
f_x_transform = execute(qc, backend=backend).result().get_statevector(qc) 

# ===== NORMALISATION =====
f_x_transform_2 = f_x_transform * norm_factor

# ===== DIVISION =====
p_trans = np.zeros_like(f_x_transform_2, dtype=complex)
omega = 2 * np.pi * freq
nonzero = omega != 0

for i in range(len(omega)):
    if omega[i] != 0:
        p_trans[i] = - f_x_transform_2[i] / (omega[i] ** 2)
    else:
        p_trans[i] = 0.0

# ===== QUANTUM CIRCUIT IQFT =====
qr2 = QuantumRegister(num_qubits, name='x')
qc_iqft = QuantumCircuit(qr2)
qc_iqft.initialize(p_trans / np.linalg.norm(p_trans), qr2)   # normalize to satisfy initialize
qc_iqft.append(QFT(num_qubits=num_qubits, inverse=True, do_swaps=True).to_gate(), qr2)

backend = Aer.get_backend('statevector_simulator')
job = execute(qc_iqft, backend=backend)

result = job.result()

p_x_transform = result.get_statevector(qc_iqft)

p_x_transform_2 = p_x_transform  * np.linalg.norm(p_trans)
 
#===== EXACT SOL ======
x_real = 5/3*(x**3-x)

#===== ERROR =======
mask = x_real != 0
error = np.zeros_like(x_real)
error[mask] = 100 * np.abs((p_x_transform_2.real[mask] - x_real[mask]) / x_real[mask])

# ===== PLOTTING =======
fig, ax1 = plt.subplots()

ax1.plot(x, p_x_transform_2.real, '--', label='QFT')
ax1.plot(x, x_real, '--', label='Exact')
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


plt.show()
