from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.circuit.library import QFT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

# ===== PARAMETERS ======
num_qubit_x = 7
num_qubit_y = 7
Nx = 2**num_qubit_x  
Ny = 2**num_qubit_y 
x_sup = 1
x_inf = 0
y_sup = 1
y_inf = 0
dx = (x_sup - x_inf) / (Nx - 1)
dy = (y_sup-y_inf) / (Ny - 1)
x = np.linspace(x_inf, x_sup, Nx) 
y = np.linspace(y_inf, y_sup, Ny) 
X, Y = np.meshgrid(x, y, indexing='ij')  

freqx = np.fft.fftfreq(Nx, d=dx)
freqy = np.fft.fftfreq(Ny, d=dy)

# ===== FUNTION ======
f_raw = -8*np.pi**2 * np.cos(2*np.pi * X)*np.cos(2*np.pi * Y)
norm_factor = np.linalg.norm(f_raw.ravel())
f = f_raw.ravel() / norm_factor             

# ===== QUANTUM CIRCUIT QFT ======
qrx = QuantumRegister(num_qubit_x, name='x')
qry = QuantumRegister(num_qubit_y, name='y')
qc = QuantumCircuit(qrx,qry)

qc.initialize(f,[*qrx,*qry])
qc.barrier()

qft_circ_x = QFT(num_qubits=num_qubit_x,do_swaps=True)
qft_circ_y = QFT(num_qubits=num_qubit_y,do_swaps=True)
qc.append(qft_circ_x.to_gate(), qrx)
qc.append(qft_circ_y.to_gate(), qry)
qc.barrier()

backend = Aer.get_backend('statevector_simulator')
job = execute(qc, backend=backend)
result = job.result()
f_transform = result.get_statevector(qc)

f_transform_2 = f_transform *  norm_factor
f_transform_2 = f_transform_2.reshape((Nx,Ny))

# ===== DIVISION ======
p_trans = np.zeros_like(f_transform_2, dtype=complex)

omegax = 2 * np.pi * freqx
omegay = 2 * np.pi * freqy

for i in range(len(omegax)):
    for j in range(len(omegay)):
        denom = (omegax[i]**2+omegay[j]**2)
        if denom != 0 :
            p_trans[i,j] = -f_transform_2[i,j] / (denom)
        else:
            p_trans[i,j] = 0.0

p_trans = p_trans / np.linalg.norm(p_trans)
amps = np.asarray(p_trans, dtype=complex).ravel()
amps /= np.linalg.norm(amps)

# Convertir en list Python pour éviter CircuitError
amps_list = amps.tolist()

# ===== QUANTUM CRCUIT IQFT ======
qrx = QuantumRegister(num_qubit_x, name='x')
qry = QuantumRegister(num_qubit_y, name='y')
qc_iqft = QuantumCircuit(qrx,qry)

qc_iqft.initialize(amps_list,[*qrx,*qry])
qc_iqft.barrier()

iqft_circ_x = QFT(num_qubits=num_qubit_x,inverse=True,do_swaps=True)
iqft_circ_y = QFT(num_qubits=num_qubit_y,inverse=True,do_swaps=True)
qc_iqft.append(iqft_circ_x.to_gate(), qrx)
qc_iqft.append(iqft_circ_y.to_gate(), qry)
qc_iqft.barrier()

backend = Aer.get_backend('statevector_simulator')
job = execute(qc_iqft, backend=backend)
result = job.result()
p = result.get_statevector(qc_iqft)

p = p * np.linalg.norm(p_trans) * Nx
p = p.reshape(Nx, Ny)

# ===== EXACT SOLUTION ======
#p_real = np.sin(2*np.pi*X)*np.sin(2*np.pi*Y) #Dirichlet 
p_real = np.cos(2*np.pi * X)*np.cos(2*np.pi * Y) #Neumann

# ===== ERROR ======
error = np.zeros_like(p_real)
mask = p_real != 0
error[mask] = 100 * np.abs((p_real[mask] - p.real[mask]) / p_real[mask])

# ===== PLOTTING ======
fig = plt.figure(figsize=(14, 6))

# --- QFT solution (3D) ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, p.real, cmap='viridis')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('p(x,y)')
ax1.set_title('QFT solution (3D)')

# --- Exact solution (3D) ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, p_real, cmap='plasma')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('p_real(x,y)')
ax2.set_title('Exact solution (3D)')

plt.tight_layout()
plt.show()

# ===== ERROR PLOT ======
error_plot = error.copy()
error_plot[error_plot <= 0] = 1e-12

vmax = np.percentile(error_plot, 95)

fig3 = plt.figure(figsize=(8,6))
im = plt.imshow(error_plot,
                origin='lower',
                extent=(x_inf, x_sup, y_inf, y_sup),
                aspect='auto',
                cmap='inferno',
                norm=LogNorm(vmin=1e-6, vmax=vmax))
plt.colorbar(im, label='Relative error (%) [log scale]')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Relative error (log scale)')
plt.tight_layout()
plt.show()