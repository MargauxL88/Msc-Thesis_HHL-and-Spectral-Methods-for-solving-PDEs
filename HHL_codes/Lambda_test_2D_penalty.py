import numpy as np
import matplotlib.pyplot as plt

def build_neumann_system(N=21, x_inf=0.0, x_sup=1.0, y_inf=0.0, y_sup=1.0):
    # Grille
    dx = (x_sup - x_inf) / N
    dy = (y_sup - y_inf) / N
    x = np.linspace(x_inf, x_sup, N+1)
    y = np.linspace(y_inf, y_sup, N+1)
    X, Y = np.meshgrid(x, y, indexing='ij')

    N2 = (N-1)*(N-1)
    A = np.zeros((N2, N2))
    b = np.zeros(N2)

    for i in range(N-1):
        for j in range(N-1):
            idx = i + (N-1)*j
            is_corner = (i in (0, N-2)) and (j in (0, N-2))
            is_edge   = ((i in (0, N-2)) or (j in (0, N-2))) and not is_corner
            A[idx, idx] = -2.0 if is_corner else (-3.0 if is_edge else -4.0)
            if i > 0:       A[idx, idx-1]         = 1.0
            if i < N-2:     A[idx, idx+1]         = 1.0
            if j > 0:       A[idx, idx-(N-1)]     = 1.0
            if j < N-2:     A[idx, idx+(N-1)]     = 1.0

            ii, jj = i+1, j+1
            f_ij = np.cos(np.pi*X[ii, jj]) * np.cos(np.pi*Y[ii, jj])
            b[idx] = (dx*dy) * f_ij if abs(dx-dy) > 1e-15 else (dx**2) * f_ij  # dx=dy => dx^2

    w = (dx*dy) * np.ones((N2, 1))
    return A, b.reshape(-1,1), w, (X, Y, dx, dy)

# ===== LAGRANGE REF ======
def solve_with_lagrange(A, b, w):
    N2 = A.shape[0]
    A_aug = np.block([[A, w],
                      [w.T, np.zeros((1,1))]])
    rhs   = np.vstack([b, [[0.0]]])
    sol   = np.linalg.solve(A_aug, rhs)
    x_ref = sol[:N2, :]
    lam_ref = float(sol[-1, 0])
    return x_ref, lam_ref

# ===== PENALTY MEHTOD =====
def solve_with_penalty(A, b, w, lambdas):
    vals_lam = []
    vals_ct  = []  
    vals_lamc= []    
    errs_rel = []
    conds    = []

    x_ref, lam_ref = solve_with_lagrange(A, b, w)
    nrm_ref = np.linalg.norm(x_ref)

    wwt = (w @ w.T)  
    for lam in lambdas:
        M = A + lam * wwt
        try:
            conds.append(np.linalg.cond(M))
        except np.linalg.LinAlgError:
            conds.append(np.inf)

        x_lam = np.linalg.solve(M, b)
        c = float(w.T @ x_lam)     
        vals_lam.append(lam)
        vals_ct.append(c)
        vals_lamc.append(lam * c)
        err = np.linalg.norm(x_lam - x_ref) / (nrm_ref + 1e-30)
        errs_rel.append(err)

    return (np.array(vals_lam),
            np.array(vals_ct),
            np.array(vals_lamc),
            np.array(errs_rel),
            np.array(conds),
            x_ref, lam_ref)

# ================== MAIN ==================
if __name__ == "__main__":

    N = 21  
    A, b, w, (X, Y, dx, dy) = build_neumann_system(N=N)

    lambdas = np.logspace(-6, 6, 25)
    (vals_lam, vals_ct, vals_lamc, errs_rel, conds, x_ref, lam_ref) = \
        solve_with_penalty(A, b, w, lambdas)

    # ===== PLOTTING ======
    plt.figure(figsize=(6,4.5))
    plt.loglog(vals_lam, np.abs(vals_ct), marker='o')
    plt.xlabel(r'$\lambda$ (penalty)')
    plt.ylabel(r'$|e^T\top p|$')
    plt.title('Convergence de la contrainte (moyenne)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4.5))
    plt.loglog(vals_lam, conds, marker='d')
    plt.xlabel(r'$\lambda$ (penalty)')
    plt.ylabel(r'$\kappa$($(A+\lambda ee^\top)$)')
    plt.title('Conditionnement vs pénalité')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
