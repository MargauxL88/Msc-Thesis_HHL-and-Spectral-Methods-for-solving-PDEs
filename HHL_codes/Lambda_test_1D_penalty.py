import numpy as np
import matplotlib.pyplot as plt

# ====== Construction du système 1D (Neumann homogène) ======
def build_neumann_1d(N_pts=18, x_inf=0.0, x_sup=1.0):

    dx = (x_sup - x_inf) / (N_pts - 1)
    x = np.linspace(x_inf, x_sup, N_pts)

    M = N_pts - 2          
    A = np.zeros((M, M))
    b = np.zeros((M, 1))

    for k in range(M):
        if k == 0:
            A[k, k] = -1.0
            if M > 1:
                A[k, k+1] = 1.0
        elif k == M-1:
            A[k, k] = -1.0
            A[k, k-1] = 1.0
        else:
            A[k, k] = -2.0
            A[k, k-1] = 1.0
            A[k, k+1] = 1.0

    for i in range(1, N_pts-1):
        b[i-1, 0] = np.cos(np.pi * x[i]) * (dx**2)

    w = (dx) * np.ones((M, 1))   
    e = np.ones((M, 1))          
    return A, b, w, e, x, dx

# ====== LAGRANGE REF ======
def solve_lagrange_1d(A, b, w):
    M = A.shape[0]
    A_aug = np.block([[A, w],
                      [w.T, np.zeros((1,1))]])
    rhs   = np.vstack([b, [[0.0]]])
    sol   = np.linalg.solve(A_aug, rhs)
    x_ref = sol[:M, :]
    lam_w = float(sol[-1, 0])    
    return x_ref, lam_w

# ====== PENALTY ======
def sweep_penalty_1d(A, b, w, e, lambdas):
    x_ref, lam_ref = solve_lagrange_1d(A, b, w)
    nrm_ref = np.linalg.norm(x_ref)

    wwt = w @ w.T
    dx = float(w[0,0])

    rows = []
    cw_list, ce_list = [], []
    lcw_list, lce_list = [], []
    errs_list, cond_list = [], []

    for lam in lambdas:
        Mmat = A + lam * wwt
        try:
            condM = np.linalg.cond(Mmat)
        except np.linalg.LinAlgError:
            condM = np.inf

        x_lam = np.linalg.solve(Mmat, b)
        c_w = float(w.T @ x_lam)   
        c_e = float(e.T @ x_lam)   

        err = np.linalg.norm(x_lam - x_ref) / (nrm_ref + 1e-30)

        rows.append((lam, abs(c_w), abs(c_e), abs(lam*c_w), abs(lam*c_e), err, condM))
        cw_list.append(c_w); ce_list.append(c_e)
        lcw_list.append(lam * c_w); lce_list.append(lam * c_e)
        errs_list.append(err); cond_list.append(condM)

    return (np.array(lambdas),
            np.array(cw_list), np.array(ce_list),
            np.array(lcw_list), np.array(lce_list),
            np.array(errs_list), np.array(cond_list),
            x_ref, lam_ref, dx, rows)

# ========================== MAIN ==========================
if __name__ == "__main__":
    N_pts = 18      
    x_inf, x_sup = 0.0, 1.0

    #==== 1D SYSTEM ======
    A, b, w, e, x, dx = build_neumann_1d(N_pts=N_pts, x_inf=x_inf, x_sup=x_sup)

    lambdas = np.logspace(-6, 6, 25)
    (vals_lam,
     c_w, c_e,
     lc_w, lc_e,
     errs_rel, conds,
     x_ref, lam_ref, dx, table_rows) = sweep_penalty_1d(A, b, w, e, lambdas)

    ref_w = abs(lam_ref)
    ref_e = abs(lam_ref) / dx

    plt.figure(figsize=(6,4.6))
    plt.loglog(vals_lam, np.abs(c_e), 's--', label=r'$|e^\top p|$')
    plt.xlabel(r'$\lambda$ (penalty)'); plt.ylabel('Magnitude')
    plt.title('Contrainte (moyenne) vs pénalité — 1D')
    plt.grid(True, which='both', ls='--', alpha=0.5); plt.legend()
    plt.tight_layout(); plt.show()


    plt.figure(figsize=(6,4.6))
    plt.loglog(vals_lam, conds, marker='d', color='tab:orange')
    plt.xlabel(r'$\lambda$ (penalty)'); plt.ylabel(r'cond$(A+\lambda\,ww^\top)$')
    plt.title('Conditionnement vs pénalité — 1D')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()
