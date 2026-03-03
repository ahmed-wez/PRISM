# subdiffusion_solver.py
import numpy as np
from scipy.special import gamma
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class SubdiffusionSolver1D:
    def __init__(self, nx=32, nt=32, T=1.0, beta=0.5):
        self.nx, self.nt, self.beta = nx, nt, beta
        self.x = np.linspace(0, 1, nx)
        self.h = 1.0 / (nx - 1)
        self.tau = T / (nt - 1)
        # L1 Scheme weights
        self.b = np.array([(j+1)**(1-beta) - j**(1-beta) for j in range(nt)])
        self.coeff = (self.tau**(-beta)) / gamma(2 - beta)

    def solve(self, q_val):
        # Initial condition: u(x,0) = sin(pi*x)
        u = np.zeros((self.nt, self.nx))
        u[0, :] = np.sin(np.pi * self.x)
        
        # Matrix A = -Δ + qI
        main_diag = (2.0 / self.h**2 + q_val) * np.ones(self.nx)
        off_diag = (-1.0 / self.h**2) * np.ones(self.nx - 1)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tolil()
        
        # Dirichlet BCs (u=0)
        A[0, :], A[-1, :] = 0, 0
        A[0, 0], A[-1, -1] = 1, 1
        A = A.tocsr()
        
        # LHS = coeff * b[0] * I + A
        LHS = diags([self.coeff * self.b[0]], [0], shape=(self.nx, self.nx)).tocsr() + A
        LHS = LHS.tolil()
        LHS[0, :], LHS[-1, :] = 0, 0
        LHS[0, 0], LHS[-1, -1] = 1, 1
        LHS = LHS.tocsr()

        for n in range(1, self.nt):
            # L1 history term
            # Weighting the previous time steps
            history = self.b[0] * u[n-1, :]
            for j in range(1, n):
                # Correct sum for the L1-scheme history
                history += (self.b[j] * u[n-j-1, :] - self.b[j] * u[n-j, :])
            
            rhs = self.coeff * history 
            rhs[0], rhs[-1] = 0, 0 
            u[n, :] = spsolve(LHS, rhs)
            
        return u[-1, :] # Terminal time