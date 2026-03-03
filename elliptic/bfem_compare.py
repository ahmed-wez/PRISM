import numpy as np
import matplotlib.pyplot as plt
import os
from elliptic_solver import BFEMSolver1D
from bayesian_inference import BayesianPoissonInference

def run_bfem_comparison():
    print("\n" + "="*60)
    print("ELLIPTIC: BFEM VS STANDARD FEM CALIBRATION")
    print("="*60)
    
    h_list = [1/4, 1/8, 1/16, 1/32]
    theta_true = 1.0
    sigma_obs = 0.001 # Small noise to make discretization bias visible
    x_obs = np.array([0.2, 0.5, 0.8])
    y_true = np.sin(np.pi * x_obs) * theta_true
    
    std_coverage = []
    bfem_coverage = []
    
    for h in h_list:
        solver = BFEMSolver1D(int(1/h))
        std_hits = 0
        bfem_hits = 0
        trials = 200 # More trials for smooth coverage curve
        
        for _ in range(trials):
            y = y_true + np.random.normal(0, sigma_obs, len(x_obs))
            
            # 1. Standard (Deterministic)
            inf = BayesianPoissonInference(x_obs, sigma_obs)
            m, s = inf.gaussian_posterior(y)
            if abs(m - theta_true) < 2*s: std_hits += 1
            
            # 2. BFEM (Probabilistic)
            sigma_total = np.sqrt(sigma_obs**2 + solver.sigma_pn**2)
            inf_pn = BayesianPoissonInference(x_obs, sigma_total)
            m_p, s_p = inf_pn.gaussian_posterior(y)
            if abs(m_p - theta_true) < 2*s_p: bfem_hits += 1
            
        std_coverage.append(std_hits/trials)
        bfem_coverage.append(bfem_hits/trials)
        print(f"h={h:.3f} | Std: {std_hits/trials:.1%} | BFEM: {bfem_hits/trials:.1%}")

    plt.figure(figsize=(8, 6))
    plt.semilogx(h_list, std_coverage, 'r-o', label='Standard FEM')
    plt.semilogx(h_list, bfem_coverage, 'g-s', label='BFEM (PN)')
    plt.axhline(0.95, color='k', linestyle='--', label='95% Target')
    plt.xlabel('Discretization h')
    plt.ylabel('Coverage')
    plt.title('Reliability Comparison: Elliptic PDE')
    plt.legend()
    plt.savefig('../figures/elliptic_bfem_compare.png', dpi=300)

if __name__ == "__main__":
    run_bfem_comparison()