import numpy as np
import matplotlib.pyplot as plt
from subdiffusion_solver import BFEMSubdiffusion1D
from bayesian_inference import BayesianSubdiffusionInference

def run_bfem_subdiffusion():
    h = 1/8
    solver = BFEMSubdiffusion1D(nx=int(1/h), nt=int(1/h))
    q_true, sigma_obs = 1.5, 0.001
    x_obs = np.linspace(0.2, 0.8, 3)
    
    # High-res truth
    u_true_full = SubdiffusionSolver1D(nx=100, nt=100).solve(q_true)
    y_true = np.interp(x_obs, np.linspace(0,1,100), u_true_full)
    
    hits_std, hits_pn = 0, 0
    for _ in range(100):
        y = y_true + np.random.normal(0, sigma_obs, len(x_obs))
        
        # Standard
        inf = BayesianSubdiffusionInference(solver, x_obs, sigma_obs)
        samples = inf.run_adaptive_mcmc(y, n_samples=2000)
        if abs(np.mean(samples[1000:]) - q_true) < 2*np.std(samples[1000:]): hits_std += 1
        
        # PN
        sigma_total = np.sqrt(sigma_obs**2 + solver.sigma_pn**2)
        inf_pn = BayesianSubdiffusionInference(solver, x_obs, sigma_total)
        samples_pn = inf_pn.run_adaptive_mcmc(y, n_samples=2000)
        if abs(np.mean(samples_pn[1000:]) - q_true) < 2*np.std(samples_pn[1000:]): hits_pn += 1
        
    print(f"Standard Coverage: {hits_std}%")
    print(f"BFEM (PN) Coverage: {hits_pn}%")

if __name__ == "__main__": run_bfem_subdiffusion()