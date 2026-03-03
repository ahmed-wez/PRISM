# reaction_diffusion/gold_standard_reaction_diffusion.py
import os
import numpy as np
import matplotlib.pyplot as plt
import dolfinx
from rd_solver import CoupledRDSolver2D
from bayesian_rd_inference import BayesianRDInference
from tqdm import tqdm

def run_gold_standard_rd():
    print("\n" + "="*85)
    print("PHASE 7.3: REACTION-DIFFUSION GOLD STANDARD (UNIFIED)")
    print("Strategy: 10 Trials | 10k Samples | Pre-mapped Points")
    print("="*85)
    
    N_list = [16, 36, 64, 81, 100, 144, 196, 225] 
    num_trials = 10
    theta_true = np.array([0.3, 0.2]) 
    sigma_noise = 0.001 
    res = 32 
    
    solver = CoupledRDSolver2D(res=res, k=0.5)
    print("Step 1: Pre-calculating high-precision ground truth...")
    u_v_true, _ = solver.solve(theta_true[0], theta_true[1], tol=1e-11)
    
    ensemble_means = []
    ensemble_stds = []

    for N in N_list:
        print(f"\n>>> EVALUATING N = {N}")
        n_side = int(np.sqrt(N))
        grid_pts = np.linspace(0.15, 0.85, n_side)
        xv, yv = np.meshgrid(grid_pts, grid_pts)
        obs_points = np.zeros((N, 3))
        obs_points[:, 0] = xv.ravel(); obs_points[:, 1] = yv.ravel()
        
        # Pre-find cell indices once per N
        cell_candidates = dolfinx.geometry.compute_collisions_points(solver.bb_tree, obs_points)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(solver.domain, cell_candidates, obs_points)
        cells = [colliding_cells.links(i)[0] for i in range(N)]
        
        y_clean = solver.get_fast_observations(u_v_true, obs_points, cells)
        trial_errors = []

        for trial in range(num_trials):
            y_obs = y_clean + np.random.normal(0, sigma_noise, len(y_clean))
            inference = BayesianRDInference(solver, obs_points, sigma_noise)
            
            samples = inference.run_adaptive_mcmc(y_obs, cells, n_samples=10000)
            
            # Use last 5000 samples for MAP
            theta_map = np.mean(samples[5000:], axis=0)
            err = np.linalg.norm(theta_map - theta_true)
            trial_errors.append(err)
            print(f"    Trial {trial+1}/10 | Error: {err:.6f}")

        m_err = np.mean(trial_errors)
        ensemble_means.append(m_err)
        ensemble_stds.append(np.std(trial_errors))
        print(f"--- N={N} ENSEMBLE MEAN ERROR: {m_err:.6f} ± {np.std(trial_errors):.6f}")

    log_N, log_E = np.log(N_list), np.log(ensemble_means)
    slope, _ = np.polyfit(log_N, log_E, 1)
    
    print("\n" + "="*85)
    print(f"ULTIMATE HARMONIZED RD RATE: N^({slope:.4f})")
    print(f"Target: N^(-0.5000)")
    print("="*85)

    plt.figure(figsize=(10, 7))
    plt.errorbar(N_list, ensemble_means, yerr=ensemble_stds, fmt='o-', color='tab:green', label=f'RD (Rate: {slope:.3f})')
    theory = ensemble_means[0]*(N_list[0]/np.array(N_list))**0.5
    plt.loglog(N_list, theory, '--', color='tab:red', label='Theory N^-0.5')
    plt.xscale('log'); plt.yscale('log'); plt.legend(); plt.grid(True, which="both", alpha=0.3)
    if not os.path.exists('../figures'): os.makedirs('../figures')
    plt.savefig('../figures/rd_gold_standard.png', dpi=300)

if __name__ == "__main__":
    run_gold_standard_rd()