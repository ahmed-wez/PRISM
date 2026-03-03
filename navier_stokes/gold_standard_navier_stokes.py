import os
import numpy as np
import matplotlib.pyplot as plt
import dolfinx
from ns_solver import NavierStokesSolver2D
from tqdm import tqdm

def run_gold_standard_ns():
    print("\n" + "="*85)
    print("PHASE 6.3: NAVIER-STOKES GOLD STANDARD (UNIFIED PROTOCOL)")
    print("Protocol: Res=32 | 10k Samples | 10 Trials | Matched Tolerance")
    print("="*85)
    
    # --- STATED CONSTANTS ---
    N_list = [16, 36, 64, 81, 100, 144, 196, 225] 
    num_trials = 10
    theta_true = 1.0   # Boosting signal to overcome noise floor
    sigma_noise = 0.05 
    res = 32 
    
    solver = NavierStokesSolver2D(res=res, mu=0.5)
    u_true_mixed, _ = solver.solve(theta_true, tol=1e-11)
    
    ensemble_means = []
    ensemble_stds = []

    for N in N_list:
        print(f"\n>>> EVALUATING N = {N}")
        n_side = int(np.sqrt(N))
        grid_pts = np.linspace(0.15, 0.85, n_side)
        xv, yv = np.meshgrid(grid_pts, grid_pts)
        obs_points = np.zeros((N, 3))
        obs_points[:, 0] = xv.ravel(); obs_points[:, 1] = yv.ravel()
        
        cell_candidates = dolfinx.geometry.compute_collisions_points(solver.bb_tree, obs_points)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(solver.domain, cell_candidates, obs_points)
        cells = [colliding_cells.links(i)[0] for i in range(N)]
        
        y_clean = solver.get_fast_observations(u_true_mixed, obs_points, cells)
        trial_errors = []

        for trial in range(num_trials):
            y_obs = y_clean + np.random.normal(0, sigma_noise, y_clean.shape)
            
            # --- ADAPTIVE MCMC ENGINE ---
            current_theta = 0.8 # Start near truth
            u_curr, _ = solver.solve(current_theta)
            y_curr = solver.get_fast_observations(u_curr, obs_points, cells)
            current_lp = -0.5 * np.sum(((y_obs - y_curr)/sigma_noise)**2)
            
            samples = []
            accepted = 0
            scale = 0.1 / np.sqrt(N) # Correct initial scaling
            target_acc = 0.25
            
            pbar = tqdm(range(10000), desc=f"  N={N} Trial {trial+1}/10", leave=False)
            for i in pbar:
                prop = current_theta + np.random.normal(0, scale)
                if prop < 0.01 or prop > 2.0:
                    samples.append(current_theta); continue
                
                u_prop, _ = solver.solve(prop)
                y_prop = solver.get_fast_observations(u_prop, obs_points, cells)
                prop_lp = -0.5 * np.sum(((y_obs - y_prop)/sigma_noise)**2)
                
                if np.log(np.random.rand()) < (prop_lp - current_lp):
                    current_theta, current_lp = prop, prop_lp
                    accepted += 1
                
                samples.append(current_theta)

                # ROBBINS-MONRO ADAPTATION (Crucial for Non-Linear Stability)
                if (i + 1) % 100 == 0 and i < 5000:
                    acc_rate = accepted / (i + 1)
                    scale *= np.exp(acc_rate - target_acc)

            theta_map = np.mean(samples[5000:])
            err = abs(theta_map - theta_true)
            trial_errors.append(err)
            print(f"    Trial {trial+1} Error: {err:.6f} | Acc: {accepted/10000:.1%}")

        m_err = np.mean(trial_errors)
        ensemble_means.append(m_err)
        ensemble_stds.append(np.std(trial_errors))
        print(f"--- N={N} ENSEMBLE MEAN ERROR: {m_err:.6f}")

    log_N, log_E = np.log(N_list), np.log(ensemble_means)
    slope, _ = np.polyfit(log_N, log_E, 1)
    print(f"\nFINAL HARMONIZED NS RATE: N^({slope:.4f})")

    plt.figure(figsize=(10, 7))
    plt.errorbar(N_list, ensemble_means, yerr=ensemble_stds, fmt='o-', color='tab:blue')
    theory = ensemble_means[0]*(N_list[0]/np.array(N_list))**0.5
    plt.loglog(N_list, theory, '--', color='tab:red', label='Theory N^-0.5')
    plt.xscale('log'); plt.yscale('log'); plt.savefig('../figures/ns_gold_standard.png', dpi=300)

if __name__ == "__main__":
    run_gold_standard_ns()