import numpy as np
import matplotlib.pyplot as plt
import os
import dolfinx
from stokes_solver import StokesSolver2D
from tqdm import tqdm

def run_gold_standard_stokes():
    print("\n" + "="*85)
    print("PHASE 4.1: STOKES HARMONIZED GOLD STANDARD")
    print("Protocol: Res=32 | 10k Samples | 10 Trials | Metric: MAP Error")
    print("="*85)
    
    # 1. PARAMETERS (STRICTLY HARMONIZED)
    N_list = [16, 36, 64, 81, 100, 144, 196, 225] 
    num_trials = 10
    theta_true = 0.8   
    sigma_noise = 0.05 
    res = 32         
    
    solver = StokesSolver2D(res=res)
    # Generate ground truth at ultra-high precision
    u_true_standalone = solver.solve(theta_true, tol=1e-12)

    ensemble_means = []
    ensemble_stds = []

    for N in N_list:
        print(f"\n>>> EVALUATING N = {N}")
        
        n_side = int(np.sqrt(N))
        grid_pts = np.linspace(0.15, 0.85, n_side)
        xv, yv = np.meshgrid(grid_pts, grid_pts)
        obs_points = np.zeros((N, 3), dtype=np.float64)
        obs_points[:, 0] = xv.ravel(); obs_points[:, 1] = yv.ravel()
        
        cell_candidates = dolfinx.geometry.compute_collisions_points(solver.bb_tree, obs_points)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(solver.domain, cell_candidates, obs_points)
        cells = [colliding_cells.links(i)[0] for i in range(N)]
        
        y_clean = solver.get_fast_observations(u_true_standalone, obs_points, cells)
        trial_errors = []

        for trial in range(num_trials):
            y_noisy = y_clean + np.random.normal(0, sigma_noise, len(y_clean))
            
            # --- ADAPTIVE MCMC ENGINE ---
            current_theta = 0.5 
            u_curr = solver.solve(current_theta, tol=1e-11)
            y_curr = solver.get_fast_observations(u_curr, obs_points, cells)
            current_lp = -0.5 * np.sum(((y_noisy - y_curr)/sigma_noise)**2)
            
            samples = []
            accepted_count = 0
            scale = 0.1 / np.sqrt(N)
            target_acc = 0.25
            
            pbar = tqdm(range(10000), desc=f"  N={N} Trial {trial+1}/10", leave=False)
            for i in pbar:
                prop = current_theta + np.random.normal(0, scale)
                if prop < 0.01 or prop > 5.0:
                    samples.append(current_theta); continue
                
                u_prop = solver.solve(prop, tol=1e-11)
                y_prop = solver.get_fast_observations(u_prop, obs_points, cells)
                prop_lp = -0.5 * np.sum(((y_noisy - y_prop)/sigma_noise)**2)
                
                if np.log(np.random.rand()) < (prop_lp - current_lp):
                    current_theta, current_lp = prop, prop_lp
                    accepted_count += 1
                samples.append(current_theta)

                if (i + 1) % 50 == 0 and i < 5000:
                    recent_acc = accepted_count / (i + 1)
                    scale *= np.exp(recent_acc - target_acc)

            # MAP Error Calculation
            theta_map = np.mean(samples[5000:])
            err = abs(theta_map - theta_true)
            trial_errors.append(err)
            print(f"    Trial {trial+1}/10 | Error: {err:.6f} | Acc: {accepted_count/10000:.1%}")

        m_err = np.mean(trial_errors)
        s_err = np.std(trial_errors)
        ensemble_means.append(m_err)
        ensemble_stds.append(s_err)
        print(f"--- N={N} ENSEMBLE MEAN ERROR: {m_err:.6f} ± {s_err:.6f}")

    # Final Rate Calculation
    log_N, log_E = np.log(N_list), np.log(ensemble_means)
    slope, _ = np.polyfit(log_N, log_E, 1)
    
    print("\n" + "="*85)
    print(f"ULTIMATE HARMONIZED STOKES RATE: N^({slope:.4f})")
    print(f"Target: N^(-0.5000)")
    print("="*85)

    plt.figure(figsize=(10, 7))
    plt.errorbar(N_list, ensemble_means, yerr=ensemble_stds, fmt='o-', label=f'Stokes (Rate: {slope:.3f})')
    theory = ensemble_means[0] * (N_list[0]/np.array(N_list))**0.5
    plt.loglog(N_list, theory, '--', color='tab:red', label='Theory N^-0.5')
    plt.xscale('log'); plt.yscale('log'); plt.legend(); plt.grid(True, which="both", alpha=0.3)
    if not os.path.exists('../figures'): os.makedirs('../figures')
    plt.savefig('../figures/stokes_gold_standard.png', dpi=300)

if __name__ == "__main__":
    run_gold_standard_stokes()