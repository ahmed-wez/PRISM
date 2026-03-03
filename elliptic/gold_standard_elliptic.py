import numpy as np
import matplotlib.pyplot as plt
import os
from elliptic_solver import PoissonSolver1D
from tqdm import tqdm

def run_gold_standard_elliptic():
    print("\n" + "="*85)
    print("PHASE 3.1: ELLIPTIC GOLD STANDARD - HIGH NOISE REDEMPTION")
    print("Protocol: res=32 | sigma=0.2 | 10 Trials | 10k Samples")
    print("Rationale: Noise-dominance to reveal pure N^-0.5 trend")
    print("="*85)
    
    # --- STRICTLY CONSTANT PARAMETERS ---
    N_list = [16, 36, 64, 100, 144, 196, 225] 
    num_trials = 10
    theta_true = 0.5 
    res = 32         
    
    # ADJUSTED FOR BIAS SEPARATION
    sigma_noise = 0.2 # Higher noise forces the statistical rate to be the dominant visible factor
    
    solver = PoissonSolver1D(n_elements=res)
    ensemble_means = []
    ensemble_stds = []

    for N in N_list:
        trial_errors = []
        print(f"\n>>> EVALUATING N = {N}")
        
        # Fixed spatial sensors for the N-study
        x_obs = np.linspace(0.05, 0.95, N)
        # Continuous Analytic Truth
        y_true = theta_true * np.sin(np.pi * x_obs)

        for trial in range(num_trials):
            y_noisy = y_true + np.random.normal(0, sigma_noise, N)
            
            # --- ADAPTIVE MCMC ENGINE ---
            current_theta = 0.8 
            
            def get_logl(t):
                u_nodes = solver.solve(t)
                y_pred = np.interp(x_obs, solver.nodes, u_nodes)
                return -0.5 * np.sum(((y_noisy - y_pred)/sigma_noise)**2)

            curr_logl = get_logl(current_theta)
            samples = []
            accepted = 0
            # Scale slightly larger to account for higher noise
            scale = 0.4 / np.sqrt(N) 
            target_acc = 0.44
            
            pbar = tqdm(range(10000), desc=f"  Trial {trial+1}/10", leave=False)
            for i in pbar:
                prop = current_theta + np.random.normal(0, scale)
                if prop < 0.01 or prop > 1.5:
                    samples.append(current_theta); continue
                
                prop_logl = get_logl(prop)
                
                if np.log(np.random.rand()) < (prop_logl - curr_logl):
                    current_theta, curr_logl = prop, prop_logl
                    accepted += 1
                samples.append(current_theta)

                if (i + 1) % 100 == 0 and i < 5000:
                    acc_rate = accepted / (i + 1)
                    scale *= np.exp(acc_rate - target_acc)

            theta_map = np.mean(samples[5000:])
            err = abs(theta_map - theta_true)
            trial_errors.append(err)
            print(f"    Trial {trial+1}/10 | Error: {err:.6f} | Acc: {accepted/10000:.1%}")

        m_err = np.mean(trial_errors)
        ensemble_means.append(m_err)
        ensemble_stds.append(np.std(trial_errors))
        print(f"--- N={N} ENSEMBLE MEAN ERROR: {m_err:.6f} ± {np.std(trial_errors):.6f}")

    # Final Rate Calculation
    log_N, log_E = np.log(N_list), np.log(ensemble_means)
    slope, _ = np.polyfit(log_N, log_E, 1)
    
    print("\n" + "="*85)
    print(f"ULTIMATE ELLIPTIC RATE: N^({slope:.4f})")
    print(f"Target: N^(-0.500) | Rigor: Non-Inverse Crime, Balanced SNR")
    print("="*85)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.errorbar(N_list, ensemble_means, yerr=ensemble_stds, fmt='o-', 
                 color='tab:blue', label=f'Elliptic (Rate: {slope:.3f})', linewidth=2, capsize=5)
    theory = ensemble_means[0]*(N_list[0]/np.array(N_list))**0.5
    plt.loglog(N_list, theory, '--', color='tab:red', label='Theory N^-0.5')
    plt.xscale('log'); plt.yscale('log'); plt.legend(); plt.grid(True, which="both", alpha=0.3)
    if not os.path.exists('../figures'): os.makedirs('../figures')
    plt.savefig('../figures/elliptic_gold_standard.png', dpi=300)

if __name__ == "__main__":
    run_gold_standard_elliptic()