# gold_standard_subdiffusion.py
import numpy as np
import matplotlib.pyplot as plt
import os
from subdiffusion_solver import SubdiffusionSolver1D
from bayesian_inference import BayesianSubdiffusionInference
from tqdm import tqdm

def run_gold_standard_subdiffusion():
    print("\n" + "="*85)
    print("PHASE 5.3: SUBDIFFUSION ULTIMATE GOLD STANDARD (HARMONIZED)")
    print("Protocol: Res=32 | 10k Samples | 10 Trials | Full N-Range")
    print("="*85)
    
    # YOUR SPECIFIC HARMONIZED N_LIST
    N_list = [16, 36, 64, 81, 100, 144, 196, 225]
    num_trials = 10
    q_true = 1.5
    sigma_noise = 0.01 # Harmonized with Elliptic/NS
    res = 32
    
    # 1. MATCHED PHYSICS: Eliminate the systematic bias floor
    solver = SubdiffusionSolver1D(nx=res, nt=res)
    u_true = solver.solve(q_true)
    
    ensemble_means = []
    ensemble_stds = []

    for N in N_list:
        trial_errors = []
        print(f"\n>>> EVALUATING N = {N}")
        
        # Sensor placement
        x_obs = np.linspace(0.1, 0.9, N)
        y_clean = np.interp(x_obs, solver.x, u_true)

        # trial progress bar
        for trial in tqdm(range(num_trials), desc=f"Trials for N={N}"):
            y_obs = y_clean + np.random.normal(0, sigma_noise, N)
            inference = BayesianSubdiffusionInference(solver, x_obs, sigma_noise)
            
            # CONSISTENT 10,000 SAMPLES
            samples, acc = inference.run_adaptive_mcmc(y_obs, n_samples=10000)
            
            # MAP estimate (discard 50% burn-in)
            q_map = np.mean(samples[5000:])
            err = abs(q_map - q_true)
            trial_errors.append(err)
            
        m_err = np.mean(trial_errors)
        s_err = np.std(trial_errors)
        ensemble_means.append(m_err)
        ensemble_stds.append(s_err)
        print(f"--- N={N} ENSEMBLE MEAN ERROR: {m_err:.6f} ± {s_err:.6f}")

    # Final Rate Fit
    log_N, log_E = np.log(N_list), np.log(ensemble_means)
    slope, _ = np.polyfit(log_N, log_E, 1)
    
    print("\n" + "="*85)
    print(f"ULTIMATE HARMONIZED SUBDIFFUSION RATE: N^({slope:.4f})")
    print(f"Target: N^(-0.500) | Rigor: Ensemble Averaged")
    print("="*85)

    plt.figure(figsize=(10, 7))
    plt.errorbar(N_list, ensemble_means, yerr=ensemble_stds, fmt='o-', 
                 color='tab:orange', label=f'Subdiffusion (Rate: {slope:.3f})', linewidth=2, capsize=5)
    
    theory = ensemble_means[0]*(N_list[0]/np.array(N_list))**0.5
    plt.loglog(N_list, theory, '--', color='red', label='Theory N^-0.5')
    
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('N (Observations)'); plt.ylabel('Mean MAP Error')
    plt.title('Subdiffusion: Final Project Validation')
    plt.legend(); plt.grid(True, which="both", alpha=0.3)
    
    if not os.path.exists('../figures'): os.makedirs('../figures')
    plt.savefig('../figures/subdiffusion_gold_standard.png', dpi=300)

if __name__ == "__main__":
    run_gold_standard_subdiffusion()