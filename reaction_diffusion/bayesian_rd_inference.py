# reaction_diffusion/bayesian_rd_inference.py
import numpy as np
from tqdm import tqdm

class BayesianRDInference:
    def __init__(self, solver, obs_points, sigma_obs):
        self.solver = solver
        self.obs_points = obs_points
        self.sigma = sigma_obs
        self.limit = solver.Re_max

    def log_likelihood(self, theta, y_obs, cells):
        if np.any(theta < 0.01) or np.any(theta > self.limit): return -np.inf
        u_v_mixed, converged = self.solver.solve(theta[0], theta[1], tol=1e-10)
        if not converged: return -np.inf
        # Call the corrected fast observation method
        y_pred = self.solver.get_fast_observations(u_v_mixed, self.obs_points, cells)
        return -0.5 * np.sum(((y_obs - y_pred)/self.sigma)**2)

    def log_prior(self, theta):
        if np.any(theta < 0.01) or np.any(theta > self.limit): return -np.inf
        return 0.0 

    def run_adaptive_mcmc(self, y_obs, cells, n_samples=10000):
        current_theta = np.array([0.5, 0.5]) 
        current_lp = self.log_likelihood(current_theta, y_obs, cells)
        
        scales = np.array([0.02, 0.08]) # Balanced scales from 7.2 validation
        target_acc = 0.25
        samples = []
        accepted_history = []
        
        for i in tqdm(range(n_samples), desc="  Sampling RD", leave=False):
            proposal = current_theta + np.random.normal(0, 1, size=2) * scales
            prop_lp = self.log_likelihood(proposal, y_obs, cells)
            
            if np.log(np.random.rand()) < (prop_lp - current_lp):
                current_theta, current_lp = proposal, prop_lp
                accepted_history.append(1)
            else:
                accepted_history.append(0)
            samples.append(current_theta)

            if (i + 1) % 100 == 0 and i < (n_samples * 0.5):
                recent_acc = np.mean(accepted_history[-100:])
                scales *= np.exp(recent_acc - target_acc)
                
        return np.array(samples)