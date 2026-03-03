import numpy as np
from tqdm import tqdm

class BayesianNSInference:
    def __init__(self, solver, obs_points, sigma_obs, prior_mean=0.5):
        self.solver = solver
        self.obs_points = obs_points
        self.sigma = sigma_obs
        self.limit = solver.Re_max

    def log_likelihood(self, theta, y_obs, cells):
        if theta < 0.01 or theta > self.limit: return -np.inf
        u, converged = self.solver.solve(theta, tol=1e-10)
        if not converged: return -np.inf
        y_pred = self.solver.get_fast_observations(u, self.obs_points, cells)
        return -0.5 * np.sum(((y_obs - y_pred)/self.sigma)**2)

    def run_adaptive_mcmc(self, y_obs, cells, n_samples=10000):
        current_theta = 0.5 
        current_lp = self.log_likelihood(current_theta, y_obs, cells)
        
        scale = 0.02 
        target_acc = 0.25
        samples = []
        accepted_history = []
        
        # VISIBLE BAR
        for i in tqdm(range(n_samples), desc="  Sampling", leave=False):
            proposal = current_theta + np.random.normal(0, scale)
            prop_lp = self.log_likelihood(proposal, y_obs, cells)
            
            if np.log(np.random.rand()) < (prop_lp - current_lp):
                current_theta, current_lp = proposal, prop_lp
                accepted_history.append(1)
            else:
                accepted_history.append(0)
            samples.append(current_theta)

            if (i + 1) % 100 == 0 and i < (n_samples * 0.5):
                acc_rate = np.mean(accepted_history[-100:])
                scale *= np.exp(acc_rate - target_acc)
                
        return np.array(samples)