# bayesian_inference.py
import numpy as np
from tqdm import tqdm

class BayesianSubdiffusionInference:
    def __init__(self, solver, x_obs, sigma_obs):
        self.solver = solver
        self.x_obs = x_obs
        self.sigma = sigma_obs

    def log_likelihood(self, q, y_obs):
        # q is the potential parameter
        if q < 0.01 or q > 10.0: return -np.inf
        u_final = self.solver.solve(q)
        # Interpolate numerical solution to sensor locations
        y_pred = np.interp(self.x_obs, self.solver.x, u_final)
        return -0.5 * np.sum(((y_obs - y_pred)/self.sigma)**2)

    def run_adaptive_mcmc(self, y_obs, n_samples=10000):
        current_q = 1.0 # Neutral start
        current_lp = self.log_likelihood(current_q, y_obs)
        scale = 0.1
        target_acc = 0.44
        samples, accepted = [], 0
        
        # Nested bar for trial transparency
        for i in range(n_samples):
            prop = current_q + np.random.normal(0, scale)
            prop_lp = self.log_likelihood(prop, y_obs)
            
            if np.log(np.random.rand()) < (prop_lp - current_lp):
                current_q, current_lp = prop, prop_lp
                accepted += 1
            samples.append(current_q)

            # Robbins-Monro adaptation (first 50% of chain)
            if (i + 1) % 100 == 0 and i < 5000:
                acc_rate = accepted / (i+1)
                scale *= np.exp(acc_rate - target_acc)
                
        return np.array(samples), accepted/n_samples