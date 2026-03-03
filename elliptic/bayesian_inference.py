import numpy as np
from typing import Tuple

class BayesianPoissonInference:
    def __init__(self, x_obs: np.ndarray, sigma_obs: float, tau_prior: float = 1.0):
        self.x_obs = x_obs
        self.sigma_obs = sigma_obs
        self.tau_prior = tau_prior
        
    def get_H(self) -> np.ndarray:
        """Observation operator for sine-problem: u(x) = theta * sin(pi*x)"""
        return np.sin(np.pi * self.x_obs)

    def gaussian_posterior(self, y: np.ndarray) -> Tuple[float, float]:
        H = self.get_H()
        prior_prec = 1.0 / (self.tau_prior**2)
        like_prec = np.sum(H**2) / (self.sigma_obs**2)
        
        post_var = 1.0 / (prior_prec + like_prec)
        post_mean = post_var * (np.sum(H * y) / (self.sigma_obs**2))
        return post_mean, np.sqrt(post_var)