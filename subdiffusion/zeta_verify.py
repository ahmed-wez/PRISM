import numpy as np
import matplotlib.pyplot as plt
from subdiffusion_solver import SubdiffusionSolver1D

def run_zeta_subdiffusion():
    print("Verifying Subdiffusion ZETA Bias (h^2 + tau^(2-beta))")
    h_list = [1/10, 1/20, 1/40, 1/80]
    biases = []
    # Ref solution
    u_ref = SubdiffusionSolver1D(nx=200, nt=200).solve(1.0)
    
    for h in h_list:
        nx = int(1/h)
        u_test = SubdiffusionSolver1D(nx=nx, nt=nx).solve(1.0)
        # Interpolate to compare
        u_test_interp = np.interp(np.linspace(0,1,200), np.linspace(0,1,nx), u_test)
        biases.append(np.linalg.norm(u_test_interp - u_ref))
        
    slope, _ = np.polyfit(np.log(h_list), np.log(biases), 1)
    print(f"Subdiffusion Bias Exponent: {slope:.2f} (Expected near 1.5-2.0)")

if __name__ == "__main__": run_zeta_subdiffusion()