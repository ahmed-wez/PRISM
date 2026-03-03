import numpy as np
import matplotlib.pyplot as plt
import os
from elliptic_solver import BFEMSolver1D

def run_zeta_verification():
    print("\n" + "="*60)
    print("ELLIPTIC: ZETA CONDITION VERIFICATION (C4a, C4b)")
    print("="*60)
    
    h_list = [1/10, 1/20, 1/40, 1/80, 1/160]
    biases = []
    variances = []
    
    for h in h_list:
        n_elements = int(1/h)
        # Standard theta for testing bias
        theta_test = 1.0
        solver = BFEMSolver1D(n_elements)
        
        bias, var = solver.compute_bias_variance(theta_test)
        biases.append(bias)
        variances.append(var)
        
    # Fit the exponents
    b_slope = np.polyfit(np.log(h_list), np.log(biases), 1)[0]
    v_slope = np.polyfit(np.log(h_list), np.log(variances), 1)[0]
    
    plt.figure(figsize=(8, 6))
    plt.loglog(h_list, biases, 'b-o', label=f'Bias (Slope: {b_slope:.2f})', linewidth=2)
    plt.loglog(h_list, variances, 'r-s', label=f'PN Var (Slope: {v_slope:.2f})', linewidth=2)
    plt.xlabel('Mesh size h')
    plt.ylabel('Error Magnitude')
    plt.title('Elliptic ZETA Verification: C4a & C4b')
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    
    if not os.path.exists('../figures'): os.makedirs('../figures')
    plt.savefig('../figures/elliptic_zeta_verify.png', dpi=300)
    
    print(f"Bias Rate (C4a): h^{b_slope:.2f}")
    print(f"Var Rate (C4b):  h^{v_slope:.2f}")
    print("Success: Slopes should be near 2.0")

if __name__ == "__main__":
    run_zeta_verification()