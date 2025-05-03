import numpy as np

# --- Simulation Parameters ---
min_reg = 1e-1          # Minimum regularization strength to test
max_reg = 10000         # Maximum regularization strength to test
num_reg = 16            # Number of regularization values to generate between min and max
k_steps = 16            # Number of correlation parameter values (k) to test
k_values = np.linspace(0, 1, k_steps) # Correlation parameter 'k' values (controls P(Y,Z))
# Generate regularization strengths: linearly spaced up to 10, then geometrically spaced
reg_values = np.concatenate([
    np.linspace(min_reg, 10, num_reg // 2, endpoint=False), 
    np.geomspace(10, max_reg, num_reg - num_reg // 2)
])

# --- Data Sampling Parameters ---
num_xz_samples = 200    # Number of samples (X, Z) for training the intermediate model X -> Z
num_xy_samples = 2700   # Number of samples (X, Y) for training the final model (X, Z_pred) -> Y
t = 100                 # Number of samples used in each evaluation run within eval.py
sample_run = 4          # Number of independent evaluation runs in eval.py

# --- Reproducibility ---
random_seed = 42        # Seed for numpy's random number generator for reproducibility


