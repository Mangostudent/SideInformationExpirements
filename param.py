import numpy as np
# Parameter definitions
min_reg = 1e-1
max_reg = 10000
num_reg = 5
k_steps = 4
k_values =   np.linspace(0,1,k_steps)# Correlation parameter values
reg_values = np.concatenate([np.linspace(min_reg, 10, num_reg//2, endpoint =False), np.geomspace(10, max_reg, num_reg - num_reg//2)])  # Regularization strengths
num_xz_samples = 200
num_xy_samples = 2700
random_seed = 42
t = 100
sample_run = 1


