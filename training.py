import numpy as np
from data import JointDistribution
from models import RegularizedLogisticRegressionModel, RegularizedLogisticRegressionModelFullInformation
from sklearn.linear_model import LogisticRegression
# Import necessary parameters from param.py
from param import k_values, reg_values, num_xz_samples, num_xy_samples, random_seed, t, sample_run 

# Set random seed for reproducibility if specified
if random_seed is not None:
    np.random.seed(random_seed)

# Dictionaries to store trained models
trained_models = {}     # Stores RegularizedLogisticRegressionModel instances: {(k, reg): model}
benchmark_models = {}   # Stores benchmark models: {(k, reg): [lower_bench, upper_bench, reg_bench]}

# Loop over different correlation parameters 'k'
for k in k_values:
    print(f"Training models for k={k}...")
    
    # Generate a combined dataset for this k
    total_samples = num_xz_samples + num_xy_samples
    dist = JointDistribution(k)
    X, Y, Z = dist.sample(size=total_samples)
    X = X.reshape(-1, 1) # Ensure X is a 2D array [n_samples, 1 feature]

    # Split data for the two-stage model training
    X_xz = X[:num_xz_samples]   # Features for X -> Z training
    Z_xz = Z[:num_xz_samples]   # Labels for X -> Z training
    X_xy = X[num_xz_samples:]   # Features for (X, Z_pred) -> Y training
    Y_xy = Y[num_xz_samples:]   # Labels for (X, Z_pred) -> Y training

    # --- Benchmark Models (trained on the full dataset) ---
    # Lower Benchmark: Model trained on (X, Z) -> Y (uses true Z)
    XZ = np.column_stack([X, Z]) # Combine X and true Z
    lower_benchmark_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
    lower_benchmark_model.fit(XZ, Y)

    # Upper Benchmark: Model trained on X -> Y (ignores Z)
    upper_benchmark_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
    upper_benchmark_model.fit(X, Y) # Fit using only X

    # --- Train Models for each Regularization Strength ---
    for reg in reg_values:
        # Train the two-stage regularized model
        try:
            model = RegularizedLogisticRegressionModel(
                reg=reg,
                sample_xz=(X_xz, Z_xz),
                sample_xy=(X_xy, Y_xy)
            )
            model.fit() # Fit the two-stage model
            trained_models[(k, reg)] = model # Store the fitted model
        except Exception as e:
            # Log errors during model fitting
            print(f"Error fitting two-stage model for k={k}, reg={reg}: {str(e)}")
            continue # Skip to the next regularization value if fitting fails

        # Train the regularized benchmark model (full information)
        # This model uses (X, Y, Z) and applies regularization 'reg' to the Z weight
        try:
            reg_benchmark_model = RegularizedLogisticRegressionModelFullInformation(
                reg=reg, 
                sample=(X, Y, Z) # Uses the complete dataset
            )
            reg_benchmark_model.fit() # Fit the full information regularized model
        except Exception as e:
             # Log errors during model fitting
            print(f"Error fitting regularized benchmark model for k={k}, reg={reg}: {str(e)}")
            # If this fails, we might still want to store the other benchmarks
            reg_benchmark_model = None # Indicate failure

        # Store all benchmark models together for this (k, reg) pair
        # Note: lower_benchmark and upper_benchmark are the same for all 'reg' within a 'k'
        benchmark_models[(k, reg)] = [lower_benchmark_model, upper_benchmark_model, reg_benchmark_model]

print("Training complete.")

    
