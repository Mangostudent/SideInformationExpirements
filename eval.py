import numpy as np
from data import JointDistribution
# Assumes trained_models and benchmark_models dictionaries are populated by running training.py first
from training import trained_models, benchmark_models 
# Import parameters needed for evaluation
from param import t, sample_run, k_values, reg_values, min_reg 

# Dictionaries to store evaluation results
risk = {}       # Stores average difference: trained_loss - lower_benchmark_loss
upperbound = {} # Stores average calculated upper bound value

# Loop over correlation parameters 'k'
for k in k_values:
    print(f"Evaluating for k={k}...")
    
    # Loop over regularization strengths 'reg'
    for reg in reg_values:
        # Lists to store results for each sample run
        diffs_list = [] # Stores (trained_loss - Rwstar) for each run
        expr_list = []  # Stores the calculated upper bound expression for each run

        # Perform multiple sample runs for averaging
        for _ in range(sample_run):
            # --- Sample new data for this evaluation run ---
            dist = JointDistribution(k)
            X, Y, Z = dist.sample(size=t) # Sample 't' data points
            X = X.reshape(-1, 1) # Ensure X is 2D
            XZ = np.column_stack([X, Z]) # Combine X and true Z

            # --- Calculate Benchmark Losses ---
            # Retrieve the non-regularized benchmark models (trained once per k)
            # benchmark_models[(k, reg)][0] is lower_benchmark_model (X,Z -> Y)
            # benchmark_models[(k, reg)][1] is upper_benchmark_model (X -> Y)
            # Note: These benchmarks don't depend on 'reg', so accessing with any reg for the given k is fine.
            # Using reg_values[0] to access the stored benchmarks for clarity.
            if (k, reg_values[0]) not in benchmark_models:
                 print(f"Warning: Benchmark models not found for k={k}. Skipping evaluation for reg={reg}.")
                 continue # Should not happen if training.py ran successfully
            
            lower_bench_model = benchmark_models[(k, reg_values[0])][0] 
            upper_bench_model = benchmark_models[(k, reg_values[0])][1]

            # Rwstar: Loss of the lower benchmark model (optimal with full info X, Z) on the current sample
            lower_bench_logits = lower_bench_model.predict_log_proba(XZ)[:, 1] - lower_bench_model.predict_log_proba(XZ)[:, 0] # More stable logit calculation
            # Rwstar = np.mean(np.log(1 + np.exp(-Y * lower_bench_logits))) # Original calculation
            Rwstar = np.mean(np.log1p(np.exp(-Y * lower_bench_logits))) # Using log1p

            # Lvstar: Loss of the upper benchmark model (optimal with only X) on the current sample
            upper_bench_logits = upper_bench_model.predict_log_proba(X)[:, 1] - upper_bench_model.predict_log_proba(X)[:, 0]
            # Lvstar = np.mean(np.log(1 + np.exp(-Y * upper_bench_logits))) # Original calculation
            Lvstar = np.mean(np.log1p(np.exp(-Y * upper_bench_logits))) # Using log1p

            # --- Calculate Trained Model Loss ---
            if (k, reg) not in trained_models:
                print(f"Warning: Trained model not found for k={k}, reg={reg}. Skipping evaluation.")
                continue # Skip if the specific trained model is missing

            trained_model = trained_models[(k, reg)]
            n_samples = X.shape[0]
            # Prepare augmented features [1, X, Z_true] for evaluating the trained model's weights 'w'
            # Note: The trained model's 'w' expects [1, X, Z_pred], but for loss calculation here,
            # it seems the intention might be to use the true Z with the learned weights. 
            # Revisit this if the goal is to evaluate based on Z_pred. Assuming Z_true for now based on context.
            X_aug = np.hstack([np.ones((n_samples, 1)), X, Z.reshape(-1, 1)]) 
            logits = X_aug @ trained_model.w  # Use the learned weights 'w'
            # trained_loss = np.mean(np.log(1 + np.exp(-Y * logits))) # Original calculation
            trained_loss = np.mean(np.log1p(np.exp(-Y * logits))) # Using log1p

            # Calculate difference: Trained Loss - Lower Benchmark Loss
            diffs_list.append(trained_loss - Rwstar)

            # --- Calculate Upper Bound Expression ---
            # Rustar: Error rate of the intermediate model (X -> Z) on the current sample
            intermediate_trained_model = trained_model.intermediate_model
            Z_pred = intermediate_trained_model.predict(X)
            Rustar = np.mean(Z_pred != Z.flatten()) # Misclassification rate Z_pred vs Z_true

            # Retrieve necessary weights for the bound calculation
            # wstar: Weights of the lower benchmark model (X, Z -> Y)
            wstar = np.hstack([lower_bench_model.intercept_, lower_bench_model.coef_.flatten()])
            
            # wphistarstar: Weights of the two-stage model trained with minimum regularization (min_reg)
            if (k, min_reg) not in trained_models:
                 print(f"Warning: Trained model not found for k={k}, reg={min_reg}. Skipping expr calculation.")
                 wphistarstar = np.zeros_like(wstar) # Placeholder, or handle error
            else:
                 wphistarstar = trained_models[(k, min_reg)].w

            # wregstar: Weights of the regularized benchmark model (X, Z -> Y, reg applied to Z weight)
            if (k, reg) not in benchmark_models or benchmark_models[(k, reg)][2] is None:
                 print(f"Warning: Regularized benchmark model not found for k={k}, reg={reg}. Skipping expr calculation.")
                 wregstar = np.zeros_like(wstar) # Placeholder, or handle error
            else:
                 reg_bench_model_instance = benchmark_models[(k, reg)][2]
                 # Ensure the model was fitted and weights exist
                 if hasattr(reg_bench_model_instance, 'w') and reg_bench_model_instance.w is not None:
                     wregstar = reg_bench_model_instance.w
                 else:
                     print(f"Warning: Regularized benchmark model weights missing for k={k}, reg={reg}. Skipping expr calculation.")
                     wregstar = np.zeros_like(wstar) # Placeholder

            # Calculate the two terms of the upper bound expression
            # Term 1: min(reg * (w*_z)^2, L(v*) - R(w*) + reg * (w_{phi*,min_reg}_z)^2)
            term1 = min(reg * (wstar[-1])**2, Lvstar - Rwstar + reg * (wphistarstar[-1])**2)
            # Term 2: min(log(2)/reg, max(||w_reg*||, ||w*||)) * Ru*
            term2_factor = min(np.log(2) / reg if reg > 0 else np.inf, max(np.linalg.norm(wregstar), np.linalg.norm(wstar)))
            term2 = term2_factor * Rustar
            
            expr_list.append(term1 + term2)

        # Average results over the sample runs
        if diffs_list: # Avoid division by zero if lists are empty due to errors
             risk[(k, reg)] = np.mean(diffs_list)
        else:
             risk[(k, reg)] = np.nan # Indicate missing result

        if expr_list:
             upperbound[(k, reg)] = np.mean(expr_list)
        else:
             upperbound[(k, reg)] = np.nan # Indicate missing result

print("Evaluation complete.")


