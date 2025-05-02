import numpy as np
from data import JointDistribution
from model import RegularizedLogisticRegressionModel
from training import trained_models, benchmark_models  # assumes dictionaries are created at import
from param import t, sample_run, k_values, reg_values

results = {}

for k in k_values:
    print(f"Evaluating for k={k}...")
    # Sample data once per k value
    dist = JointDistribution(k)
    X, Y, Z = dist.sample(size=t)
    X = X.reshape(-1, 1)
    XZ = np.column_stack([X, Z])
    Y_pm = Y  # Ensure Y is in {-1, 1}
    
    for reg in reg_values:
        print(f"  Evaluating for reg={reg}...")
        diffs = []
        for _ in range(sample_run):
            # Benchmark model
            bench_model = benchmark_models[(k, reg_values[0])]
            bench_logits = bench_model.intercept_ + bench_model.coef_[0, 0] * X.flatten() + bench_model.coef_[0, 1] * Z.flatten()
            bench_loss = np.mean(np.log(1 + np.exp(-Y * bench_logits)))

            # Trained model
            trained_model = trained_models[(k, reg)]
            n_samples = X.shape[0]
            X_aug = np.hstack([np.ones((n_samples, 1)), X, Z.reshape(-1, 1)])
            logits = X_aug @ trained_model.fit()['w']
            trained_loss = np.mean(np.log(1 + np.exp(-Y * logits)))

            diffs.append(trained_loss - bench_loss)
        results[(k, reg)] = np.mean(diffs)

