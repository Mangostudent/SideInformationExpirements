import numpy as np
from data import JointDistribution
from model import RegularizedLogisticRegressionModel
from training import trained_models, benchmark_models  # assumes dictionaries are created at import
from param import t, sample_run, k_values, reg_values

results = {}

for k in k_values:
    # Sample data once per k value
    dist = JointDistribution(k)
    X, Y, Z = dist.sample(size=t)
    X = X.reshape(-1, 1)
    XZ = np.column_stack([X, Z])
    
    
    for reg in reg_values:
        diffs = []
        for _ in range(sample_run):
            # Benchmark model
            bench_model = benchmark_models[(k, reg_values[0])]
            bench_logits = XZ @ bench_model.coef_ + bench_model.intercept_
            bench_loss = np.mean(np.log(1 + np.exp(-Y * bench_logits)))

            # Trained model
            trained_model = trained_models[(k, reg)]
            n_samples = X.shape[0]
            X_aug = np.hstack([np.ones((n_samples, 1)), X, Z.reshape(-1, 1)])
            logits = X_aug @ trained_model.fit()['w']
            trained_loss = np.mean(np.log(1 + np.exp(-Y * logits)))

            diffs.append(trained_loss - bench_loss)
        results[(k, reg)] = np.mean(diffs)

