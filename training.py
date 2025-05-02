import numpy as np
from data import JointDistribution
from model import RegularizedLogisticRegressionModel
from sklearn.linear_model import LogisticRegression
from param import k_values, reg_values, num_xz_samples, num_xy_samples, random_seed

if random_seed is not None:
    np.random.seed(random_seed)

trained_models = {}
benchmark_models = {}

for k in k_values:
    for reg in reg_values:
        try:
            total_samples = num_xz_samples + num_xy_samples
            dist = JointDistribution(k)
            X, Y, Z = dist.sample(size=total_samples)
            X = X.reshape(-1, 1)
            X_xz = X[:num_xz_samples]
            Z_xz = Z[:num_xz_samples]
            X_xy = X[num_xz_samples:]
            Y_xy = Y[num_xz_samples:]
            model = RegularizedLogisticRegressionModel(
                reg=reg,
                sample_xz=(X_xz, Z_xz),
                sample_xy=(X_xy, Y_xy)
            )
            model.fit()
        except Exception as e:
            print(f"Error fitting model for k={k}, reg={reg}: {str(e)}")
            continue
        trained_models[(k, reg)] = model

    # Fit the benchmark model (X, Z) -> Y using all data
    XZ = np.column_stack([X, Z])
    benchmark_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
    benchmark_model.fit(XZ, Y)
    # Store the same model for all reg values for this k
    for reg in reg_values:
        benchmark_models[(k, reg)] = benchmark_model
