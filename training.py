import numpy as np
from data import JointDistribution
from models import RegularizedLogisticRegressionModel, RegularizedLogisticRegressionModelFullInformation
from sklearn.linear_model import LogisticRegression
from param import k_values, reg_values, num_xz_samples, num_xy_samples, random_seed

if random_seed is not None:
    np.random.seed(random_seed)

trained_models = {}
benchmark_models = {}

for k in k_values:
    print(f"Training models for k={k}...")
    
    total_samples = num_xz_samples + num_xy_samples
    dist = JointDistribution(k)
    X, Y, Z = dist.sample(size=total_samples)
    X = X.reshape(-1, 1)
    X_xz = X[:num_xz_samples]
    Z_xz = Z[:num_xz_samples]
    X_xy = X[num_xz_samples:]
    Y_xy = Y[num_xz_samples:]

    # Fit the lower benchmark model (X, Z) -> Y using all data
    XZ = np.column_stack([X, Z])
    lower_benchmark_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
    lower_benchmark_model.fit(XZ, Y)

    # Fit the upper benchmark model (X) -> Y using all data
    upper_benchmark_model = LogisticRegression(fit_intercept=True, solver='lbfgs')
    upper_benchmark_model.fit(X.reshape(-1, 1), Y)

    for reg in reg_values:
        try:
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

        # Fit the regularised benchmark model (X, Z) -> Y using all data
        reg_benchmark_model = RegularizedLogisticRegressionModelFullInformation(reg=reg, sample=(X, Y, Z))
        reg_benchmark_model.fit()

        benchmark_models[(k, reg)] = [lower_benchmark_model, upper_benchmark_model, reg_benchmark_model]

    
