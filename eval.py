import numpy as np
from data import JointDistribution
from models import RegularizedLogisticRegressionModel
from training import trained_models, benchmark_models  # assumes dictionaries are created at import

def logistic_loss_pm1(y_true, y_pred_prob):
    # y_true in {-1, +1}, y_pred_prob is probability of y=+1
    # logistic loss: mean(log(1 + exp(-y * f(x))))
    # f(x) = logit = log(p/(1-p))
    # y_pred_prob should be clipped to avoid log(0)
    eps = 1e-12
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    logits = np.log(y_pred_prob / (1 - y_pred_prob))
    return np.mean(np.log(1 + np.exp(-y_true * logits)))

# Set sample size for evaluation
t = 1000  # or any value you want
sample_run = 10  # set this to however many runs you want

k_values = sorted(set(k for k, reg in trained_models.keys()))
reg_values = sorted(set(reg for k, reg in trained_models.keys()))

results = {}

for k in k_values:
    for reg in reg_values:
        diffs = []
        for _ in range(sample_run):
            # Sample t points from the joint distribution for this k
            dist = JointDistribution(k)
            X, Y, Z = dist.sample(size=t)
            X = X.reshape(-1, 1)
            XZ = np.column_stack([X, Z])

            # Ensure Y is in {-1, 1}
            Y_pm = Y

            # Benchmark model: predict probabilities for logistic loss
            bench_model = benchmark_models[(k, reg_values[0])]  # reg doesn't matter for benchmark
            bench_probs = bench_model.predict_proba(XZ)[:, 1]
            bench_loss = logistic_loss_pm1(Y_pm, bench_probs)

            # Trained model: need to predict Z using the intermediate model
            trained_model = trained_models[(k, reg)]
            Z_pred_proba = trained_model.intermediate_model.predict_proba(X)[:, 1]
            Z_pred = np.where(Z_pred_proba >= 0.5, 1, -1)
            n_samples = X.shape[0]
            X_aug = np.hstack([np.ones((n_samples, 1)), X, Z_pred.reshape(-1, 1)])
            logits = X_aug @ trained_model.w
            trained_probs = 1 / (1 + np.exp(-logits))
            trained_loss = logistic_loss_pm1(Y_pm, trained_probs)

            # Store the difference (trained_loss - bench_loss)
            diffs.append(trained_loss - bench_loss)
        results[(k, reg)] = np.mean(diffs)

print("Mean difference in logistic loss (trained - benchmark) over all runs:")
for key, value in results.items():
    print(f"k={key[0]}, reg={key[1]}: {value:.4f}")