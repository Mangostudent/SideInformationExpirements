import numpy as np
from data import JointDistribution
from models import RegularizedLogisticRegressionModel
import pandas as pd


# --- Concrete training for a range of k and reg values ---

k_values = [0.1, 0.5, 0.9]
reg_values = [0.01, 0.1, 1.0]
num_xz_samples = 100
num_xy_samples = 200
random_seed = 42

if random_seed is not None:
    np.random.seed(random_seed)

trained_models = {}
for k in k_values:
    for reg in reg_values:
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
        trained_models[(k, reg)] = model

# Example: print weights for each (k, reg)
for (k, reg), model in trained_models.items():
    print(f"k={k}, reg={reg}: w={model.w}")

# Display the weights as a 2D table (k as rows, reg as columns) using pandas
weights_data = []
for k in k_values:
    row = []
    for reg in reg_values:
        w = trained_models[(k, reg)].w
        # If w is an array, flatten and take the first element for display
        w_val = w.flatten()[0] if hasattr(w, 'flatten') else w
        row.append(w_val)
    weights_data.append(row)

weights_df = pd.DataFrame(weights_data, index=k_values, columns=reg_values)
print("\nWeights DataFrame (rows: k, columns: reg):")
print(weights_df)
print("Weights table (rows: k, columns: reg):")
header = "k/reg\t" + "\t".join([str(reg) for reg in reg_values])
print(header)
for k in k_values:
    row = [f"{k}"]
    for reg in reg_values:
        w = trained_models[(k, reg)].w
        # If w is an array, flatten and convert to string
        w_str = np.array2string(w.flatten(), precision=3, separator=',')
        row.append(w_str)
    print("\t".join(row))
